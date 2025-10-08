/*
 * Fused scaled dot-product attention CUDA kernel.
 *
 * Fuses Q@K^T scaling, causal masking, softmax, and @V into a single kernel
 * with no intermediate matrices in global memory.
 *
 * Input shapes: Q, K, V are (batch, seq_len, d_head)
 * Output shape: O is (batch, seq_len, d_head)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void fused_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int seq_len,
    int d_head,
    float scale
) {
    int batch_idx = blockIdx.x;
    int query_pos = blockIdx.y;

    if (batch_idx >= batch_size || query_pos >= seq_len) return;

    int tid = threadIdx.x;

    int base_q = (batch_idx * seq_len + query_pos) * d_head;

    extern __shared__ float smem[];
    float* s_query = smem;

    for (int d = tid; d < d_head; d += blockDim.x) {
        s_query[d] = Q[base_q + d];
    }
    __syncthreads();

    /* ---- Pass 1: compute max score for numerical stability ---- */
    float local_max = -FLT_MAX;
    for (int key_pos = tid; key_pos <= query_pos; key_pos += blockDim.x) {
        int base_k = (batch_idx * seq_len + key_pos) * d_head;
        float dot = 0.0f;
        for (int d = 0; d < d_head; d++) {
            dot += s_query[d] * K[base_k + d];
        }
        dot *= scale;
        if (dot > local_max) local_max = dot;
    }

    /* Block-wide max reduction via shared memory */
    float* s_reduce = smem + d_head;
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_reduce[tid + stride] > s_reduce[tid]) {
            s_reduce[tid] = s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float global_max = s_reduce[0];

    /* ---- Pass 2: compute exp-sum ---- */
    float local_sum = 0.0f;
    for (int key_pos = tid; key_pos <= query_pos; key_pos += blockDim.x) {
        int base_k = (batch_idx * seq_len + key_pos) * d_head;
        float dot = 0.0f;
        for (int d = 0; d < d_head; d++) {
            dot += s_query[d] * K[base_k + d];
        }
        dot = expf(dot * scale - global_max);
        local_sum += dot;
    }

    s_reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = s_reduce[0];

    /* ---- Pass 3: compute weighted sum of V ---- */
    int out_base = (batch_idx * seq_len + query_pos) * d_head;

    for (int d = tid; d < d_head; d += blockDim.x) {
        float acc = 0.0f;
        for (int key_pos = 0; key_pos <= query_pos; key_pos++) {
            int base_k = (batch_idx * seq_len + key_pos) * d_head;
            float dot = 0.0f;
            for (int dd = 0; dd < d_head; dd++) {
                dot += s_query[dd] * K[base_k + dd];
            }
            float w = expf(dot * scale - global_max) / global_sum;

            int base_v = (batch_idx * seq_len + key_pos) * d_head;
            acc += w * V[base_v + d];
        }
        O[out_base + d] = acc;
    }
}


torch::Tensor fused_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dim() == 3, "Q must be (batch, seq_len, d_head)");

    int batch_size = Q.size(0);
    int seq_len = Q.size(1);
    int d_head = Q.size(2);
    float scale = 1.0f / sqrtf((float)d_head);

    auto O = torch::zeros_like(Q);

    int threads = min(256, seq_len);
    /* pad to next power of 2 for reductions */
    int threads_pow2 = 1;
    while (threads_pow2 < threads) threads_pow2 <<= 1;
    threads = threads_pow2;

    dim3 grid(batch_size, seq_len);
    size_t smem_bytes = (d_head + threads) * sizeof(float);

    fused_attention_forward_kernel<<<grid, threads, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size, seq_len, d_head, scale
    );

    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_forward", &fused_attention_forward,
          "Fused scaled dot-product attention (causal) forward pass");
}
