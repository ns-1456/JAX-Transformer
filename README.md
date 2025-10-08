# Transformer Language Model from Scratch (JAX/XLA + CUDA)

A complete transformer language model implemented entirely from scratch—byte-level BPE tokenizer, pre-norm Transformer with RoPE and SwiGLU in JAX, hand-written AdamW optimizer, and a fused CUDA attention kernel. Built as a portfolio project following the Stanford CS336 curriculum.

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│  BPE Tokenizer      │  Byte-level, GPT-2 pre-tokenization
│  (regex + merges)   │
└────────┬────────────┘
         │ token IDs
         ▼
┌─────────────────────┐
│  Token Embedding    │  vocab_size × d_model
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Transformer Block  ×N             │
│  ┌───────────────────────────────┐ │
│  │ RMSNorm                       │ │
│  │ Multi-Head Attention (RoPE)   │ │
│  │   Q, K, V projections        │ │
│  │   Rotary position embeddings │ │
│  │   Causal scaled dot-product  │ │
│  │ + Residual                    │ │
│  ├───────────────────────────────┤ │
│  │ RMSNorm                       │ │
│  │ SwiGLU FFN                    │ │
│  │   SiLU(W1·x) ⊙ (W3·x)      │ │
│  │   W2 · (...)                 │ │
│  │ + Residual                    │ │
│  └───────────────────────────────┘ │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  RMSNorm            │
│  LM Head (tied)     │  Shares embedding weights
└────────┬────────────┘
         │ logits
         ▼
      Output
```

## Features

- **Byte-level BPE tokenizer** with GPT-2 regex pre-tokenization, parallel training, and incremental pair-count merging
- **JAX/XLA Transformer** with RMSNorm, Rotary Position Embeddings, SwiGLU FFN, and weight tying
- **AdamW optimizer** from scratch using JAX pytrees with bias correction and decoupled weight decay
- **KV-cache generation** with two inference paths:
  - Standard KV cache (full K, V per layer)
  - MLA (Multi-head Latent Attention) — compressed KV cache via shared low-rank latent
  - DSA (DeepSeek Sparse Attention) — top-k sparsification of attention scores
- **CUDA fused attention kernel** — single-pass Q@K^T + scale + causal mask + softmax + @V with no intermediate global memory

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Demo notebook (train + generate + plots)

**Local:** From the repo root, open `notebooks/demo_train_generate.ipynb` and run all cells (kernel cwd = repo root).

**Google Colab:** Open [the notebook on Colab](https://colab.research.google.com/github/ns-1456/JAX-XLA-LM/blob/main/notebooks/demo_train_generate.ipynb). The first cell clones this repo and installs dependencies; then run all cells to train a tiny BPE tokenizer and model, plot loss, and generate with KV cache and MLA+DSA.

### Train tokenizer

```python
from src.tokenizer import train_bpe, Tokenizer

vocab, merges = train_bpe("data/corpus.txt", vocab_size=4096, special_tokens=["<|endoftext|>"])
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
tokenizer.save("tokenizer.json")
```

### Train model

```bash
# Tiny config for quick experiments
python src/train.py --data data/tokens.npy --config tiny --num-steps 5000

# Small config
python src/train.py --data data/corpus.txt --tokenizer tokenizer.json --config small
```

### Generate text

```bash
python src/generate.py \
    --checkpoint checkpoints/step_5000.pkl \
    --tokenizer tokenizer.json \
    --prompt "The meaning of life is" \
    --max-tokens 200 \
    --temperature 0.8

# With MLA + DSA inference
python src/generate.py \
    --checkpoint checkpoints/step_5000.pkl \
    --tokenizer tokenizer.json \
    --prompt "Once upon a time" \
    --use-mla --sparse-top-k 64
```

### Build and benchmark CUDA kernel

```bash
cd cuda
pip install -e .
python benchmark_cuda.py
```

## Project Structure

```
project-2-lm-jax/
├── src/
│   ├── tokenizer.py      Byte-level BPE tokenizer (train + encode/decode)
│   ├── model.py           Transformer LM in JAX (RMSNorm, RoPE, SwiGLU, MHA)
│   ├── optimizer.py       AdamW optimizer from scratch
│   ├── train.py           Training loop with checkpointing and logging
│   └── generate.py        Text generation with KV cache / MLA / DSA
├── cuda/
│   ├── fused_attention.cu Fused scaled dot-product attention CUDA kernel
│   ├── setup.py           PyTorch C++ extension build script
│   └── benchmark_cuda.py  Benchmark: CUDA vs naive PyTorch vs torch.compile
├── requirements.txt
└── README.md
```

## Model Configurations

| Config | d_model | Layers | Heads | d_ff  | Params (approx) |
|--------|---------|--------|-------|-------|------------------|
| Tiny   | 128     | 4      | 4     | 512   | ~2M              |
| Small  | 768     | 12     | 12    | 3072  | ~85M             |

## Results

| Metric              | Tiny (4L-128D) | Small (12L-768D) |
|---------------------|----------------|------------------|
| Training loss       | —              | —                |
| Validation perplexity | —            | —                |
| Tokens/sec (CPU)    | —              | —                |
| Tokens/sec (GPU)    | —              | —                |

*Fill in after training on your dataset.*

## References

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
