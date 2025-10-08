"""Benchmark: CUDA fused attention vs. naive PyTorch vs. torch.compile."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_head = Q.size(-1)
    scale = d_head ** -0.5
    scores = (Q @ K.transpose(-2, -1)) * scale
    seq_len = Q.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V


@torch.compile
def compiled_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_head = Q.size(-1)
    scale = d_head ** -0.5
    scores = (Q @ K.transpose(-2, -1)) * scale
    seq_len = Q.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def bench(fn, Q, K, V, warmup: int = 10, repeats: int = 50) -> float:
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn(Q, K, V)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000  # ms


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    try:
        import fused_attention
    except ImportError:
        print("Build the CUDA extension first: cd cuda && pip install -e .")
        return

    device = torch.device("cuda")
    batch_size = 4
    d_head = 64
    seq_lengths = [128, 256, 512, 1024, 2048]

    header = f"{'Seq Len':>8} | {'Naive (ms)':>11} | {'Compiled (ms)':>14} | {'CUDA Fused (ms)':>16}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        Q = torch.randn(batch_size, seq_len, d_head, device=device)
        K = torch.randn(batch_size, seq_len, d_head, device=device)
        V = torch.randn(batch_size, seq_len, d_head, device=device)

        t_naive = bench(naive_attention, Q, K, V)
        t_compiled = bench(compiled_attention, Q, K, V)
        t_fused = bench(fused_attention.fused_attention_forward, Q, K, V)

        print(f"{seq_len:>8} | {t_naive:>11.3f} | {t_compiled:>14.3f} | {t_fused:>16.3f}")


if __name__ == "__main__":
    main()
