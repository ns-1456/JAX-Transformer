"""Transformer LM in JAX: RoPE, SwiGLU, RMSNorm, tied LM head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import einsum, rearrange


@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout_rate: float = 0.1

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads


def _trunc_normal_init(key: jax.Array, shape: tuple[int, ...], std: float) -> jax.Array:
    return jax.random.truncated_normal(key, -2.0, 2.0, shape) * std


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        g = self.param("scale", nn.initializers.ones, (self.d_model,))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return g * (x / rms)


def precompute_rope_freqs(d_head: int, max_seq_len: int, base: float = 10000.0) -> jax.Array:
    """Returns complex rotation frequencies of shape (max_seq_len, d_head // 2)."""
    dim_pairs = jnp.arange(0, d_head, 2, dtype=jnp.float32)
    theta = 1.0 / (base ** (dim_pairs / d_head))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = einsum(positions, theta, "seq, d -> seq d")
    return angles


def apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """Apply rotary embeddings to x of shape (..., seq_len, d_head)."""
    seq_len = x.shape[-2]
    freqs = freqs[:seq_len]
    x_pairs = rearrange(x, "... s (d two) -> ... s d two", two=2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    cos_f = jnp.cos(freqs)
    sin_f = jnp.sin(freqs)
    y0 = x0 * cos_f - x1 * sin_f
    y1 = x0 * sin_f + x1 * cos_f
    out = jnp.stack([y0, y1], axis=-1)
    return rearrange(out, "... s d two -> ... s (d two)", two=2)


class MultiHeadAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self, x: jax.Array, rope_freqs: jax.Array, deterministic: bool = True
    ) -> jax.Array:
        cfg = self.config
        d_model, num_heads, d_head = cfg.d_model, cfg.num_heads, cfg.d_head

        def _linear(name: str, d_out: int) -> jax.Array:
            std = (2.0 / (d_model + d_out)) ** 0.5
            w = self.param(
                name,
                lambda rng, s: _trunc_normal_init(rng, s, std),
                (d_model, d_out),
            )
            return x @ w

        qkv_dim = num_heads * d_head
        q = _linear("W_q", qkv_dim)
        k = _linear("W_k", qkv_dim)
        v = _linear("W_v", qkv_dim)

        q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        scale = d_head ** -0.5
        attn_scores = einsum(q, k, "b h sq d, b h sk d -> b h sq sk") * scale

        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn_scores = jnp.where(causal_mask, attn_scores, jnp.finfo(attn_scores.dtype).min)

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        if not deterministic:
            attn_weights = nn.Dropout(rate=cfg.dropout_rate)(attn_weights, deterministic=False)

        attn_out = einsum(attn_weights, v, "b h sq sk, b h sk d -> b h sq d")
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        std_out = (2.0 / (qkv_dim + d_model)) ** 0.5
        w_o = self.param(
            "W_o",
            lambda rng, s: _trunc_normal_init(rng, s, std_out),
            (qkv_dim, d_model),
        )
        return attn_out @ w_o


class SwiGLUFFN(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        cfg = self.config
        d_model, d_ff = cfg.d_model, cfg.d_ff

        def _make_w(name: str, shape: tuple[int, int]) -> jax.Array:
            std = (2.0 / (shape[0] + shape[1])) ** 0.5
            return self.param(name, lambda rng, s: _trunc_normal_init(rng, s, std), shape)

        w1 = _make_w("W1", (d_model, d_ff))
        w2 = _make_w("W2", (d_ff, d_model))
        w3 = _make_w("W3", (d_model, d_ff))

        gate = jax.nn.silu(x @ w1)
        up = x @ w3
        out = (gate * up) @ w2

        if not deterministic:
            out = nn.Dropout(rate=cfg.dropout_rate)(out, deterministic=False)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self, x: jax.Array, rope_freqs: jax.Array, deterministic: bool = True
    ) -> jax.Array:
        cfg = self.config
        h = RMSNorm(cfg.d_model)(x)
        h = MultiHeadAttention(cfg)(h, rope_freqs, deterministic=deterministic)
        x = x + h
        h = RMSNorm(cfg.d_model)(x)
        h = SwiGLUFFN(cfg)(h, deterministic=deterministic)
        x = x + h
        return x


class TransformerLM(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self, token_ids: jax.Array, deterministic: bool = True
    ) -> jax.Array:
        cfg = self.config

        embedding = self.param(
            "embedding",
            lambda rng, s: _trunc_normal_init(rng, s, 1.0),
            (cfg.vocab_size, cfg.d_model),
        )

        x = embedding[token_ids]
        rope_freqs = precompute_rope_freqs(cfg.d_head, cfg.max_seq_len)

        for i in range(cfg.num_layers):
            x = TransformerBlock(cfg, name=f"block_{i}")(
                x, rope_freqs, deterministic=deterministic
            )

        x = RMSNorm(cfg.d_model, name="final_norm")(x)
        logits = x @ embedding.T
        return logits


def create_model(config: TransformerConfig) -> TransformerLM:
    return TransformerLM(config)


@jax.jit
def forward(model: TransformerLM, params: Any, token_ids: jax.Array) -> jax.Array:
    return model.apply(params, token_ids, deterministic=True)


TINY_CONFIG = TransformerConfig(
    vocab_size=4096, d_model=128, num_layers=4, num_heads=4,
    d_ff=512, max_seq_len=256, dropout_rate=0.1,
)

SMALL_CONFIG = TransformerConfig(
    vocab_size=10000, d_model=768, num_layers=12, num_heads=12,
    d_ff=3072, max_seq_len=512, dropout_rate=0.1,
)
