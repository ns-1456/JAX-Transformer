"""Text generation with KV cache, MLA, and DSA support. Sampling and KV cache."""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from einops import einsum, rearrange

from model import (
    TransformerConfig,
    TransformerLM,
    RMSNorm,
    apply_rope,
    precompute_rope_freqs,
)
from tokenizer import Tokenizer


@dataclass
class KVCache:
    """Standard KV cache: stores full K, V per layer per head."""
    keys: list[jax.Array]    # [num_layers] each (batch, num_heads, cached_len, d_head)
    values: list[jax.Array]  # same

    @classmethod
    def empty(cls, num_layers: int) -> KVCache:
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    def update(self, layer: int, k: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
        if self.keys[layer] is None:
            self.keys[layer] = k
            self.values[layer] = v
        else:
            self.keys[layer] = jnp.concatenate([self.keys[layer], k], axis=2)
            self.values[layer] = jnp.concatenate([self.values[layer], v], axis=2)
        return self.keys[layer], self.values[layer]


@dataclass
class MLACache:
    """MLA cache: stores compressed latent c_kv per layer instead of full K, V."""
    latents: list[jax.Array | None]  # [num_layers] each (batch, cached_len, d_latent)

    @classmethod
    def empty(cls, num_layers: int) -> MLACache:
        return cls(latents=[None] * num_layers)

    def update(self, layer: int, c_kv: jax.Array) -> jax.Array:
        if self.latents[layer] is None:
            self.latents[layer] = c_kv
        else:
            self.latents[layer] = jnp.concatenate([self.latents[layer], c_kv], axis=1)
        return self.latents[layer]


def _extract_layer_params(params: dict, layer_idx: int) -> dict:
    return params["params"][f"block_{layer_idx}"]["MultiHeadAttention_0"]


def _single_layer_attn_with_kv_cache(
    x: jax.Array,
    layer_params: dict,
    rope_freqs: jax.Array,
    cache: KVCache,
    layer_idx: int,
    cfg: TransformerConfig,
    pos: int,
) -> jax.Array:
    """Run one layer's attention with standard KV cache."""
    d_model, num_heads, d_head = cfg.d_model, cfg.num_heads, cfg.d_head

    q = x @ layer_params["W_q"]
    k = x @ layer_params["W_k"]
    v = x @ layer_params["W_v"]

    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

    pos_freqs = rope_freqs[pos : pos + 1]
    q = apply_rope(q, pos_freqs)
    k = apply_rope(k, pos_freqs)

    k, v = cache.update(layer_idx, k, v)

    scale = d_head ** -0.5
    scores = einsum(q, k, "b h sq d, b h sk d -> b h sq sk") * scale

    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = einsum(attn_weights, v, "b h sq sk, b h sk d -> b h sq d")
    out = rearrange(out, "b h s d -> b s (h d)")
    return out @ layer_params["W_o"]


def _single_layer_attn_with_mla(
    x: jax.Array,
    layer_params: dict,
    mla_params: dict,
    rope_freqs: jax.Array,
    cache: MLACache,
    layer_idx: int,
    cfg: TransformerConfig,
    pos: int,
    sparse_top_k: int | None = None,
) -> jax.Array:
    """Run one layer's attention with MLA compressed KV cache + optional DSA."""
    d_model, num_heads, d_head = cfg.d_model, cfg.num_heads, cfg.d_head
    d_latent = mla_params["d_latent"]

    q = x @ layer_params["W_q"]
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)

    pos_freqs = rope_freqs[pos : pos + 1]
    q = apply_rope(q, pos_freqs)

    c_kv = x @ mla_params["W_dkv"]
    c_kv = cache.update(layer_idx, c_kv)

    all_k = []
    all_v = []
    for h in range(num_heads):
        k_h = c_kv @ mla_params[f"W_uk_{h}"]
        v_h = c_kv @ mla_params[f"W_uv_{h}"]
        all_k.append(k_h)
        all_v.append(v_h)

    k = jnp.stack(all_k, axis=1)  # (batch, heads, seq, d_head)
    v = jnp.stack(all_v, axis=1)

    k_freqs = rope_freqs[: k.shape[2]]
    k = apply_rope(k, k_freqs)

    scale = d_head ** -0.5
    scores = einsum(q, k, "b h sq d, b h sk d -> b h sq sk") * scale

    if sparse_top_k is not None and scores.shape[-1] > sparse_top_k:
        top_vals, top_idx = jax.lax.top_k(scores, sparse_top_k)
        mask = jnp.zeros_like(scores).at[
            jnp.arange(scores.shape[0])[:, None, None, None],
            jnp.arange(scores.shape[1])[None, :, None, None],
            jnp.arange(scores.shape[2])[None, None, :, None],
            top_idx,
        ].set(1.0)
        scores = jnp.where(mask > 0, scores, jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = einsum(attn_weights, v, "b h sq sk, b h sk d -> b h sq d")
    out = rearrange(out, "b h s d -> b s (h d)")
    return out @ layer_params["W_o"]


def init_mla_params(
    key: jax.Array, cfg: TransformerConfig, d_latent: int = 64
) -> list[dict]:
    """Initialize MLA projection parameters for each layer."""
    mla_params = []
    for i in range(cfg.num_layers):
        layer_key = jax.random.fold_in(key, i)
        keys = jax.random.split(layer_key, 2 + 2 * cfg.num_heads)
        lp = {
            "d_latent": d_latent,
            "W_dkv": jax.random.normal(keys[0], (cfg.d_model, d_latent)) * 0.02,
        }
        for h in range(cfg.num_heads):
            lp[f"W_uk_{h}"] = jax.random.normal(keys[1 + h], (d_latent, cfg.d_head)) * 0.02
            lp[f"W_uv_{h}"] = (
                jax.random.normal(keys[1 + cfg.num_heads + h], (d_latent, cfg.d_head)) * 0.02
            )
        mla_params.append(lp)
    return mla_params


def generate(
    model: TransformerLM,
    params: Any,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 0,
    use_mla: bool = False,
    mla_params: list[dict] | None = None,
    sparse_top_k: int | None = None,
) -> str:
    cfg = model.config
    token_ids = tokenizer.encode(prompt)
    input_ids = jnp.array([token_ids], dtype=jnp.int32)

    rope_freqs = precompute_rope_freqs(cfg.d_head, cfg.max_seq_len)
    rng_key = jax.random.PRNGKey(seed)

    logits = model.apply(params, input_ids, deterministic=True)

    if use_mla and mla_params is not None:
        cache = MLACache.empty(cfg.num_layers)
        # Warm the MLA cache with the full prompt
        embedding = params["params"]["embedding"]
        x_full = embedding[input_ids]
        for layer_idx in range(cfg.num_layers):
            lp = _extract_layer_params(params, layer_idx)
            mlp = mla_params[layer_idx]
            c_kv = x_full @ mlp["W_dkv"]
            cache.update(layer_idx, c_kv[:, :, :])  # seed entire prompt
    else:
        cache = KVCache.empty(cfg.num_layers)
        embedding = params["params"]["embedding"]
        x_full = embedding[input_ids]
        for layer_idx in range(cfg.num_layers):
            lp = _extract_layer_params(params, layer_idx)
            k_full = x_full @ lp["W_k"]
            v_full = x_full @ lp["W_v"]
            k_full = rearrange(k_full, "b s (h d) -> b h s d", h=cfg.num_heads)
            v_full = rearrange(v_full, "b s (h d) -> b h s d", h=cfg.num_heads)
            k_full = apply_rope(k_full, rope_freqs[:len(token_ids)])
            cache.keys[layer_idx] = k_full
            cache.values[layer_idx] = v_full

    generated = list(token_ids)

    for step in range(max_tokens):
        next_logits = logits[:, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            top_values, _ = jax.lax.top_k(next_logits, top_k)
            threshold = top_values[:, -1:]
            next_logits = jnp.where(next_logits < threshold, jnp.finfo(next_logits.dtype).min, next_logits)

        rng_key, sample_key = jax.random.split(rng_key)
        next_token = jax.random.categorical(sample_key, next_logits, axis=-1)
        next_token_id = int(next_token[0])
        generated.append(next_token_id)

        next_input = jnp.array([[next_token_id]], dtype=jnp.int32)
        pos = len(generated) - 1

        embedding = params["params"]["embedding"]
        x = embedding[next_input]

        for layer_idx in range(cfg.num_layers):
            block_key = f"block_{layer_idx}"
            block_params = params["params"][block_key]

            norm1_scale = block_params["RMSNorm_0"]["scale"]
            rms1 = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
            h = norm1_scale * (x / rms1)

            attn_params = block_params["MultiHeadAttention_0"]
            if use_mla and mla_params is not None:
                attn_out = _single_layer_attn_with_mla(
                    h, attn_params, mla_params[layer_idx], rope_freqs,
                    cache, layer_idx, cfg, pos, sparse_top_k=sparse_top_k,
                )
            else:
                attn_out = _single_layer_attn_with_kv_cache(
                    h, attn_params, rope_freqs, cache, layer_idx, cfg, pos,
                )
            x = x + attn_out

            norm2_scale = block_params["RMSNorm_1"]["scale"]
            rms2 = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
            h = norm2_scale * (x / rms2)

            ffn_params = block_params["SwiGLUFFN_0"]
            gate = jax.nn.silu(h @ ffn_params["W1"])
            up = h @ ffn_params["W3"]
            ffn_out = (gate * up) @ ffn_params["W2"]
            x = x + ffn_out

        final_norm_scale = params["params"]["final_norm"]["scale"]
        rms_f = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        x = final_norm_scale * (x / rms_f)
        logits = x @ embedding.T

    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from trained LM")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer .json")
    parser.add_argument("--prompt", default="The meaning of life is", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--use-mla", action="store_true", help="Use MLA+DSA inference")
    parser.add_argument("--sparse-top-k", type=int, default=None, help="DSA sparse top-k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    cfg = ckpt["config"]

    tokenizer = Tokenizer.load(args.tokenizer)
    model = TransformerLM(cfg)

    mla_params = None
    if args.use_mla:
        mla_params = init_mla_params(jax.random.PRNGKey(0), cfg)

    output = generate(
        model, params, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        use_mla=args.use_mla,
        mla_params=mla_params,
        sparse_top_k=args.sparse_top_k,
    )
    print(output)


if __name__ == "__main__":
    main()
