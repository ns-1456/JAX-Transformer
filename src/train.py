"""Training loop for the Transformer LM. Checkpointing and logging."""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from model import TransformerConfig, TransformerLM, TINY_CONFIG, SMALL_CONFIG
from optimizer import AdamWState, adamw_step, init_adamw
from tokenizer import Tokenizer


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Shifted cross-entropy: logits[:, :-1] predicts targets[:, 1:]."""
    logits = logits[:, :-1]
    targets = targets[:, 1:]

    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def loss_fn(
    params: Any, model: TransformerLM, batch: jax.Array
) -> jax.Array:
    logits = model.apply(params, batch, deterministic=True)
    return cross_entropy_loss(logits, batch)


def get_batch(
    data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> np.ndarray:
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=(batch_size,))
    return np.stack([data[s : s + seq_len] for s in starts])


def save_checkpoint(
    path: str, params: Any, opt_state: AdamWState, step: int, config: TransformerConfig
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "opt_state": opt_state, "step": step, "config": config}, f)


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(data_path: str, tokenizer_path: str | None = None) -> np.ndarray:
    if data_path.endswith(".npy"):
        return np.load(data_path).astype(np.int32)

    tokenizer = Tokenizer.load(tokenizer_path) if tokenizer_path else None
    if tokenizer is None:
        raise ValueError("Need --tokenizer-path for raw text data")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    return np.array(ids, dtype=np.int32)


def train(
    config: TransformerConfig,
    data: np.ndarray,
    lr: float = 3e-4,
    batch_size: int = 32,
    seq_len: int = 256,
    num_steps: int = 10000,
    log_every: int = 100,
    save_every: int = 1000,
    checkpoint_dir: str = "checkpoints",
    weight_decay: float = 0.01,
    seed: int = 42,
) -> Any:
    rng_key = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    model = TransformerLM(config)
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input, deterministic=True)

    opt_state = init_adamw(params)

    grad_fn = jax.jit(jax.grad(lambda p, b: loss_fn(p, model, b)))
    jit_loss = jax.jit(lambda p, b: loss_fn(p, model, b))

    print(f"Training {config.num_layers}L-{config.d_model}D model | "
          f"{sum(x.size for x in jax.tree.leaves(params)):,} params")
    print(f"Data: {len(data):,} tokens | batch_size={batch_size} seq_len={seq_len}")

    t0 = time.time()
    for step in tqdm(range(1, num_steps + 1), desc="Training"):
        batch = jnp.array(get_batch(data, batch_size, seq_len, np_rng))
        grads = grad_fn(params, batch)
        params, opt_state = adamw_step(
            params, grads, opt_state,
            lr=lr, weight_decay=weight_decay,
        )

        if step % log_every == 0:
            loss_val = float(jit_loss(params, batch))
            ppl = float(jnp.exp(loss_val))
            elapsed = time.time() - t0
            tokens_per_sec = (step * batch_size * seq_len) / elapsed
            print(f"  step {step:>6d} | loss {loss_val:.4f} | ppl {ppl:.2f} | "
                  f"{tokens_per_sec:.0f} tok/s")

        if step % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pkl")
            save_checkpoint(ckpt_path, params, opt_state, step, config)
            print(f"  saved {ckpt_path}")

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    parser.add_argument("--data", required=True, help="Path to .npy or .txt training data")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer .json (for .txt data)")
    parser.add_argument("--config", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TINY_CONFIG if args.config == "tiny" else SMALL_CONFIG
    config = TransformerConfig(
        **{**vars(TINY_CONFIG if args.config == "tiny" else SMALL_CONFIG).__dict__,
           **{k: v for k, v in {"max_seq_len": args.seq_len}.items()
              if v is not None}}
    ) if args.seq_len else config

    data = load_data(args.data, args.tokenizer)

    train(
        config=config,
        data=data,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
