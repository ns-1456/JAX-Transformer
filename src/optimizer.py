"""AdamW optimizer from scratch in JAX. Weight decay applied to params."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

PyTree = Any


class AdamWState(NamedTuple):
    m: PyTree
    v: PyTree
    step: int


def init_adamw(params: PyTree) -> AdamWState:
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return AdamWState(m=m, v=v, step=0)


def adamw_step(
    params: PyTree,
    grads: PyTree,
    state: AdamWState,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[PyTree, AdamWState]:
    t = state.step + 1

    new_m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.m, grads)
    new_v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, state.v, grads)

    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    m_hat = jax.tree.map(lambda m: m / bc1, new_m)
    v_hat = jax.tree.map(lambda v: v / bc2, new_v)

    new_params = jax.tree.map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
        params,
        m_hat,
        v_hat,
    )

    return new_params, AdamWState(m=new_m, v=new_v, step=t)
