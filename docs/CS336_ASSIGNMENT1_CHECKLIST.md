# CS336 Assignment 1 Checklist (Project 2)

This document maps every **problem** and **deliverable** from CS336 Assignment 1 PDF (in workspace root: `cs336_spring2025_assignment1_basics.pdf`) to this repo. Our implementation uses **JAX/XLA** (not PyTorch); the assignment is PyTorch-based, so "solving" here means **equivalent functionality** and **tasks you can run** (including via the demo notebook).

---

## §2 BPE Tokenizer

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **unicode1** | 1 | Written only (PDF §2.1) | Answer in writeup: chr(0), repr vs print, behavior in text. |
| **unicode2** | 3 | Written only (PDF §2.2) | Answer in writeup: UTF-8 vs UTF-16/32; decode_utf8_bytes_to_str_wrong example; invalid 2-byte sequence. |
| **train_bpe** | 15 | `src/tokenizer.py`: `train_bpe(input_path, vocab_size, special_tokens)` → `(vocab, merges)`. Byte-level BPE, GPT-2 regex pre-tokenization, lexicographic tie-break, special tokens stripped before pre-tokenization. Merges stored as `list[tuple[int,int]]` (IDs); assignment uses `tuple[bytes,bytes]`—convert via `vocab[id]` if needed. | **Notebook §1**: Train on sample text (temp file). For TinyStories/OWT run `train_bpe(path, 10000, ["<\|endoftext\|>"])` from CLI or add a cell. |
| **train_bpe_tinystories** | 2 | Same function; use TinyStories path, vocab_size=10000. | Run from script/notebook with TinyStories path; report time, memory, longest token in writeup. |
| **train_bpe_expts_owt** | 2 | Same function; use OpenWebText path, vocab_size=32000. | Run with OWT path; report longest token; compare TinyStories vs OWT tokenizer in writeup. |
| **tokenizer** | 15 | `src/tokenizer.py`: `Tokenizer(vocab, merges, special_tokens)`, `encode`, `decode`, `encode_iterable`, `save`/`load` (JSON). Decode uses `errors='replace'`. | **Notebook §1**: Build tokenizer, encode sample, save. |
| **tokenizer_experiments** | 4 | — | **Notebook**: Sample 10 docs, encode, compute bytes/token (compression ratio); OWT sample with TinyStories tokenizer; throughput (bytes/s) and time for 825GB; encode train/dev → uint16; answer in writeup. |

---

## §3 Transformer LM

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **linear** | 1 | JAX: linear is implicit in `nn.Dense` / param matrices in our modules (no separate Linear class). Assignment expects PyTorch `nn.Module` Linear. | Functionally covered by embedding and FFN projections. |
| **embedding** | 1 | `model.py`: embedding matrix `params["params"]["embedding"]`, shape `(vocab_size, d_model)`. | Used in TransformerLM forward. |
| **rmsnorm** | 1 | `model.py`: `RMSNorm` (learnable scale, upcast for stability). | Used in every block. |
| **positionwise_feedforward** | 2 | `model.py`: `SwiGLUFFN` — W2(SiLU(W1·x) ⊙ W3·x), d_ff ≈ 8/3·d_model. | Used in TransformerBlock. |
| **rope** | 2 | `model.py`: `precompute_rope_freqs`, `apply_rope`; applied to Q and K. | Used in MultiHeadAttention. |
| **softmax** | 1 | Inside attention: logsumexp trick for stability. | In scaled_dot_product logic. |
| **scaled_dot_product_attention** | 5 | `model.py`: attention scores, mask, softmax, output; causal mask. | In MultiHeadAttention. |
| **multihead_self_attention** | 5 | `model.py`: `MultiHeadAttention` with RoPE, causal masking. | Used in TransformerBlock. |
| **transformer_block** | 3 | `model.py`: `TransformerBlock` — pre-norm: RMSNorm→Attn→residual, RMSNorm→FFN→residual. | Used in TransformerLM. |
| **transformer_lm** | 3 | `model.py`: `TransformerLM` — embedding → N blocks → RMSNorm → tied LM head. | **Notebook §2–§5**: train and generate. |
| **transformer_accounting** | 5 | — | Written only: FLOPs, params, memory for GPT-2 XL/small/medium/large; use our config dimensions. |

---

## §4 Training

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **cross_entropy** | — | `train.py`: `cross_entropy_loss(logits, targets)` — shifted NLL, numerical stability. | Used in training loop. |
| **learning_rate_tuning** | 1 | — | Written: run SGD toy with lr 1e1, 1e2, 1e3; describe behavior. |
| **adamw** | 2 | `optimizer.py`: AdamW from scratch (m, v, bias correction, decoupled weight decay). | **Notebook §2**: `adamw_step` in loop. |
| **adamwAccounting** | 2 | — | Written: peak memory (params, activations, grads, optimizer state); max batch 80GB; FLOPs per step; MFU, days for 400K steps. |
| **learning_rate_schedule** | — | Not in repo (constant lr in notebook). | Add cosine + warmup in `train.py` or notebook for full A1. |
| **gradient_clipping** | 1 | Not in repo. | Add `jnp.clip` by global norm in train loop if required. |
| **data_loading** | 2 | `train.py`: `get_batch(data, batch_size, seq_len, rng)` → random contiguous segments. | **Notebook §2**: `get_batch` used. |
| **checkpointing** | 1 | `train.py`: `save_checkpoint`, `load_checkpoint` (params, opt_state, step, config). | **Notebook §2**: save to `demo_outputs/ckpt.pkl`. |

---

## §5 Training loop

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **training_together** | 4 | `train.py`: full training script with config, data loading, checkpointing, logging (loss/ppl). Memmap: use `np.load(..., mmap_mode='r')` for large data. | CLI: `python src/train.py --data ...`; **Notebook §2** is a short version. |

---

## §6 Generating text

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **decoding** | 3 | `generate.py`: prompt → sample until EOS or max_tokens; temperature scaling; top-k sampling (we have top_k; top-p can be added). | **Notebook §4–§5**: `generate(..., temperature=0.8, top_k=40)`. |

---

## §7 Experiments

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **experiment_log** | 3 | Logging: print in train.py; optional wandb. | Keep experiment log document; notebook logs loss. |
| **learning_rate** | 3 | — | Sweep LR on TinyStories; report curves; val loss ≤1.45 (or ≤2.0 low-resource). |
| **batch_size_experiment** | 1 | — | Vary batch size; report curves and commentary. |
| **generate** | 1 | `generate.py` | **Notebook §4–§5**: 256+ tokens dump and comment on fluency. |
| **layer_norm_ablation** | 1 | — | Ablation: remove RMSNorm, train; compare. |
| **pre_norm_ablation** | 1 | — | Implement post-norm, train, compare. |
| **no_pos_emb** | 1 | — | NoPE vs RoPE comparison. |
| **swiglu_ablation** | 1 | — | SwiGLU vs SiLU (match param count). |
| **main_experiment** | 2 | — | Train on OpenWebText; learning curve; generated text; compare to TinyStories. |
| **leaderboard** | — | Optional: submit perplexity to course leaderboard. | — |

---

## Summary: what the notebook runs

- **§1** BPE train (on sample text), tokenizer save, encode example.
- **§2** Data from tokenized sample, short training loop, checkpoint save.
- **§3** Loss and perplexity curve.
- **§4** Generation with standard KV cache (temperature, top_k).
- **§5** Generation with MLA + DSA.

## What you still need for a full writeup

- Written answers for: unicode1, unicode2, train_bpe_tinystories (time/memory/longest token), train_bpe_expts_owt, tokenizer_experiments (compression ratio, throughput, uint16), transformer_accounting, learning_rate_tuning, adamwAccounting, learning_rate (curves + val loss), batch_size_experiment, generate (256 tokens + commentary), layer_norm/pre_norm/no_pos/swiglu ablations, main_experiment (OWT).
- Optional: cosine LR schedule, gradient clipping, top-p sampling, experiment_log document.
