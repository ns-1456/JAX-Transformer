# TinyStories speed-run benchmark (A100 / H100)

**Notebook:** Use [notebooks/speedrun_tinystories.ipynb](../notebooks/speedrun_tinystories.ipynb) for a single-notebook speed run (download → BPE → encode → train → plot → generate). No assignment boilerplate.

## Goal

Train the Transformer LM on **TinyStories** on **A100 or H100** with:

- **Target:** Validation loss **≤ 1.45** (competitive with common TinyStories baselines; assignment-style target is val loss ≤ 1.45 or ≤ 2.0 low-resource).
- **Efficiency:** High throughput (tokens/sec), good GPU utilization (MFU), minimal CPU/IO overhead so runs are reproducible and comparable.

## Setup

1. **Data:** TinyStories (e.g. `TinyStoriesV2-GPT4-train.txt`). Use the demo notebook’s download cell or your own path.
2. **Tokenizer:** Train BPE on TinyStories, e.g. `train_bpe(path, 10000, ["<|endoftext|>"])`, then encode corpus to token IDs (e.g. uint16 `.npy`).
3. **Hardware:** Single A100 or H100 (or multi-GPU if you extend the training script).

## How to run

From repo root, with tokenized data and tokenizer in place:

```bash
# Example: small config on TinyStories token IDs
python src/train.py \
  --data path/to/tinystories_ids.npy \
  --tokenizer path/to/tokenizer_tinystories.json \
  --config small \
  --batch-size 64 \
  --num-steps 40000
```

Tune `--batch-size` to fit GPU memory (A100 80GB allows larger batches). Use `--config tiny` for faster iteration; for the benchmark, `small` (or equivalent) is typical.

## Metrics

- **Validation loss:** Target ≤ 1.45 (report final and best).
- **Efficiency:** Report **tokens/sec** (and optionally MFU, time to target loss) so the run is comparable and clearly efficient.

## Notes

- Add a proper train/val split and validation loop if not already present, so “validation loss” is well-defined.
- Learning rate schedule (e.g. cosine + warmup) and gradient clipping help reach the target stably; see CS336 assignment for typical choices.
