"""Byte-level BPE tokenizer (GPT-2 style)."""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from typing import Iterator

import regex


GPT2_PRETOKENIZE_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_chunk(text: str) -> list[str]:
    return regex.findall(GPT2_PRETOKENIZE_PATTERN, text)


def _pretokenize_parallel(text: str, num_workers: int | None = None) -> list[str]:
    num_workers = num_workers or max(1, os.cpu_count() or 1)
    if len(text) < 100_000 or num_workers <= 1:
        return _pretokenize_chunk(text)

    chunk_size = len(text) // num_workers
    chunks: list[str] = []
    start = 0
    for i in range(num_workers):
        end = start + chunk_size if i < num_workers - 1 else len(text)
        while end < len(text) and not text[end].isspace():
            end += 1
        chunks.append(text[start:end])
        start = end

    with mp.Pool(num_workers) as pool:
        results = pool.map(_pretokenize_chunk, chunks)
    return [tok for part in results for tok in part]


def _get_pair_counts(
    word_freqs: dict[tuple[int, ...], int],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge_pair(
    word_freqs: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
    pair_counts: dict[tuple[int, int], int],
) -> dict[tuple[int, ...], int]:
    """Merge a pair in all words, updating pair_counts incrementally."""
    new_word_freqs: dict[tuple[int, ...], int] = {}
    a, b = pair

    for word, freq in word_freqs.items():
        new_word: list[int] = []
        i = 0
        changed = False
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                if new_word:
                    old_left_pair = (new_word[-1], a)
                    pair_counts[old_left_pair] = pair_counts.get(old_left_pair, 0) - freq
                    new_left_pair = (new_word[-1], new_id)
                    pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + freq
                if i + 2 < len(word):
                    old_right_pair = (b, word[i + 2])
                    pair_counts[old_right_pair] = pair_counts.get(old_right_pair, 0) - freq
                    new_right_pair = (new_id, word[i + 2])
                    pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + freq
                new_word.append(new_id)
                i += 2
                changed = True
            else:
                new_word.append(word[i])
                i += 1

        key = tuple(new_word) if changed else word
        new_word_freqs[key] = new_word_freqs.get(key, 0) + freq

    pair_counts.pop(pair, None)
    return new_word_freqs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Returns (vocab, merges) where vocab maps token_id -> bytes and
    merges is the ordered list of (id_a, id_b) merge operations.
    """
    special_tokens = special_tokens or []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    for st in special_tokens:
        text = text.replace(st, "")

    tokens = _pretokenize_parallel(text)

    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
    for tok in tokens:
        word_freqs[tuple(tok.encode("utf-8"))] += 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []

    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")

    next_id = 256 + len(special_tokens)
    num_merges = vocab_size - next_id

    pair_counts = _get_pair_counts(word_freqs)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], p),
        )

        if pair_counts[best_pair] < 1:
            break

        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append(best_pair)
        word_freqs = _merge_pair(word_freqs, best_pair, next_id, pair_counts)
        next_id += 1

        pair_counts = {p: c for p, c in pair_counts.items() if c > 0}

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self._id_to_bytes = vocab
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        self._merge_priority: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        self._special_pattern: regex.Pattern | None = None
        if self.special_tokens:
            escaped = [regex.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
            self._special_pattern = regex.compile("|".join(escaped))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        if self._special_pattern:
            parts = self._special_pattern.split(text)
            specials = self._special_pattern.findall(text)
            for i, part in enumerate(parts):
                if part:
                    ids.extend(self._encode_chunk(part))
                if i < len(specials):
                    ids.append(self._bytes_to_id[specials[i].encode("utf-8")])
        else:
            ids = self._encode_chunk(text)

        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        tokens = regex.findall(GPT2_PRETOKENIZE_PATTERN, text)
        ids: list[int] = []
        for tok in tokens:
            byte_ids = list(tok.encode("utf-8"))
            ids.extend(self._apply_merges(byte_ids))
        return ids

    def _apply_merges(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            best_idx = -1
            best_priority = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                prio = self._merge_priority.get(pair)
                if prio is not None and prio < best_priority:
                    best_priority = prio
                    best_idx = i

            if best_idx == -1:
                break

            pair = (ids[best_idx], ids[best_idx + 1])
            merged_id = self._bytes_to_id[
                self._id_to_bytes[pair[0]] + self._id_to_bytes[pair[1]]
            ]
            ids = ids[:best_idx] + [merged_id] + ids[best_idx + 2 :]

        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self._id_to_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        import json

        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        import json

        with open(path, "r") as f:
            data = json.load(f)

        vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        merges = [tuple(m) for m in data["merges"]]
        return cls(vocab, merges, data.get("special_tokens", []))
