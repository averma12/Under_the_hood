"""Chapter 9 data pipeline: tokenize once, write shards, memory-map, measure bpb.

Shard format: a raw little-endian uint16 array of GPT-2 token IDs (the GPT-2
vocabulary, 50,257, fits in 16 bits). Every document ends with <|endoftext|>.
manifest.json records each shard's token, document, and UTF-8 byte counts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

EOT = 50256  # GPT-2 <|endoftext|>: marks the end of each document
VOCAB_SIZE = 50257
# Rounded up to a multiple of 64 so the LM-head matmul tiles evenly on GPU.
# IDs 50257..50303 never occur in the data.
PADDED_VOCAB_SIZE = 50304


def get_encoder():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def encode_documents(texts, encoder, threads=8):
    """Tokenize a batch of documents; each one gets <|endoftext|> appended."""
    encoded = encoder.encode_ordinary_batch(list(texts), num_threads=threads)
    return [np.array([*ids, EOT], dtype=np.uint16) for ids in encoded]


class ShardWriter:
    """Pack tokenized documents into fixed-size shards on disk.

    The first shard (val_tokens long) is the validation split, so validation documents are
    never trained on. Documents may cross a shard boundary; the EOT tokens
    still mark where each one ends.
    """

    def __init__(self, out_dir, shard_tokens, val_tokens=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.val_tokens = val_tokens or shard_tokens
        self.buffer = np.empty(max(shard_tokens, self.val_tokens), dtype=np.uint16)
        self.filled = 0
        self.docs = 0
        self.bytes = 0
        self.shards = []

    def add(self, tokens, text_bytes):
        self.docs += 1
        self.bytes += text_bytes
        while len(tokens):
            capacity = self.shard_tokens if self.shards else self.val_tokens
            take = min(len(tokens), capacity - self.filled)
            self.buffer[self.filled:self.filled + take] = tokens[:take]
            self.filled += take
            tokens = tokens[take:]
            if self.filled == capacity:
                self._flush()

    def _flush(self):
        split = "val" if not self.shards else "train"
        name = f"{split}_{len(self.shards):04d}.bin"
        self.buffer[:self.filled].tofile(self.out_dir / name)
        self.shards.append({"file": name, "split": split, "tokens": int(self.filled),
                            "docs": self.docs, "text_bytes": self.bytes})
        self.filled = self.docs = self.bytes = 0

    def close(self, **info):
        if self.filled:
            self._flush()
        manifest = {"format": "uint16 GPT-2 token IDs, EOT after each document",
                    "shards": self.shards, **info}
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def token_byte_lengths(encoder):
    """UTF-8 bytes each token decodes to; EOT counts as 0 (it is not text)."""
    lengths = np.array([len(encoder.decode_single_token_bytes(i)) for i in range(VOCAB_SIZE)])
    lengths[EOT] = 0
    padded = np.zeros(PADDED_VOCAB_SIZE, dtype=np.int64)
    padded[:VOCAB_SIZE] = lengths
    return torch.from_numpy(padded)


class ShardLoader:
    """Random training windows read straight from memory-mapped shards.

    np.memmap maps the file into virtual memory: the OS pages in only the
    slices a batch touches, so the dataset can be far larger than RAM.
    """

    def __init__(self, data_dir, split, seq_len, seed=0, limit_tokens=0):
        manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
        # A short final shard can be smaller than one window; skip it.
        self.arrays = [np.memmap(Path(data_dir) / s["file"], dtype=np.uint16, mode="r")
                       for s in manifest["shards"]
                       if s["split"] == split and s["tokens"] > seq_len + 1]
        if not self.arrays:
            raise ValueError(f"No {split} shard in {data_dir} holds a {seq_len}-token window.")
        if limit_tokens:
            # BREAK IT: train on one small slice again and again.
            self.arrays = [self.arrays[0][:limit_tokens]]
        self.seq_len = seq_len
        sizes = np.array([len(a) - seq_len - 1 for a in self.arrays], dtype=np.float64)
        self.weights = sizes / sizes.sum()  # sample shards in proportion to size
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, device="cpu"):
        shard_ids = self.rng.choice(len(self.arrays), size=batch_size, p=self.weights)
        rows = []
        for s in shard_ids:
            start = self.rng.integers(0, len(self.arrays[s]) - self.seq_len - 1)
            rows.append(self.arrays[s][start:start + self.seq_len + 1].astype(np.int64))
        window = torch.from_numpy(np.stack(rows))
        if str(device).startswith("cuda"):
            # Pinned host memory lets the copy to the GPU run asynchronously.
            window = window.pin_memory().to(device, non_blocking=True)
        else:
            window = window.to(device)
        return window[:, :-1], window[:, 1:]

    def fixed_windows(self, num_tokens, batch_size, device="cpu"):
        """The same consecutive windows every call: a stable eval set."""
        data = self.arrays[0]
        count = min(num_tokens // self.seq_len, (len(data) - 1) // self.seq_len)
        for first in range(0, count, batch_size):
            rows = [data[i * self.seq_len:(i + 1) * self.seq_len + 1].astype(np.int64)
                    for i in range(first, min(first + batch_size, count))]
            window = torch.from_numpy(np.stack(rows)).to(device)
            yield window[:, :-1], window[:, 1:]

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def bits_per_byte(total_nats, total_bytes):
    """Cross-entropy in nats, converted to bits, per byte of UTF-8 text.

    Unlike loss per token, this does not depend on the tokenizer: a
    tokenizer with longer tokens has fewer, harder predictions per byte.
    """
    return total_nats / (math.log(2) * total_bytes)
