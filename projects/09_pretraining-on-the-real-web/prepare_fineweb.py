"""Tokenize documents once and write uint16 shards + manifest.json.

Sources:
  fineweb  stream HuggingFaceFW/fineweb-edu (sample-10BT) without downloading
           the whole dataset; stops after --max-tokens. Used on Modal.
  text     local text files, one document per blank-line-separated block.
           Used for CPU smoke tests.

    .venv/bin/python projects/09_pretraining-on-the-real-web/prepare_fineweb.py \
        --source text --text-files book.txt --out-dir /tmp/shards \
        --shard-tokens 200000 --val-tokens 50000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretrain_data import ShardWriter, encode_documents, get_encoder  # noqa: E402


def fineweb_documents(name="sample-10BT"):
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceFW/fineweb-edu", name=name, split="train", streaming=True)
    for row in rows:
        yield row["text"]  # keep only the text; drop URL, score, and other metadata


def text_file_documents(paths):
    for path in paths:
        for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8")):
            if block.strip():
                yield block.strip()


def prepare(documents, out_dir, max_tokens, shard_tokens, val_tokens, batch_docs=1024,
            source=""):
    encoder = get_encoder()
    writer = ShardWriter(out_dir, shard_tokens, val_tokens)
    total, started, batch = 0, time.perf_counter(), []

    def flush(batch):
        nonlocal total
        for text, tokens in zip(batch, encode_documents(batch, encoder)):
            writer.add(tokens, len(text.encode("utf-8")))
            total += len(tokens)

    for text in documents:
        batch.append(text)
        if len(batch) == batch_docs:
            flush(batch)
            batch = []
            print(f"{total / 1e6:,.1f}M tokens, {total / (time.perf_counter() - started):,.0f}"
                  " tokens/s", flush=True)
            if total >= max_tokens:
                break
    else:
        if batch:
            flush(batch)
    return writer.close(source=source, total_tokens=total,
                        seconds=round(time.perf_counter() - started, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("fineweb", "text"), required=True)
    p.add_argument("--text-files", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=10**12)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--val-tokens", type=int, default=10_000_000)
    args = p.parse_args()
    docs = (fineweb_documents() if args.source == "fineweb"
            else text_file_documents(args.text_files))
    manifest = prepare(docs, args.out_dir, args.max_tokens, args.shard_tokens,
                       args.val_tokens, source=args.source)
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2))
    for shard in manifest["shards"]:
        print(shard)


if __name__ == "__main__":
    main()
