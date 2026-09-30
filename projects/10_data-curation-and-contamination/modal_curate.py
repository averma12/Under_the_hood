"""Chapter 10 on Modal: audit the exact Chapter 9 FineWeb-Edu corpus.

Re-streams the same first 1,067,008 documents that became Chapter 9's shards
(the first 9,671 formed the validation shard), then runs audit_corpus.audit
against the validation split plus MMLU and ARC.

  W=/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts
  $W run projects/10_data-curation-and-contamination/modal_curate.py --mode audit
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CH9_MANIFEST = HERE.parent / "09_pretraining-on-the-real-web" / "outputs" / "fineweb_manifest.json"
VOLUME_NAME = "under-the-hood-pretrain"
OUT = "/vol/curation"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "mmh3", "datasets")
    .add_local_file(HERE / "curate.py", "/root/curate.py")
    .add_local_file(HERE / "audit_corpus.py", "/root/audit_corpus.py")
)
CH9 = HERE.parent / "09_pretraining-on-the-real-web"
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "tiktoken")
    .add_local_file(HERE.parent / "05_your-gpt-from-a-blank-file" / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(CH9 / "pretrain_data.py", "/root/pretrain_data.py")
)
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "tiktoken", "datasets")
    .add_local_file(CH9 / "pretrain_data.py", "/root/pretrain_data.py")
    .add_local_file(CH9 / "prepare_fineweb.py", "/root/prepare_fineweb.py")
)
app = modal.App("under-the-hood-curation")
volume = modal.Volume.from_name(VOLUME_NAME)


def load_benchmarks():
    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    benchmarks = {"mmlu": [f"{r['question']} {' '.join(r['choices'])}" for r in mmlu]}
    for name in ("ARC-Challenge", "ARC-Easy"):
        rows = load_dataset("allenai/ai2_arc", name, split="test")
        benchmarks[name.lower()] = [f"{r['question']} {' '.join(r['choices']['text'])}"
                                    for r in rows]
    return benchmarks


@app.function(image=image, cpu=16, memory=65536, timeout=3 * 3600, volumes={"/vol": volume})
def audit_fineweb(total_docs: int, val_docs: int) -> dict:
    import itertools
    import sys
    import time

    import numpy as np
    from datasets import load_dataset

    sys.path.insert(0, "/root")
    import audit_corpus

    started = time.time()
    benchmarks = load_benchmarks()
    rows = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                        streaming=True)
    docs = (row["text"] for row in itertools.islice(rows, total_docs))
    summary, examples, arrays = audit_corpus.audit(
        docs, val_docs, benchmarks, processes=16, chunk=256,
        log=lambda msg: print(f"[{time.time() - started:6.0f}s] {msg}", flush=True))
    summary["seconds"] = round(time.time() - started)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{OUT}/audit_arrays.npz", **arrays)
    Path(f"{OUT}/audit_summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return {"summary": summary, "examples": examples}


@app.function(image=gpu_image, gpu="A100-40GB", timeout=1800, volumes={"/vol": volume})
def val_doc_bpb(runs: list[str], batch: int = 32) -> dict:
    """Per-document validation bpb for Chapter 9 checkpoints.

    Consecutive 1,024-token windows cover the whole 10 M-token validation
    shard. Each target token is attributed to the document it belongs to
    (documents are separated by <|endoftext|>).
    """
    import sys

    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    import my_gpt as m
    from pretrain_data import EOT, get_encoder, token_byte_lengths

    tokens = np.fromfile("/vol/fineweb_edu_gpt2/val_0000.bin", dtype=np.uint16).astype(np.int64)
    doc_of = np.concatenate([[0], np.cumsum(tokens[:-1] == EOT)])  # doc id of each position
    byte_len = token_byte_lengths(get_encoder()).numpy()
    seq = 1024
    windows = (len(tokens) - 1) // seq
    out = {"docs": int(doc_of[-1] + 1), "windows": windows}
    for run in runs:
        saved = torch.load(f"/vol/runs/{run}/latest.pt", map_location="cpu", weights_only=False)
        model = m.GPT(m.TransformerConfig(**saved["config"])).cuda().eval()
        model.load_state_dict(saved["model"])
        nats = np.zeros(out["docs"])
        nbytes = np.zeros(out["docs"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for first in range(0, windows, batch):
                idx = np.arange(first, min(first + batch, windows))
                starts = idx * seq
                span = starts[:, None] + np.arange(seq + 1)[None, :]
                window = torch.from_numpy(tokens[span]).cuda()
                logits, _ = model(window[:, :-1])
                loss = F.cross_entropy(logits.float().flatten(0, 1), window[:, 1:].flatten(),
                                       reduction="none").cpu().numpy()
                targets = span[:, 1:].ravel()
                text = byte_len[tokens[targets]] > 0  # skip <|endoftext|> targets
                np.add.at(nats, doc_of[targets][text], loss[text])
                np.add.at(nbytes, doc_of[targets][text], byte_len[tokens[targets]][text])
        out[run] = {"nats": nats.tolist(), "bytes": nbytes.tolist(), "step": saved["step"]}
        print(run, "overall bpb", nats.sum() / (np.log(2) * nbytes.sum()), flush=True)
    audit = np.load("/vol/curation/audit_arrays.npz")
    out["val_leak"] = audit["val_leak"].tolist()
    # BREAK IT trained only on the first 5 M training tokens. Did that slice
    # contain training copies of validation text?
    train_tokens = np.fromfile("/vol/fineweb_edu_gpt2/train_0001.bin", dtype=np.uint16,
                               count=5_000_000)
    slice_docs = int((train_tokens == EOT).sum()) + 1
    first_train = out["docs"] - 1  # the last validation doc continues into train_0001
    overlap = audit["doc_overlap"]
    out["breakit_slice_docs"] = slice_docs
    out["breakit_slice_docs_overlapping_val"] = int(
        (overlap[first_train:first_train + slice_docs] >= 50).sum())
    return out


@app.function(image=prep_image, cpu=8, memory=16384, timeout=3 * 3600,
              volumes={"/vol": volume})
def build_decontaminated(total_docs: int, val_docs: int) -> dict:
    """Re-tokenize Chapter 9's corpus without duplicates, validation overlap, or
    benchmark hits. The same validation documents go first, so val_0000.bin is
    byte-identical to Chapter 9's and results stay comparable."""
    import hashlib
    import itertools
    import sys

    import numpy as np
    from datasets import load_dataset

    sys.path.insert(0, "/root")
    from prepare_fineweb import prepare

    a = np.load("/vol/curation/audit_arrays.npz")
    n = len(a["is_val"])
    near_dup = a["clusters"] != np.arange(n)
    keep = (a["is_val"] | (~near_dup & (a["doc_overlap"] == 0) & ~a["bench_hit"]))
    rows = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                        streaming=True)
    docs = (row["text"] for i, row in enumerate(itertools.islice(rows, total_docs)) if keep[i])
    out = "/vol/fineweb_edu_gpt2_decontaminated"
    manifest = prepare(docs, out, 10**12, 100_000_000, 10_000_000,
                       source="Chapter 9 corpus minus near duplicates, any 13-gram overlap "
                              "with validation, and MMLU/ARC hits")
    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    same_val = digest(f"{out}/val_0000.bin") == digest("/vol/fineweb_edu_gpt2/val_0000.bin")
    volume.commit()
    return {"kept_training_docs": int(keep.sum() - val_docs),
            "dropped_training_docs": int((~keep).sum()),
            "total_tokens": manifest["total_tokens"], "shards": len(manifest["shards"]),
            "validation_shard_identical": same_val}


@app.local_entrypoint()
def main(mode: str = "audit"):
    manifest = json.loads(CH9_MANIFEST.read_text())
    total_docs = sum(s["docs"] for s in manifest["shards"])
    val_docs = manifest["shards"][0]["docs"]
    if mode == "audit":
        result = audit_fineweb.remote(total_docs, val_docs)
        out = HERE / "outputs"
        out.mkdir(exist_ok=True)
        (out / "audit_summary.json").write_text(json.dumps(result["summary"], indent=2))
        (out / "audit_examples.json").write_text(
            json.dumps(result["examples"], indent=2, ensure_ascii=False))
        print(json.dumps(result["summary"], indent=2))
    elif mode == "decontaminate":
        print(json.dumps(build_decontaminated.remote(total_docs, val_docs), indent=2))
    elif mode == "val_eval":
        result = val_doc_bpb.remote(["main-lr0.003", "sweep-lr3e-3", "breakit-5M-repeated",
                                     "decontam-lr3e-3"])
        (HERE / "outputs" / "val_doc_bpb.json").write_text(json.dumps(result))
        print({k: v for k, v in result.items() if not isinstance(v, (dict, list))})
    else:
        raise ValueError("mode must be audit, decontaminate, or val_eval")
