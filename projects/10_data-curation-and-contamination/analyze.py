"""Chapter 10 analyses on the saved audit outputs.

    .venv/bin/python projects/10_data-curation-and-contamination/analyze.py val
    .venv/bin/python projects/10_data-curation-and-contamination/analyze.py mmlu

val:  per-document validation bpb of three Chapter 9 checkpoints, grouped by
      how much of each validation document also appears in training
      (outputs/val_doc_bpb.json from `modal_curate.py --mode val_eval`).
mmlu: classify MMLU 13-gram hits into quoted passages, partial overlaps, and
      strong leaks (question and answer). Needs the full outputs/audit_examples.json
      written by `modal_curate.py --mode audit` (not committed: 12 MB of web text).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))

import curate as c  # noqa: E402

RUNS = {"main-lr0.003": "main run (1 B tokens: saw the copies ~once)",
        "sweep-lr3e-3": "sweep run (150 M tokens: saw ~14% of training)",
        "breakit-5M-repeated": "BREAK IT (5 M distinct tokens: saw almost none)",
        "decontam-lr3e-3": "decontaminated corpus (150 M tokens: saw none of the overlap)"}
# (exposed run, unexposed run): same-strength pair first, then the Chapter 9 pair.
CONTRASTS = [("sweep-lr3e-3", "decontam-lr3e-3"), ("main-lr0.003", "breakit-5M-repeated")]
GROUPS = [("< 5%", 0.0, 0.05), ("5–50%", 0.05, 0.5), ("50–80%", 0.5, 0.8), ("≥ 80%", 0.8, 1.01)]


def val():
    r = json.loads((OUT / "val_doc_bpb.json").read_text())
    leak = np.array(r["val_leak"])
    nbytes = np.array(r["main-lr0.003"]["bytes"])
    nats = {run: np.array(r[run]["nats"]) for run in RUNS}

    def bpb(run, idx):
        return nats[run][idx].sum() / (np.log(2) * nbytes[idx].sum())

    masks = [np.flatnonzero((leak >= lo) & (leak < hi)) for _, lo, hi in GROUPS]
    table = {run: [bpb(run, m) for m in masks] for run in RUNS}
    everything = np.arange(leak.size)
    decontaminated = np.flatnonzero(leak < 0.5)
    print(f"{'leak group':12} {'docs':>5} " + " ".join(f"{run[:14]:>14}" for run in RUNS))
    for (name, _, _), m, i in zip(GROUPS, masks, range(len(masks))):
        print(f"{name:12} {m.size:5d} " + " ".join(f"{table[run][i]:14.3f}" for run in RUNS))
    for run in RUNS:
        print(f"{run}: all {bpb(run, everything):.4f}, decontaminated {bpb(run, decontaminated):.4f}")

    print(f"main run, only docs with < 5% leak: {table['main-lr0.003'][0]:.4f}")
    # Difference in differences: how much more the main run gains on a leak group
    # (relative to clean docs) than BREAK IT, which saw almost none of the copies.
    rng = np.random.default_rng(0)
    clean = masks[0]
    for exposed, unexposed in CONTRASTS:
        print(f"extra gain on leaked docs, {exposed} vs {unexposed}:")
        for (name, _, _), group, i in list(zip(GROUPS, masks, range(len(masks))))[1:]:
            did = []
            for _ in range(2000):
                gi, ci = rng.choice(group, group.size), rng.choice(clean, clean.size)
                did.append((bpb(exposed, ci) - bpb(exposed, gi))
                           - (bpb(unexposed, ci) - bpb(unexposed, gi)))
            point = ((table[exposed][0] - table[exposed][i])
                     - (table[unexposed][0] - table[unexposed][i]))
            print(f"  leak {name:7} {point:+.4f} bpb, 95% bootstrap CI "
                  f"[{np.percentile(did, 2.5):+.4f}, {np.percentile(did, 97.5):+.4f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for run, label in RUNS.items():
        ax.plot([g[0] for g in GROUPS], [v - table[run][0] for v in table[run]], marker="o",
                label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="share of the validation document's 13-grams also found in training",
           ylabel="bpb relative to clean docs", title="Are leaked validation docs easier?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "val_leak_bpb.png", dpi=150)


def load_mmlu():
    path = OUT / "mmlu_test.parquet"
    if not path.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/cais/mmlu/resolve/main/"
                                   "all/test-00000-of-00001.parquet", path)
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def contains(words, phrase):
    return bool(phrase) and f" {' '.join(phrase)} " in f" {' '.join(words)} "


def mmlu():
    items = load_mmlu()
    hits = json.loads((OUT / "audit_examples.json").read_text())["benchmark_hit_details"]["mmlu"]
    best = {}
    for h in hits:  # keep each item's strongest matching training document
        frac = h["matched"] / h["item_grams"]
        if frac > best.get(h["item"], (0, ""))[0]:
            best[h["item"]] = (frac, h["doc_text"])
    rows = []
    for i, (frac, doc) in best.items():
        item = items[i]
        words = c.normalize(doc)
        answer = c.normalize(item["choices"][item["answer"]])
        ask = c.normalize(item["question"].strip().split("\n")[-1])[-13:]
        rows.append({
            "item": i, "subject": item["subject"], "matched_fraction": round(frac, 3),
            "quoted_passage": item["question"].startswith(
                "This question refers to the following information"),
            "question_in_doc": len(ask) >= 8 and contains(words, ask),
            "answer_in_doc": len(answer) >= 3 and contains(words, answer),
        })
    strong = [r for r in rows if r["question_in_doc"]
              and (r["answer_in_doc"] or r["matched_fraction"] >= 0.5)]
    summary = {
        "mmlu_items": len(items), "items_with_13gram_hits": len(rows),
        "quoted_passage_items": sum(r["quoted_passage"] for r in rows),
        "question_in_doc": sum(r["question_in_doc"] for r in rows),
        "strong_leaks": len(strong),
        "strong_leak_subjects": Counter(r["subject"] for r in strong).most_common(),
        "hit_subjects": Counter(r["subject"] for r in rows).most_common(8),
    }
    print(json.dumps(summary, indent=2))
    (OUT / "mmlu_hits_classified.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    {"val": val, "mmlu": mmlu}[sys.argv[1]]()
