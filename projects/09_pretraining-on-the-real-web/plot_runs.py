"""Plot Chapter 9 runs from outputs/modal_runs/*/metrics.jsonl.

    .venv/bin/python projects/09_pretraining-on-the-real-web/plot_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "outputs" / "modal_runs"
FIGURES = HERE / "figures"


def load(run):
    rows = [json.loads(line) for line in (RUNS / run / "metrics.jsonl").open()]
    evals = [r for r in rows if "val_bpb" in r]
    logs = [r for r in rows if "train_loss" in r]
    return evals, logs


def plot_sweep(runs):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        evals, logs = load(run)
        label = "LR " + run.removeprefix("sweep-lr")
        left.plot([r["step"] for r in logs], [r["train_loss"] for r in logs], label=label)
        right.plot([r["tokens_seen"] / 1e6 for r in evals], [r["val_bpb"] for r in evals],
                   marker="o", ms=3, label=label)
    left.set(xlabel="optimizer step", ylabel="train loss (nats/token)",
             title="LR sweep: training loss", ylim=(None, 8))
    right.set(xlabel="tokens seen (M)", ylabel="validation bits per byte",
              title="LR sweep: held-out bpb", ylim=(None, 2.0))
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "lr_sweep.png", dpi=150)


def plot_main(run):
    evals, logs = load(run)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    tokens = [r["tokens_seen"] / 1e6 for r in evals]
    left.plot(tokens, [r["train_bpb_fixed"] for r in evals], label="train (fixed held-in windows)")
    left.plot(tokens, [r["val_bpb"] for r in evals], label="validation (held-out shard)")
    left.set(xlabel="tokens seen (M)", ylabel="bits per byte", title=f"{run}: bpb",
             ylim=(None, 2.0))
    right.plot([r["step"] for r in logs], [r["tokens_per_s"] / 1e3 for r in logs])
    right.set(xlabel="optimizer step", ylabel="thousand tokens / s", title="A100 throughput")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    left.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{run}.png", dpi=150)


def plot_breakit(run, reference):
    evals, _ = load(run)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    tokens = [r["tokens_seen"] / 1e6 for r in evals]
    ax.plot(tokens, [r["train_bpb_fixed"] for r in evals], label="repeated 5M: train windows")
    ax.plot(tokens, [r["val_bpb"] for r in evals], label="repeated 5M: validation")
    if (RUNS / reference / "metrics.jsonl").exists():
        ref, _ = load(reference)
        ref = [r for r in ref if r["tokens_seen"] <= evals[-1]["tokens_seen"]]
        ax.plot([r["tokens_seen"] / 1e6 for r in ref], [r["val_bpb"] for r in ref], "--",
                color="gray", label="fresh data (main run): validation")
    ax.set(xlabel="tokens seen (M)", ylabel="bits per byte", ylim=(None, 2.2),
           title="BREAK IT: 5M distinct tokens, ~16 passes")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "breakit_repeated_data.png", dpi=150)


def main():
    FIGURES.mkdir(exist_ok=True)
    sweeps = sorted((p.name for p in RUNS.glob("sweep-lr*")),
                    key=lambda name: float(name.removeprefix("sweep-lr")))
    if sweeps:
        plot_sweep(sweeps)
    for run in sorted(p.name for p in RUNS.glob("main-*")):
        plot_main(run)
    for run in sorted(p.name for p in RUNS.glob("breakit-*")):
        plot_breakit(run, reference="main-lr0.003")
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
