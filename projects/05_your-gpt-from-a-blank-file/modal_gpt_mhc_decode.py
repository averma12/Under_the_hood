"""Evaluate and sample the 5k milestones of the mHC + FlashAttention run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
VOLUME_NAME = "under-the-hood-gpt-a100"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(DATA / "tokenizer.json", "/root/data/tokenizer.json")
    .add_local_file(DATA / "val_ids.json", "/root/data/val_ids.json")
)
app = modal.App("under-the-hood-gpt-mhc-decode")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(image=image, gpu="A100-40GB", timeout=900,
              volumes={"/artifacts": volume})
def decode_milestones(run_id: str) -> dict:
    import sys

    import torch

    sys.path.insert(0, "/root")
    import my_gpt as m

    volume.reload()
    run_dir = Path("/artifacts") / run_id
    tokenizer = m.BPETokenizer.load("/root/data/tokenizer.json")
    val_ids = torch.tensor(json.loads(Path("/root/data/val_ids.json").read_text()),
                           dtype=torch.long, device="cuda")
    rows = []
    checkpoints = [
        (f"step_{step:05d}", run_dir / f"step_{step:05d}.pt")
        for step in range(5000, 30001, 5000)
    ] + [("best", run_dir / "best.pt")]
    for label, path in checkpoints:
        saved = torch.load(path, map_location="cuda", weights_only=False)
        step = int(saved["step"])
        if label != "best" and label != f"step_{step:05d}":
            raise ValueError(f"Wrong checkpoint step in {path}")
        cfg = m.TransformerConfig(**saved["config"])
        model = m.GPT(cfg).cuda().eval()
        model.load_state_dict(saved["model"])
        val_rng = torch.Generator(device="cuda").manual_seed(101)
        offsets = torch.arange(cfg.block_size, device="cuda")
        losses = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                starts = torch.randint(len(val_ids) - cfg.block_size,
                                       (cfg.batch_size,), generator=val_rng,
                                       device="cuda")
                positions = starts[:, None] + offsets
                _, loss = model(val_ids[positions], val_ids[positions + 1])
                losses.append(loss.item())
            samples = []
            for seed, prompt in ((100, "Harry looked at"),
                                 (101, "Hermione said,"),
                                 (102, "The door opened and")):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
                generated = model.generate(ids, 80, temperature=0.8)
                samples.append({
                    "prompt": prompt, "seed": seed,
                    "continuation": tokenizer.decode(
                        generated[0, ids.shape[1]:].tolist()),
                })
        val_loss = sum(losses) / len(losses)
        row = {"checkpoint": label, "step": step, "val_loss_20_batches": val_loss,
               "val_perplexity": math.exp(val_loss), "samples": samples}
        rows.append(row)
        print(f"{label} step={step} val={val_loss:.4f} "
              f"ppl={row['val_perplexity']:.3f}", flush=True)
        del model, saved
    output = {"run_id": run_id, "decode_temperature": 0.8,
              "new_tokens": 80, "milestones": rows}
    (run_dir / "milestone_decode.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False))
    volume.commit()
    return output


@app.local_entrypoint()
def main(run_id: str = "gpt-mhc-flash-stable-30k"):
    output = decode_milestones.remote(run_id)
    local_dir = DATA / "modal_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "milestone_decode.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False))
    rows = [json.loads(line) for line in (local_dir / "metrics.jsonl").read_text().splitlines()]
    os.environ.setdefault("MPLCONFIGDIR", str(local_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (loss_ax, lr_ax) = plt.subplots(2, 1, figsize=(8, 7),
                                         sharex=True, layout="constrained")
    for key in ("train", "val"):
        loss_ax.plot([row["step"] for row in rows],
                     [row[key] for row in rows], label=key)
    milestones = [row for row in output["milestones"] if row["checkpoint"] != "best"]
    best = next(row for row in output["milestones"] if row["checkpoint"] == "best")
    loss_ax.scatter([row["step"] for row in milestones],
                    [row["val_loss_20_batches"] for row in milestones],
                    s=16, label="val (20 batches)")
    loss_ax.scatter([best["step"]], [best["val_loss_20_batches"]],
                    marker="*", s=120, label="best checkpoint")
    loss_ax.set(ylabel="Cross-entropy / BPE token",
                title="mHC-4 + FlashAttention: Harry Potter GPT")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()
    lr_ax.plot([row["step"] for row in rows[1:]],
               [row["lr"] for row in rows[1:]])
    lr_ax.set(xlabel="Optimizer updates", ylabel="Learning rate")
    lr_ax.grid(alpha=0.25)
    fig.savefig(local_dir / "loss_and_lr.png", dpi=160)
    plt.close(fig)
    print(f"Results: {local_dir}", flush=True)
