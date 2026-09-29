"""Train our Chapter 5 GPT on Modal A100 using the dedicated qwen-tts-lab wrapper.

Run only after confirming the GPU budget:
  /Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts run \
    projects/05_your-gpt-from-a-blank-file/modal_gpt_a100.py \
    --steps 5000 --max-seconds 600 --variant chapter7_rmsnorm_swiglu

The generated text and training metrics return to the terminal. Checkpoints and
metrics persist in the under-the-hood-gpt-a100 Modal Volume.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
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
    .add_local_file(DATA / "train_ids.json", "/root/data/train_ids.json")
    .add_local_file(DATA / "val_ids.json", "/root/data/val_ids.json")
)
app = modal.App("under-the-hood-gpt-a100")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={"/artifacts": volume},
)
def train_on_a100(run_id: str, steps: int, max_seconds: int, variant: str) -> dict:
    import sys
    import time
    from dataclasses import asdict

    import torch

    sys.path.insert(0, "/root")
    import my_gpt as m

    if not torch.cuda.is_available():
        raise RuntimeError("A100 CUDA device is unavailable.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    cfg = m.TransformerConfig(
        vocab_size=2048,
        block_size=128,
        batch_size=32,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=min(200, max(1, steps // 10)),
        max_steps=steps,
        eval_interval=250,
        eval_steps=20,
        grad_clip=1.0,
        device="cuda",
        mlp_type="swiglu" if variant in ("chapter7_swiglu", "chapter7_rmsnorm_swiglu") else "gelu",
        norm_type="rmsnorm" if variant == "chapter7_rmsnorm_swiglu" else "layernorm",
    )
    tokenizer = m.BPETokenizer.load("/root/data/tokenizer.json")
    if len(tokenizer.vocab) != cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from the model config.")
    datasets = {
        split: torch.tensor(
            json.loads(Path(f"/root/data/{split}_ids.json").read_text()),
            dtype=torch.long,
            device=device,
        )
        for split in ("train", "val")
    }
    model = m.GPT(cfg).to(device)
    if variant in ("chapter6", "chapter7_swiglu", "chapter7_rmsnorm_swiglu"):
        m.init_scaled_residual_projections(model)
        groups = m.configure_decay_groups(model, weight_decay=0.1)
        optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate, fused=True)
    elif variant == "prototype":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, fused=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    # The extra residual reinitialization consumes CUDA RNG; align subsequent
    # dropout sampling with the prototype after both models are constructed.
    torch.cuda.manual_seed_all(42)
    train_rng = torch.Generator(device=device).manual_seed(42)
    offsets = torch.arange(cfg.block_size, device=device)
    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "variant.json").write_text(json.dumps({"variant": variant}, indent=2))

    def batch(split: str, rng: torch.Generator):
        data = datasets[split]
        starts = torch.randint(
            len(data) - cfg.block_size, (cfg.batch_size,),
            generator=rng, device=device,
        )
        positions = starts[:, None] + offsets[None, :]
        return data[positions], data[positions + 1]

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = {}
        for seed, split in enumerate(("train", "val"), start=100):
            rng = torch.Generator(device=device).manual_seed(seed)
            losses = []
            for _ in range(cfg.eval_steps):
                x, y = batch(split, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                losses.append(loss.item())
            results[split] = sum(losses) / len(losses)
        model.train()
        return results

    def checkpoint(name: str, step: int, best_val: float):
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "variant": variant,
            "best_val": best_val,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "train_rng": train_rng.get_state(),
            "tokenizer_merges": [
                [a, b, idx] for (a, b), idx in tokenizer.merges.items()
            ],
        }
        path = run_dir / name
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        volume.commit()

    started = time.monotonic()
    initial = evaluate()
    best_val = initial["val"]
    checkpoint("best.pt", 0, best_val)
    rows = [{"step": 0, **initial}]
    completed = 0
    model.train()
    with (run_dir / "metrics.jsonl").open("w") as log:
        log.write(json.dumps(rows[0]) + "\n")
        log.flush()
        print(f"A100={torch.cuda.get_device_name(0)} | params={sum(p.numel() for p in model.parameters()):,}", flush=True)
        print(f"step=0 train={initial['train']:.4f} val={initial['val']:.4f}", flush=True)
        for step in range(steps):
            if time.monotonic() - started >= max_seconds:
                print(f"Stopped at time cap after {completed} steps", flush=True)
                break
            lr = m.get_lr(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = batch("train", train_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            completed = step + 1
            if completed % cfg.eval_interval == 0 or completed == steps:
                scores = evaluate()
                row = {
                    "step": completed,
                    "lr": lr,
                    "batch_loss": loss.item(),
                    "grad_norm_before_clip": grad_norm.item(),
                    "elapsed_seconds": time.monotonic() - started,
                    **scores,
                }
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"step={completed} lr={lr:.6f} train={scores['train']:.4f} "
                    f"val={scores['val']:.4f} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                if scores["val"] < best_val:
                    best_val = scores["val"]
                    checkpoint("best.pt", completed, best_val)
                if completed % 1000 == 0:
                    checkpoint("latest.pt", completed, best_val)
        if completed and rows[-1]["step"] != completed:
            scores = evaluate()
            row = {
                "step": completed,
                "lr": m.get_lr(completed - 1, cfg),
                "elapsed_seconds": time.monotonic() - started,
                **scores,
            }
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if scores["val"] < best_val:
                best_val = scores["val"]
                checkpoint("best.pt", completed, best_val)
        checkpoint("checkpoint.pt", completed, best_val)

    samples = []
    for seed, prompt in ((100, "Harry looked at"), (101, "Hermione said,"), (102, "The door opened and")):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(ids, 80, temperature=0.8)
        samples.append({
            "prompt": prompt,
            "continuation": tokenizer.decode(generated[0, ids.shape[1]:].tolist()),
            "seed": seed,
        })
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    volume.commit()
    return {
        "run_id": run_id,
        "variant": variant,
        "steps": completed,
        "train_tokens": len(datasets["train"]),
        "val_tokens": len(datasets["val"]),
        "initial": rows[0],
        "final": rows[-1],
        "samples": samples,
        "volume": VOLUME_NAME,
        "elapsed_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main(steps: int = 5000, max_seconds: int = 600, variant: str = "prototype"):
    if steps < 2 or not 1 <= max_seconds <= 720:
        raise ValueError("Require at least 2 steps and a 1–720 second run cap.")
    if variant not in ("prototype", "chapter6", "chapter7_swiglu", "chapter7_rmsnorm_swiglu"):
        raise ValueError("variant must be 'prototype', 'chapter6', 'chapter7_swiglu', or 'chapter7_rmsnorm_swiglu'.")
    run_id = f"gpt-2048-{variant}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = train_on_a100.remote(run_id, steps, max_seconds, variant)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    local_dir = DATA / "modal_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    wrapper = "/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts"
    for name in ("checkpoint.pt", "metrics.jsonl", "samples.json", "config.json", "variant.json"):
        subprocess.run(
            [wrapper, "volume", "get", VOLUME_NAME, f"{run_id}/{name}", str(local_dir / name)],
            check=True,
        )
    (local_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(local_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (local_dir / "metrics.jsonl").read_text().splitlines()]
    fig, (loss_ax, lr_ax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    for key in ("train", "val"):
        loss_ax.plot([row["step"] for row in rows], [row[key] for row in rows], label=key)
    loss_ax.set(ylabel="Cross-entropy loss / BPE token", title="A100 GPT training")
    loss_ax.legend()
    loss_ax.grid(alpha=0.25)
    lr_ax.plot([row["step"] for row in rows[1:]], [row["lr"] for row in rows[1:]])
    lr_ax.set(xlabel="Optimizer updates", ylabel="Learning rate")
    lr_ax.grid(alpha=0.25)
    fig.savefig(local_dir / "loss_and_lr.png", dpi=160)
    plt.close(fig)
    print(f"Local results: {local_dir.resolve()}")
