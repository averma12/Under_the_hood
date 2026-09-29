"""Matched CPU comparison of learned positions and RoPE on Harry Potter BPE.

Both models start with identical shared weights and use the same data windows,
dropout seed, optimizer recipe, learning-rate schedule, and evaluation windows.
Only the position mechanism differs. This intentionally uses a smaller model
than the 30k-step A100 run so the comparison can finish locally.

Run from the repository root:
    .venv/bin/python projects/07_the-details-that-matter/experiment_rope_cpu.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "05_your-gpt-from-a-blank-file"
DATA = P5 / "outputs" / "my_gpt_2048"
spec = importlib.util.spec_from_file_location("chapter5_my_gpt_rope_experiment", P5 / "my_gpt.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def load_data():
    return {
        split: torch.tensor(json.loads((DATA / f"{split}_ids.json").read_text()), dtype=torch.long)
        for split in ("train", "val")
    }


def batch(data, cfg, rng, offsets):
    starts = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,), generator=rng)
    positions = starts[:, None] + offsets[None, :]
    return data[positions], data[positions + 1]


@torch.no_grad()
def evaluate(model, datasets, cfg, offsets):
    model.eval()
    scores = {}
    for seed, split in ((100, "train"), (101, "val")):
        rng = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(cfg.eval_steps):
            x, y = batch(datasets[split], cfg, rng, offsets)
            _, loss = model(x, y)
            losses.append(loss.item())
        scores[split] = sum(losses) / len(losses)
    model.train()
    return scores


def train_variant(name, model, cfg, datasets, offsets, output_dir):
    torch.manual_seed(1000)  # Align dropout draws across the two runs.
    train_rng = torch.Generator().manual_seed(42)
    optimizer = torch.optim.AdamW(m.configure_decay_groups(model, 0.1), lr=cfg.learning_rate)
    rows = []
    initial = {"step": 0, "lr": 0.0, **evaluate(model, datasets, cfg, offsets)}
    rows.append(initial)
    print(f"{name} step=0 train={initial['train']:.4f} val={initial['val']:.4f}", flush=True)
    started = time.perf_counter()
    for step in range(cfg.max_steps):
        lr = m.get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = batch(datasets["train"], cfg, train_rng, offsets)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {name} loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % cfg.eval_interval == 0 or step + 1 == cfg.max_steps:
            scores = evaluate(model, datasets, cfg, offsets)
            row = {"step": step + 1, "lr": lr, "elapsed_seconds": time.perf_counter() - started, **scores}
            rows.append(row)
            print(
                f"{name} step={step + 1} train={scores['train']:.4f} "
                f"val={scores['val']:.4f} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    (output_dir / f"metrics_{name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": asdict(cfg), "step": cfg.max_steps, "variant": name},
        output_dir / f"checkpoint_{name}.pt",
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    torch.set_num_threads(args.threads)
    datasets = load_data()
    tokenizer = m.BPETokenizer.load(DATA / "tokenizer.json")
    cfg = m.TransformerConfig(
        vocab_size=len(tokenizer.vocab), block_size=128, batch_size=8,
        d_model=128, n_heads=4, n_layers=2, dropout=0.1,
        learning_rate=3e-4, min_lr=3e-5, warmup_steps=min(200, args.steps // 10),
        max_steps=args.steps, eval_interval=min(500, args.steps), eval_steps=10,
        mlp_type="swiglu", position_encoding="learned",
    )
    if cfg.eval_interval < 1 or cfg.warmup_steps < 1:
        raise ValueError("Too few steps for this schedule")
    torch.manual_seed(42)
    learned = m.GPT(cfg)
    m.init_scaled_residual_projections(learned)
    rope_cfg = replace(cfg, position_encoding="rope")
    rope = m.GPT(rope_cfg)
    shared = {key: value for key, value in learned.state_dict().items() if key in rope.state_dict()}
    rope.load_state_dict(shared)
    output_dir = HERE / "outputs" / ("rope_cpu_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    offsets = torch.arange(cfg.block_size)
    counts = {"learned": sum(p.numel() for p in learned.parameters()),
              "rope": sum(p.numel() for p in rope.parameters())}
    print(f"params={counts}; train_tokens={len(datasets['train'])}; val_tokens={len(datasets['val'])}", flush=True)
    learned_rows = train_variant("learned", learned, cfg, datasets, offsets, output_dir)
    rope_rows = train_variant("rope", rope, rope_cfg, datasets, offsets, output_dir)

    samples = {}
    for name, model in (("learned", learned), ("rope", rope)):
        model.eval()
        torch.manual_seed(100)
        prompt = "Harry looked at"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(prompt_ids, 80, temperature=0.8)
        samples[name] = tokenizer.decode(generated[0, prompt_ids.shape[1]:].tolist())
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    summary = {
        "config": asdict(cfg), "parameter_counts": counts,
        "shared_initial_weights": True,
        "initial": {"learned": learned_rows[0], "rope": rope_rows[0]},
        "final": {"learned": learned_rows[-1], "rope": rope_rows[-1]},
        "samples": samples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
