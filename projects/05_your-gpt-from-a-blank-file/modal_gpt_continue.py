"""Continue the Chapter 6 or SwiGLU Harry Potter BPE GPT from its Modal checkpoint.

Run through the dedicated Modal wrapper, with an explicit GPU time cap:
  /Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts run \
    projects/05_your-gpt-from-a-blank-file/modal_gpt_continue.py \
    --resume-run-id gpt-2048-chapter7_swiglu-20260923-062304 \
    --extra-steps 10000 --checkpoint-every 5000 \
    --max-seconds 600 --end-lr 0.00001
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
VOLUME_NAME = "under-the-hood-gpt-a100"
WRAPPER = "/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(DATA / "tokenizer.json", "/root/data/tokenizer.json")
    .add_local_file(DATA / "train_ids.json", "/root/data/train_ids.json")
    .add_local_file(DATA / "val_ids.json", "/root/data/val_ids.json")
)
app = modal.App("under-the-hood-gpt-a100-continue")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={"/artifacts": volume},
)
def continue_on_a100(
    run_id: str,
    resume_run_id: str,
    extra_steps: int,
    checkpoint_every: int,
    max_seconds: int,
    end_lr: float,
) -> dict:
    import sys
    import time
    from dataclasses import asdict

    import torch

    sys.path.insert(0, "/root")
    import my_gpt as m

    if not torch.cuda.is_available():
        raise RuntimeError("A100 CUDA device is unavailable.")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    source = Path("/artifacts") / resume_run_id / "checkpoint.pt"
    saved = torch.load(source, map_location=device, weights_only=False)
    variant = saved.get("variant")
    if variant not in ("chapter6", "chapter7_swiglu", "chapter7_rmsnorm_swiglu"):
        raise ValueError("Continuation requires a Chapter 6 or Chapter 7 checkpoint.")
    cfg = m.TransformerConfig(**saved["config"])
    if cfg.device != "cuda" or cfg.vocab_size != 2048:
        raise ValueError("Checkpoint does not match the A100 Harry Potter BPE model.")
    expected_mlp = "gelu" if variant == "chapter6" else "swiglu"
    expected_norm = "rmsnorm" if variant == "chapter7_rmsnorm_swiglu" else "layernorm"
    if cfg.mlp_type != expected_mlp:
        raise ValueError("Checkpoint variant and MLP type disagree.")
    if cfg.norm_type != expected_norm:
        raise ValueError("Checkpoint variant and normalization type disagree.")
    if checkpoint_every % cfg.eval_interval:
        raise ValueError("checkpoint_every must be a multiple of eval_interval.")
    start_step = int(saved["step"])
    start_lr = float(saved["optimizer"]["param_groups"][0]["lr"])
    if not 0 < end_lr <= start_lr:
        raise ValueError("End LR must be positive and no greater than checkpoint LR.")

    tokenizer = m.BPETokenizer.load("/root/data/tokenizer.json")
    if len(tokenizer.vocab) != cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from the checkpoint.")
    datasets = {
        split: torch.tensor(
            json.loads(Path(f"/root/data/{split}_ids.json").read_text()),
            dtype=torch.long,
            device=device,
        )
        for split in ("train", "val")
    }
    model = m.GPT(cfg).to(device)
    model.load_state_dict(saved["model"])
    groups = m.configure_decay_groups(model, weight_decay=0.1)
    optimizer = torch.optim.AdamW(groups, lr=start_lr, fused=True)
    optimizer.load_state_dict(saved["optimizer"])
    if [group["weight_decay"] for group in optimizer.param_groups] != [0.1, 0.0]:
        raise ValueError("Checkpoint optimizer does not have Chapter 6 parameter groups.")
    train_rng = torch.Generator(device=device)
    train_rng.set_state(saved["train_rng"].cpu())
    torch.set_rng_state(saved["torch_rng"].cpu())
    torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    offsets = torch.arange(cfg.block_size, device=device)

    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    continuation = {
        "resume_run_id": resume_run_id,
        "start_step": start_step,
        "extra_steps": extra_steps,
        "checkpoint_every": checkpoint_every,
        "target_step": start_step + extra_steps,
        "start_lr": start_lr,
        "end_lr": end_lr,
        "schedule": "cosine continuation, no warmup",
        "variant": variant,
    }
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "continuation.json").write_text(json.dumps(continuation, indent=2))

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
        scores = {}
        for seed, split in enumerate(("train", "val"), start=100):
            rng = torch.Generator(device=device).manual_seed(seed)
            losses = []
            for _ in range(cfg.eval_steps):
                x, y = batch(split, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                losses.append(loss.item())
            scores[split] = sum(losses) / len(losses)
        model.train()
        return scores

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
            "resume_run_id": resume_run_id,
            "continuation": continuation,
        }
        temporary = (run_dir / name).with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(run_dir / name)
        volume.commit()

    def decode_samples(step: int):
        # Generation seeds the global RNG. Restore it so periodic samples cannot
        # change dropout draws in the subsequent training steps.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        try:
            samples = []
            for seed, prompt in (
                (100, "Harry looked at"),
                (101, "Hermione said,"),
                (102, "The door opened and"),
            ):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                generated = model.generate(ids, 80, temperature=0.8)
                samples.append({
                    "step": step,
                    "prompt": prompt,
                    "continuation": tokenizer.decode(generated[0, ids.shape[1]:].tolist()),
                    "seed": seed,
                })
            return samples
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)

    initial = evaluate()
    if abs(initial["val"] - saved["best_val"]) > 0.001:
        raise RuntimeError("Resumed checkpoint does not reproduce the previous validation loss.")
    best_val = float(saved["best_val"])
    checkpoint("best.pt", start_step, best_val)
    started = time.monotonic()
    completed = start_step
    milestones = []
    milestone_samples = {}
    rows = [{"step": start_step, "lr": start_lr, **initial}]
    print(
        f"A100={torch.cuda.get_device_name(0)} | variant={variant} | resumed step={start_step} "
        f"train={initial['train']:.4f} val={initial['val']:.4f} lr={start_lr:.6f}",
        flush=True,
    )
    with (run_dir / "metrics.jsonl").open("w") as log:
        log.write(json.dumps(rows[0]) + "\n")
        log.flush()
        for local_step in range(extra_steps):
            if time.monotonic() - started >= max_seconds:
                print(f"Stopped at time cap after global step {completed}", flush=True)
                break
            progress = local_step / (extra_steps - 1)
            lr = end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = batch("train", train_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {completed + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            completed += 1
            if completed % cfg.eval_interval == 0 or local_step == extra_steps - 1:
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
                    f"step={completed} lr={lr:.7f} train={scores['train']:.4f} "
                    f"val={scores['val']:.4f} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                if scores["val"] < best_val:
                    best_val = scores["val"]
                    checkpoint("best.pt", completed, best_val)
                if (completed - start_step) % checkpoint_every == 0:
                    checkpoint(f"step_{completed}.pt", completed, best_val)
                    milestone_samples[completed] = decode_samples(completed)
                    (run_dir / f"samples_step_{completed}.json").write_text(
                        json.dumps(milestone_samples[completed], indent=2, ensure_ascii=False)
                    )
                    milestones.append(completed)
                    volume.commit()
        if completed != rows[-1]["step"]:
            scores = evaluate()
            row = {
                "step": completed,
                "lr": optimizer.param_groups[0]["lr"],
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

    samples = milestone_samples.get(completed) or decode_samples(completed)
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    volume.commit()
    return {
        "run_id": run_id,
        "resume_run_id": resume_run_id,
        "variant": variant,
        "start_step": start_step,
        "completed_step": completed,
        "milestones": milestones,
        "best_val": best_val,
        "initial": rows[0],
        "final": rows[-1],
        "samples": samples,
        "elapsed_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main(
    resume_run_id: str,
    extra_steps: int = 5000,
    checkpoint_every: int = 5000,
    max_seconds: int = 600,
    end_lr: float = 1e-5,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", resume_run_id):
        raise ValueError("Invalid source run ID.")
    if extra_steps < 2 or not 1 <= checkpoint_every <= extra_steps or not 1 <= max_seconds <= 720:
        raise ValueError("Require at least 2 steps, a valid checkpoint interval, and a 1–720 second GPU time cap.")
    run_id = "gpt-2048-continued-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = continue_on_a100.remote(
        run_id, resume_run_id, extra_steps, checkpoint_every, max_seconds, end_lr
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    local_dir = DATA / "modal_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint.pt", "metrics.jsonl", "samples.json", "config.json", "continuation.json"):
        subprocess.run(
            [WRAPPER, "volume", "get", VOLUME_NAME, f"{run_id}/{name}", str(local_dir / name)],
            check=True,
        )
    for step in result["milestones"]:
        for name in (f"step_{step}.pt", f"samples_step_{step}.json"):
            if step == result["completed_step"]:
                source = "checkpoint.pt" if name.endswith(".pt") else "samples.json"
                shutil.copyfile(local_dir / source, local_dir / name)
                continue
            subprocess.run(
                [WRAPPER, "volume", "get", VOLUME_NAME, f"{run_id}/{name}", str(local_dir / name)],
                check=True,
            )
    (local_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Local results: {local_dir.resolve()}")
