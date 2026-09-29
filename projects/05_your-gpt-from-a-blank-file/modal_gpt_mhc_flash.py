"""Fresh 30k-step mHC + forced FlashAttention A100 run, resumable by time cap.

Run with the dedicated Modal wrapper, e.g.:
  /Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts run \
    projects/05_your-gpt-from-a-blank-file/modal_gpt_mhc_flash.py \
    --run-id gpt-mhc-flash-stable-30k --target-steps 30000 --max-seconds 660
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
VOLUME_NAME = "under-the-hood-gpt-a100"
WRAPPER = "/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "safetensors", "numpy")
    .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(DATA / "tokenizer.json", "/root/data/tokenizer.json")
    .add_local_file(DATA / "train_ids.json", "/root/data/train_ids.json")
    .add_local_file(DATA / "val_ids.json", "/root/data/val_ids.json")
)
app = modal.App("under-the-hood-gpt-mhc-flash")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image, gpu="A100-40GB", cpu=4, memory=8192, timeout=900,
    volumes={"/artifacts": volume},
)
def train_chunk(run_id: str, target_steps: int, max_seconds: int) -> dict:
    import sys
    import time
    import shutil

    import torch
    from torch.profiler import ProfilerActivity, profile
    from safetensors.torch import save_model

    sys.path.insert(0, "/root")
    import my_gpt as m

    if not torch.cuda.is_available():
        raise RuntimeError("A100 CUDA is unavailable")
    volume.reload()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    cfg = m.TransformerConfig(
        vocab_size=2048, block_size=128, batch_size=32, d_model=256,
        n_heads=4, n_layers=4, dropout=0.1, learning_rate=3e-4,
        min_lr=3e-5, warmup_steps=300, max_steps=target_steps,
        eval_interval=500, eval_steps=10, grad_clip=1.0, device="cuda",
        mlp_type="swiglu", norm_type="rmsnorm", position_encoding="learned",
        attention_impl="flash", mhc_streams=4, mhc_sinkhorn_iters=64,
    )
    tokenizer = m.BPETokenizer.load("/root/data/tokenizer.json")
    if len(tokenizer.vocab) != cfg.vocab_size:
        raise ValueError("BPE vocabulary changed")
    datasets = {
        split: torch.tensor(json.loads(Path(f"/root/data/{split}_ids.json").read_text()),
                            dtype=torch.long, device=device)
        for split in ("train", "val")
    }
    offsets = torch.arange(cfg.block_size, device=device)
    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    model = m.GPT(cfg).to(device)
    m.init_scaled_residual_projections(model)
    optimizer = torch.optim.AdamW(
        m.configure_decay_groups(model, weight_decay=0.1),
        lr=cfg.learning_rate, fused=True,
    )
    train_rng = torch.Generator(device=device).manual_seed(42)
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved["config"] != asdict(cfg):
            raise ValueError("Run config changed; refusing incompatible resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        train_rng.set_state(saved["train_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        step = int(saved["step"])
        best_val = float(saved["best_val"])
        elapsed_before = float(saved["elapsed_seconds"])
    else:
        step = 0
        best_val = float("inf")
        elapsed_before = 0.0
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (run_dir / "run_info.json").write_text(json.dumps({
            "variant": "mHC-4 + forced Flash SDPA + RMSNorm + SwiGLU",
            "data": "seven Harry Potter books, preexisting 2048-token BPE split",
            "seed": 42,
        }, indent=2))
        shutil.copyfile("/root/data/tokenizer.json", run_dir / "tokenizer.json")

    def batch(split: str, rng: torch.Generator):
        data = datasets[split]
        starts = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,),
                               generator=rng, device=device)
        positions = starts[:, None] + offsets
        return data[positions], data[positions + 1]

    @torch.no_grad()
    def evaluate():
        model.eval()
        scores = {}
        routes = [(f"block_{i}_{name}", getattr(block, f"{name}_route"))
                  for i, block in enumerate(model.blocks)
                  for name in ("attn", "ffn")]
        for seed, split in ((100, "train"), (101, "val")):
            rng = torch.Generator(device=device).manual_seed(seed)
            losses = []
            for eval_index in range(cfg.eval_steps):
                for _, route in routes:
                    route.record_stats = split == "val" and eval_index == 0
                x, y = batch(split, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                losses.append(loss.item())
            scores[split] = sum(losses) / len(losses)
        for _, route in routes:
            route.record_stats = False
        scores["routing"] = {name: route.last_stats for name, route in routes}
        worst_row_error = max(stats["row_error"] for stats in scores["routing"].values())
        if worst_row_error >= 1e-3:
            raise RuntimeError(f"mHC residual mixing lost row normalization: {worst_row_error:.4g}")
        model.train()
        return scores

    started = time.monotonic()

    def checkpoint(name: str):
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": asdict(cfg), "step": step, "best_val": best_val,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "train_rng": train_rng.get_state(),
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
        }
        path = run_dir / name
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        if name == "best.pt":
            # The .pt remains the full resumable optimizer/RNG checkpoint.
            # This companion file is safe, weight-only inference export; use
            # save_model because token_embedding and lm_head share storage.
            safe_path = run_dir / "best_model.safetensors"
            safe_temporary = run_dir / "best_model.tmp.safetensors"
            save_model(model, str(safe_temporary), metadata={
                "checkpoint_step": str(step),
                "architecture": "GPT-mHC-4",
                "contents": "model weights only",
            })
            safe_temporary.replace(safe_path)
        volume.commit()

    def log_scores(scores, loss=None, grad_norm=None):
        row = {
            "step": step, "lr": m.get_lr(max(0, step - 1), cfg),
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            **scores,
        }
        if loss is not None:
            row["batch_loss"] = loss
            row["grad_norm_before_clip"] = grad_norm
        with (run_dir / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(f"step={step} train={scores['train']:.4f} val={scores['val']:.4f} "
              f"lr={row['lr']:.7f} total={row['elapsed_seconds']:.1f}s", flush=True)
        return row

    if step == 0:
        # The forced backend errors if Flash cannot serve this model. The profiler
        # additionally records the actual CUDA operator rather than guessing.
        x, y = batch("val", torch.Generator(device=device).manual_seed(999))
        model.eval()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, y)
            torch.cuda.synchronize()
        flash_events = sorted({event.key for event in prof.key_averages()
                               if "flash" in event.key.lower()})
        if not flash_events:
            raise RuntimeError("Forced Flash SDPA did not show a Flash profiler event")
        (run_dir / "flash_profile.json").write_text(json.dumps(flash_events, indent=2))
        print(f"Flash kernels: {flash_events}", flush=True)
        scores = evaluate()
        best_val = scores["val"]
        log_scores(scores)
        checkpoint("latest.pt")
        checkpoint("best.pt")
    print(f"GPU={torch.cuda.get_device_name(0)} params={sum(p.numel() for p in model.parameters()):,} "
          f"resuming_at={step}", flush=True)

    model.train()
    last_row = None
    while step < target_steps and time.monotonic() - started < max_seconds:
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
            model.parameters(), cfg.grad_clip, error_if_nonfinite=True,
        )
        optimizer.step()
        step += 1
        if step % cfg.eval_interval == 0 or step == target_steps:
            scores = evaluate()
            last_row = log_scores(scores, loss.item(), grad_norm.item())
            if scores["val"] < best_val:
                best_val = scores["val"]
                checkpoint("best.pt")
        if step % 1000 == 0 or step == target_steps:
            checkpoint("latest.pt")
        if step % 5000 == 0:
            checkpoint(f"step_{step:05d}.pt")
    if step % 1000 != 0 and step != target_steps:
        checkpoint("latest.pt")
    volume.commit()
    return {"run_id": run_id, "step": step, "target_steps": target_steps,
            "best_val": best_val, "last": last_row,
            "elapsed_seconds": elapsed_before + time.monotonic() - started}


@app.local_entrypoint()
def main(run_id: str = "gpt-mhc-flash-stable-30k", target_steps: int = 30000,
         max_seconds: int = 660, max_calls: int = 100):
    if not 1 <= max_seconds <= 720 or target_steps < 2 or max_calls < 1:
        raise ValueError("Invalid steps, time cap, or call count")
    result = None
    for call in range(1, max_calls + 1):
        print(f"A100 training chunk {call}", flush=True)
        result = train_chunk.remote(run_id, target_steps, max_seconds)
        print(json.dumps(result, indent=2), flush=True)
        if result["step"] >= target_steps:
            break
    local_dir = DATA / "modal_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("latest.pt", "best.pt", "metrics.jsonl", "config.json",
                 "run_info.json", "flash_profile.json"):
        temporary = local_dir / f"{name}.download"
        temporary.unlink(missing_ok=True)
        subprocess.run([WRAPPER, "volume", "get", VOLUME_NAME,
                        f"{run_id}/{name}", str(temporary)], check=True)
        temporary.replace(local_dir / name)
    for milestone in range(5000, result["step"] + 1, 5000):
        name = f"step_{milestone:05d}.pt"
        temporary = local_dir / f"{name}.download"
        temporary.unlink(missing_ok=True)
        subprocess.run([WRAPPER, "volume", "get", VOLUME_NAME,
                        f"{run_id}/{name}", str(temporary)], check=True)
        temporary.replace(local_dir / name)
    (local_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"Local artifacts: {local_dir}", flush=True)
