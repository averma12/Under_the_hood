"""Score non-overlapping windows across the entire held-out validation text.

This is a fuller validation check, not an independent test set: validation
data was already used to choose the best checkpoint during training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
VOLUME_NAME = "under-the-hood-gpt-a100"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(DATA / "val_ids.json", "/root/data/val_ids.json")
)
app = modal.App("under-the-hood-gpt-full-validation")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(image=image, gpu="A100-40GB", timeout=900,
              volumes={"/artifacts": volume})
def evaluate_all(run_id: str):
    import sys
    import torch

    sys.path.insert(0, "/root")
    import my_gpt as m

    volume.reload()
    val_ids = torch.tensor(
        json.loads(Path("/root/data/val_ids.json").read_text()),
        dtype=torch.long, device="cuda",
    )
    sources = (
        ("mHC best", Path("/artifacts") / run_id / "best.pt"),
        ("mHC 15k", Path("/artifacts") / run_id / "step_15000.pt"),
        ("mHC 30k", Path("/artifacts") / run_id / "step_30000.pt"),
        ("conventional 30k", Path("/artifacts") /
         "gpt-2048-continued-20260923-065041" / "checkpoint.pt"),
    )
    results = []
    for label, path in sources:
        saved = torch.load(path, map_location="cuda", weights_only=False)
        cfg = m.TransformerConfig(**saved["config"])
        model = m.GPT(cfg).cuda().eval()
        model.load_state_dict(saved["model"])
        offsets = torch.arange(cfg.block_size, device="cuda")
        starts = torch.arange(
            0, len(val_ids) - cfg.block_size, cfg.block_size, device="cuda",
        )
        total_nll = 0.0
        total_tokens = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start_batch in starts.split(cfg.batch_size):
                positions = start_batch[:, None] + offsets
                x, y = val_ids[positions], val_ids[positions + 1]
                _, loss = model(x, y)
                tokens = y.numel()
                total_nll += loss.item() * tokens
                total_tokens += tokens
        val_loss = total_nll / total_tokens
        row = {
            "model": label, "checkpoint_step": int(saved["step"]),
            "val_tokens_scored": total_tokens, "val_tokens_total": len(val_ids),
            "window_stride": cfg.block_size, "context_size": cfg.block_size,
            "val_loss": val_loss, "val_perplexity": math.exp(val_loss),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, saved
    return {"run_id": run_id, "full_validation": results}


@app.local_entrypoint()
def main(run_id: str = "gpt-mhc-flash-stable-30k"):
    results = evaluate_all.remote(run_id)
    local_dir = DATA / "modal_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "full_validation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(f"Saved {local_dir / 'full_validation.json'}", flush=True)
