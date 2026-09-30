"""Chapter 9 on Modal: FineWeb-Edu shards, an LR sweep, and one main A100 run.

Always run through the dedicated wrapper:
  W=/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts
  $W run projects/09_pretraining-on-the-real-web/modal_pretrain.py --mode prepare --tokens 1100000000
  $W run projects/09_pretraining-on-the-real-web/modal_pretrain.py --mode smoke
  $W run projects/09_pretraining-on-the-real-web/modal_pretrain.py --mode sweep --tokens 150000000
  $W run projects/09_pretraining-on-the-real-web/modal_pretrain.py --mode train --lr 1e-3 --tokens 1000000000

Each GPU call is time-capped and resumable; the local entrypoint keeps calling
until the run reaches its token budget, then downloads the small artifacts.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MY_GPT = HERE.parent / "05_your-gpt-from-a-blank-file" / "my_gpt.py"
VOLUME_NAME = "under-the-hood-pretrain"
WRAPPER = "/Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts"
DATA_DIR = "/vol/fineweb_edu_gpt2"
LOCAL_RUNS = HERE / "outputs" / "modal_runs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "tiktoken", "datasets")
    .add_local_file(MY_GPT, "/root/my_gpt.py")
    .add_local_file(HERE / "pretrain_data.py", "/root/pretrain_data.py")
    .add_local_file(HERE / "prepare_fineweb.py", "/root/prepare_fineweb.py")
    .add_local_file(HERE / "pretrain.py", "/root/pretrain.py")
)
app = modal.App("under-the-hood-pretrain")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Shared settings for every GPU run, so sweep and main runs are comparable.
COMMON = ["--data-dir", DATA_DIR, "--preset", "d512", "--device", "cuda",
          "--batch-tokens", str(2**18), "--micro-batch", "16", "--eval-tokens", str(2**20),
          "--log-interval", "10", "--compile"]


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600,
              volumes={"/vol": volume})
def prepare_data(max_tokens: int, shard_tokens: int, val_tokens: int) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from prepare_fineweb import fineweb_documents, prepare

    manifest = prepare(fineweb_documents(), DATA_DIR, max_tokens, shard_tokens, val_tokens,
                       source="HuggingFaceFW/fineweb-edu sample-10BT (streamed, in order)")
    volume.commit()
    return {k: v for k, v in manifest.items() if k != "shards"} | {
        "shards": len(manifest["shards"])}


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, timeout=3600,
              volumes={"/vol": volume})
def train_chunk(run_id: str, extra_args: list[str], max_seconds: int) -> dict:
    import sys

    import torch

    sys.path.insert(0, "/root")
    import pretrain

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    volume.reload()
    args = pretrain.parse_args([*COMMON, "--out-dir", f"/vol/runs/{run_id}",
                                "--max-seconds", str(max_seconds), *extra_args])
    summary = pretrain.train(args)
    summary["gpu"] = torch.cuda.get_device_name()
    volume.commit()
    return summary


@app.function(image=image, gpu="A100-40GB", timeout=900, volumes={"/vol": volume})
def sample(run_id: str, new_tokens: int = 120, seeds: int = 2) -> str:
    """Continue a few fixed prompts with the final checkpoint (KV-cache generate)."""
    import sys

    import torch

    sys.path.insert(0, "/root")
    import my_gpt as m
    from pretrain_data import get_encoder

    saved = torch.load(f"/vol/runs/{run_id}/latest.pt", map_location="cuda", weights_only=False)
    model = m.GPT(m.TransformerConfig(**saved["config"])).cuda().eval()
    model.load_state_dict(saved["model"])
    enc = get_encoder()
    prompts = ["The water cycle begins when", "In 1905, Albert Einstein",
               "To solve a quadratic equation,", "The main causes of the French Revolution were"]
    lines = [f"# {run_id}, step {saved['step']}, temperature 0.8"]
    for prompt in prompts:
        for seed in range(seeds):
            torch.manual_seed(seed)
            ids = torch.tensor([enc.encode_ordinary(prompt)], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(ids, new_tokens, temperature=0.8)
            text = enc.decode(out[0, ids.shape[1]:].tolist())
            lines.append(f"\n## {prompt!r} (seed {seed})\n{prompt}{text}")
    return "\n".join(lines)


def run_to_completion(run_id, extra_args, max_seconds, max_calls=20):
    for call in range(1, max_calls + 1):
        result = train_chunk.remote(run_id, extra_args, max_seconds)
        print(f"[{run_id}] chunk {call}: {json.dumps(result)}", flush=True)
        if result["done"]:
            break
    local = LOCAL_RUNS / run_id
    local.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.jsonl", "config.json", "summary.json"):
        subprocess.run([WRAPPER, "volume", "get", "--force", VOLUME_NAME,
                        f"runs/{run_id}/{name}", str(local / name)], check=False)
    return result


@app.local_entrypoint()
def main(mode: str = "smoke", tokens: int = 0, lr: float = 1e-3, run_id: str = "",
         max_seconds: int = 3000, lrs: str = "3e-4,1e-3,3e-3,6e-3", warmup_steps: int = 0,
         data_dir: str = "", eval_interval: int = 100):
    if mode == "prepare":
        print(json.dumps(prepare_data.remote(tokens or 1_100_000_000, 100_000_000,
                                             10_000_000), indent=2))
    elif mode == "smoke":
        # ~50 updates at the real model size: measures tokens/s and MFU.
        tokens = tokens or 50 * 2**18
        run_to_completion(run_id or "smoke", ["--lr", str(lr), "--total-tokens", str(tokens),
                                              "--warmup-steps", "10", "--eval-interval",
                                              "25"], max_seconds)
    elif mode == "sweep":
        tokens = tokens or 150_000_000
        warmup = warmup_steps or 100
        jobs = {f"sweep-lr{value}": ["--lr", value, "--total-tokens", str(tokens),
                                     "--warmup-steps", str(warmup), "--eval-interval", "50"]
                for value in lrs.split(",")}
        with ThreadPoolExecutor(len(jobs)) as pool:
            results = dict(zip(jobs, pool.map(
                lambda item: run_to_completion(item[0], item[1], max_seconds), jobs.items())))
        print(json.dumps(results, indent=2))
    elif mode == "breakit":
        # Same model and LR, but only 5M distinct training tokens seen ~16 times.
        tokens = tokens or 80_000_000
        run_to_completion(run_id or "breakit-5M-repeated", [
            "--lr", str(lr), "--total-tokens", str(tokens), "--train-limit-tokens",
            "5000000", "--warmup-steps", "50", "--eval-interval", "25"], max_seconds)
    elif mode == "sample":
        print(sample.remote(run_id or f"main-lr{lr}"))
    elif mode == "train":
        tokens = tokens or 1_000_000_000
        # --data-dir given here overrides COMMON's (argparse keeps the last value).
        run_to_completion(run_id or f"main-lr{lr}", [
            "--lr", str(lr), "--total-tokens", str(tokens),
            "--warmup-steps", str(warmup_steps or 200), "--eval-interval", str(eval_interval),
            *(["--data-dir", data_dir] if data_dir else [])], max_seconds)
    else:
        raise ValueError("mode must be prepare, smoke, sweep, breakit, sample, or train")
