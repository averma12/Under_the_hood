"""Export the best mHC training checkpoint as verified inference weights.

Run from the repository root after installing safetensors:
  uv pip install --python .venv/bin/python safetensors
  .venv/bin/python projects/05_your-gpt-from-a-blank-file/export_mhc_safetensors.py

The original .pt remains the resumable training checkpoint. This exporter
creates model-only safetensors plus the exact model config and BPE tokenizer.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from my_gpt import GPT, TransformerConfig


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
DEFAULT_RUN_DIR = DATA / "modal_runs" / "gpt-mhc-flash-stable-30k"


def export(run_dir: Path, checkpoint_name: str = "best.pt") -> dict:
    source = run_dir / checkpoint_name
    saved = torch.load(source, map_location="cpu", weights_only=True)
    saved_config = TransformerConfig(**saved["config"])

    # The stored config forces a CUDA Flash kernel. The weights are independent
    # of the attention implementation, so use CPU SDPA for round-trip testing.
    cpu_config = TransformerConfig(**{**asdict(saved_config), "attention_impl": "sdpa"})
    original = GPT(cpu_config).eval()
    original.load_state_dict(saved["model"], strict=True)

    stem = "best_model" if checkpoint_name == "best.pt" else Path(checkpoint_name).stem + "_model"
    weights_path = run_dir / f"{stem}.safetensors"
    temporary = weights_path.with_suffix(".tmp.safetensors")
    save_model(
        original, str(temporary),
        metadata={"checkpoint_step": str(saved["step"]),
                  "architecture": "GPT-mHC-4", "contents": "model weights only"},
    )
    temporary.replace(weights_path)

    restored = GPT(cpu_config).eval()
    missing, unexpected = load_model(restored, str(weights_path), strict=True)
    if missing or unexpected:
        raise AssertionError(f"Safetensors round trip failed: {missing=}, {unexpected=}")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        prompt = torch.tensor([[72, 97, 114, 114, 121]])
        expected, _ = original(prompt)
        actual, _ = restored(prompt)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    config_path = run_dir / f"{stem}_config.json"
    config_path.write_text(json.dumps({
        "config": asdict(saved_config), "checkpoint_step": saved["step"],
        "weights_file": weights_path.name,
        "tokenizer_file": f"{stem}_tokenizer.json",
    }, indent=2), encoding="utf-8")
    tokenizer_path = run_dir / f"{stem}_tokenizer.json"
    shutil.copyfile(DATA / "tokenizer.json", tokenizer_path)

    return {
        "checkpoint_step": saved["step"], "weights": str(weights_path),
        "size_bytes": weights_path.stat().st_size,
        "config": str(config_path), "tokenizer": str(tokenizer_path),
        "verification": "all state tensors and fixed-prompt logits match exactly",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
