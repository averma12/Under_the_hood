"""Compare eager PyTorch LayerNorm and RMSNorm on the toy GPT's A100 shape.

Run with the dedicated Modal wrapper:
  /Users/abhinavverma/Desktop/qwen-tts-lab/scripts/modal-tts run \
    projects/07_the-details-that-matter/experiment_rmsnorm_a100.py

This measures the PyTorch operators used by our model, including launch overhead;
it is not a benchmark of a custom fused RMSNorm kernel.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import modal


app = modal.App("under-the-hood-rmsnorm-benchmark")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1")


@app.function(image=image, gpu="A100-40GB", timeout=180)
def benchmark() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(42)
    x = torch.randn(32, 128, 256, device="cuda", requires_grad=True)
    modules = {
        "layernorm": torch.nn.LayerNorm(256, eps=1e-6).cuda(),
        "rmsnorm": torch.nn.RMSNorm(256, eps=1e-6).cuda(),
    }

    def measure(module: torch.nn.Module, backward: bool) -> float:
        def step() -> None:
            if backward:
                x.grad = None
                module.zero_grad(set_to_none=True)
                module(x).square().mean().backward()
            else:
                with torch.no_grad():
                    module(x)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(30):
                step()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(200):
                step()
            torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000 / 200

    samples = {name: {"forward_ms": [], "forward_backward_ms": []} for name in modules}
    for _ in range(3):
        for name, module in modules.items():
            samples[name]["forward_ms"].append(measure(module, backward=False))
            samples[name]["forward_backward_ms"].append(measure(module, backward=True))
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "shape": list(x.shape),
        "results": {
            name: {key: statistics.median(values) for key, values in measurements.items()}
            for name, measurements in samples.items()
        },
        "samples": samples,
    }


@app.local_entrypoint()
def main() -> None:
    result = benchmark.remote()
    output = Path(__file__).resolve().parent / "figures" / "rmsnorm_a100_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")
