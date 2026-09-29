"""Measure eager versus compiled mHC Sinkhorn forward/backward on A100."""

import modal
from pathlib import Path

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1")
         .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py"))
app = modal.App("under-the-hood-mhc-sinkhorn-benchmark")


@app.function(image=image, gpu="A100-40GB", timeout=900)
def benchmark():
    import sys
    import time
    import torch
    sys.path.insert(0, "/root")
    import my_gpt as m

    x = torch.randn(32, 128, 4, 4, device="cuda").requires_grad_()
    results = {}
    for label, fn in (("eager", m.sinkhorn_doubly_stochastic),
                      ("compiled", torch.compile(m.sinkhorn_doubly_stochastic,
                                                 fullgraph=True, dynamic=False))):
        started = time.monotonic()
        for _ in range(2):
            x.grad = None
            y = fn(x, 64)
            (y.square().mean()).backward()
        torch.cuda.synchronize()
        compile_and_warmup = time.monotonic() - started
        started = time.monotonic()
        for _ in range(20):
            x.grad = None
            y = fn(x, 64)
            (y.square().mean()).backward()
        torch.cuda.synchronize()
        results[label] = {
            "warmup_seconds": compile_and_warmup,
            "seconds_per_forward_backward": (time.monotonic() - started) / 20,
            "row_error": (y.sum(-1) - 1).abs().max().item(),
        }
        print(label, results[label], flush=True)
    return results


@app.local_entrypoint()
def main():
    print(benchmark.remote(), flush=True)
