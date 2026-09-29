"""Check and time KV-cache decoding on a trained checkpoint.

1. Logits: full forward vs. prefill + one-token decode steps (max |diff|).
2. generate(): recompute vs. KV cache vs. KV cache + split-KV decode,
   same seed, so the sampled tokens should be identical.

Run from the repo root:
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/benchmark_kv_cache.py
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/benchmark_kv_cache.py \
        --checkpoint path/to/checkpoint.pt --device cuda
"""

import argparse
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

import my_gpt as m

HERE = Path(__file__).resolve().parent
RUNS = HERE / "outputs" / "my_gpt_2048" / "modal_runs"


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = m.TransformerConfig(**saved["config"])
    if cfg.attention_impl == "flash" and not device.startswith("cuda"):
        # The fused Flash kernel is GPU-only; SDPA computes the same math on CPU.
        cfg = m.TransformerConfig(**{**asdict(cfg), "attention_impl": "sdpa"})
    cfg.device = device
    model = m.GPT(cfg).to(device).eval()
    model.load_state_dict(saved["model"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=RUNS / "gpt-mhc-flash-stable-30k" / "best.pt")
    parser.add_argument("--tokenizer", type=Path,
                        default=HERE / "outputs" / "my_gpt_2048" / "tokenizer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="Harry looked at the door and")
    parser.add_argument("--new-tokens", type=int, default=100)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load(args.checkpoint, args.device)
    cfg = model.cfg
    tokenizer = m.BPETokenizer.load(args.tokenizer)
    prompt = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    print(f"checkpoint: {args.checkpoint.relative_to(HERE) if args.checkpoint.is_relative_to(HERE) else args.checkpoint}")
    print(f"model: {cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_heads} heads, "
          f"context {cfg.block_size}, attention {cfg.attention_impl}, "
          f"mHC streams {cfg.mhc_streams}, device {args.device}")
    print(f"prompt: {prompt.shape[1]} tokens, generating {args.new_tokens}\n")

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if args.device.startswith("cuda") else nullcontext())
    with torch.inference_mode(), autocast:
        torch.manual_seed(args.seed)
        seq = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=args.device)
        seq[:, :prompt.shape[1]] = prompt
        full, _ = model(seq)
        for splits in (1, args.splits):
            cache = m.KVCache(cfg, decode_splits=splits)
            steps = [model(seq[:, :prompt.shape[1]], kv_cache=cache)[0]]
            steps += [model(seq[:, t:t + 1], kv_cache=cache)[0]
                      for t in range(prompt.shape[1], cfg.block_size)]
            diff = (torch.cat(steps, dim=1) - full).abs().max().item()
            print(f"logits, full vs cached (decode_splits={splits}): max |diff| = {diff:.2e}")

        modes = {
            "recompute (old)": dict(use_kv_cache=False),
            "KV cache": dict(),
            f"KV cache + split-KV x{args.splits}": dict(decode_splits=args.splits),
        }
        results = {}
        for name, kwargs in modes.items():
            model.generate(prompt, 4, **kwargs)  # warm-up
            torch.manual_seed(args.seed)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model.generate(prompt, args.new_tokens, temperature=0.8, **kwargs)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            results[name] = (time.perf_counter() - start, out)

    base_time, base_out = results["recompute (old)"]
    print(f"\n{'mode':<28} {'seconds':>8} {'tokens/s':>9} {'speedup':>8}  same tokens")
    for name, (seconds, out) in results.items():
        print(f"{name:<28} {seconds:>8.3f} {args.new_tokens / seconds:>9.1f} "
              f"{base_time / seconds:>7.2f}x  {torch.equal(out, base_out)}")
    text = tokenizer.decode(results["KV cache"][1][0, prompt.shape[1]:].tolist())
    print(f"\nsample: {args.prompt}{text}")


if __name__ == "__main__":
    main()
