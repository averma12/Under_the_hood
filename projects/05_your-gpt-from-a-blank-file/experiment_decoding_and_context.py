"""Two local experiments on the saved 5,000-step A100 checkpoint.

Run from the repo root:
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/experiment_decoding_and_context.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
RUN = DATA / "modal_runs" / "gpt-2048-20260922-185701"


def load_model():
    spec = importlib.util.spec_from_file_location("my_gpt", HERE / "my_gpt.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = module.TransformerConfig(**checkpoint["config"])
    cfg.device = "cpu"
    model = module.GPT(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = module.BPETokenizer.load(DATA / "tokenizer.json")
    return model, tokenizer


@torch.no_grad()
def sample(model, tokenizer, prompt, *, seed, temperature=0.8, top_k=None, top_p=None):
    torch.manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    prompt_size = ids.shape[1]
    for _ in range(80):
        logits, _ = model(ids[:, -model.cfg.block_size:])
        scores = logits[:, -1, :] / temperature
        if top_k is not None:
            cutoff = torch.topk(scores, top_k).values[:, -1, None]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))
        if top_p is not None:
            sorted_scores, sorted_ids = torch.sort(scores, dim=-1, descending=True)
            sorted_probs = sorted_scores.softmax(dim=-1)
            drop = sorted_probs.cumsum(dim=-1) > top_p
            # Keep the first token that crosses the cumulative probability.
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(drop, float("-inf"))
            scores = torch.empty_like(scores).scatter(-1, sorted_ids, sorted_scores)
        ids = torch.cat([ids, torch.multinomial(scores.softmax(dim=-1), 1)], dim=1)
    return tokenizer.decode(ids[0, prompt_size:].tolist())


def experiment_decoding(model, tokenizer):
    modes = (
        ("original t=0.8", {"temperature": 0.8}),
        ("top-k 40 t=0.8", {"temperature": 0.8, "top_k": 40}),
        ("top-p 0.9 t=0.8", {"temperature": 0.8, "top_p": 0.9}),
    )
    prompts = ("Harry looked at", "Hermione said,", "The door opened and")
    results = []
    for index, prompt in enumerate(prompts):
        for label, options in modes:
            text = sample(model, tokenizer, prompt, seed=100 + index, **options)
            results.append({"prompt": prompt, "mode": label, "continuation": text})
            print(f"{prompt!r} | {label}: {text!r}", flush=True)
    return results


@torch.no_grad()
def experiment_context(model):
    val_ids = torch.tensor(json.loads((DATA / "val_ids.json").read_text()), dtype=torch.long)
    window = model.cfg.block_size
    generator = torch.Generator().manual_seed(2026)
    starts = torch.randint(len(val_ids) - window, (512,), generator=generator)
    scores = {32: [], 64: [], 128: []}
    for batch_starts in starts.split(32):
        tokens = val_ids[batch_starts[:, None] + torch.arange(window)[None, :]]
        target = val_ids[batch_starts + window]
        for length in scores:
            logits, _ = model(tokens[:, -length:])
            losses = F.cross_entropy(logits[:, -1, :], target, reduction="none")
            scores[length].append(losses)
    per_example = {length: torch.cat(parts) for length, parts in scores.items()}
    results = {}
    for length, losses in per_example.items():
        results[length] = {
            "mean_nats_per_token": losses.mean().item(),
            "standard_error": losses.std(unbiased=True).item() / math.sqrt(len(losses)),
            "examples": len(losses),
        }
        print(f"Context {length:>3}: {results[length]['mean_nats_per_token']:.4f} "
              f"± {results[length]['standard_error']:.4f} nats / token")
    for length in (32, 64):
        difference = per_example[length] - per_example[128]
        results[length]["minus_128_nats"] = difference.mean().item()
        results[length]["difference_standard_error"] = (
            difference.std(unbiased=True).item() / math.sqrt(len(difference))
        )
        print(f"Context {length} − 128: {difference.mean().item():+.4f} "
              f"± {results[length]['difference_standard_error']:.4f} nats / token")
    return results


def main():
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    print("EXPERIMENT 1: Decoding settings")
    decoding = experiment_decoding(model, tokenizer)
    print("EXPERIMENT 2: Same held-out next tokens, different context lengths")
    context = experiment_context(model)
    output = RUN / "decoding_and_context_experiments.json"
    output.write_text(json.dumps({"decoding": decoding, "context": context}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
