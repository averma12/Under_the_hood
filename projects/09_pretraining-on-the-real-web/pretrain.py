"""Chapter 9 pretraining loop for our Chapter 5 GPT on memory-mapped web shards.

What is new compared with the Chapter 5 trainer:
- data comes from pre-tokenized uint16 shards via np.memmap (pretrain_data.py)
- gradient accumulation: several micro-batches per optimizer update
- BF16 autocast and the fused FlashAttention kernel on GPU
- validation bits-per-byte (bpb) on a fixed held-out token set
- throughput logging: tokens/s and model FLOPs utilization (MFU)
- time-capped, resumable runs (model, optimizer, and data RNG state)

Local CPU smoke test (build tiny shards first, see prepare_local.py):
    .venv/bin/python projects/09_pretraining-on-the-real-web/pretrain.py \
        --data-dir /tmp/shards --out-dir /tmp/run --preset tiny \
        --total-tokens 200000 --batch-tokens 4096 --micro-batch 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "05_your-gpt-from-a-blank-file"))

import my_gpt as m  # noqa: E402
from pretrain_data import (PADDED_VOCAB_SIZE, ShardLoader, bits_per_byte,  # noqa: E402
                           get_encoder, token_byte_lengths)

PRESETS = {
    # CPU smoke test: checks the pipeline end to end in seconds.
    "tiny": dict(n_layers=2, d_model=64, n_heads=2, block_size=64),
    # A100 run: ~8 layers x 512 wide, context 1024.
    "d512": dict(n_layers=8, d_model=512, n_heads=8, block_size=1024),
}
A100_BF16_FLOPS = 312e12


def make_config(args, device):
    return m.TransformerConfig(
        vocab_size=PADDED_VOCAB_SIZE, batch_size=args.micro_batch, dropout=0.0,
        learning_rate=args.lr, min_lr=args.lr * args.min_lr_ratio,
        warmup_steps=args.warmup_steps, max_steps=args.total_tokens // args.batch_tokens,
        grad_clip=1.0, device=device, mlp_type="swiglu", norm_type="rmsnorm",
        position_encoding="rope",
        attention_impl="flash" if device.startswith("cuda") else "sdpa",
        **PRESETS[args.preset],
    )


def flops_per_token(model, cfg):
    """Training FLOPs per token: 6 per parameter plus attention's T-dependent part."""
    params = sum(p.numel() for p in model.parameters())
    return 6 * params + 12 * cfg.n_layers * cfg.d_model * cfg.block_size


@torch.no_grad()
def evaluate(model, loader, byte_lengths, num_tokens, batch_size, device, autocast):
    """Loss per token and bits per byte over the same fixed windows every time.

    Targets that are <|endoftext|> decode to zero bytes, so they are left out
    of both the nats and the bytes in bpb.
    """
    was_training = model.training
    model.eval()
    nats = nats_text = tokens = text_bytes = 0.0
    for x, y in loader.fixed_windows(num_tokens, batch_size, device):
        with autocast:
            logits, _ = model(x)
        losses = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        target_bytes = byte_lengths[y.flatten()]
        nats += losses.sum().item()
        tokens += losses.numel()
        nats_text += losses[target_bytes > 0].sum().item()
        text_bytes += target_bytes.sum().item()
    model.train(was_training)
    return {"loss": nats / tokens, "bpb": bits_per_byte(nats_text, text_bytes),
            "tokens": int(tokens), "bytes_per_token": text_bytes / tokens}


def train(args):
    device = args.device
    cuda = device.startswith("cuda")
    if args.batch_tokens % (args.micro_batch * PRESETS[args.preset]["block_size"]):
        raise ValueError("batch_tokens must be a multiple of micro_batch * block_size.")
    cfg = make_config(args, device)
    grad_accum = args.batch_tokens // (args.micro_batch * cfg.block_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    model = m.GPT(cfg).to(device)
    m.init_scaled_residual_projections(model)
    optimizer = torch.optim.AdamW(m.configure_decay_groups(model, 0.1), lr=cfg.learning_rate,
                                  betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    train_loader = ShardLoader(args.data_dir, "train", cfg.block_size, seed=args.seed,
                               limit_tokens=args.train_limit_tokens)
    val_loader = ShardLoader(args.data_dir, "val", cfg.block_size)
    byte_lengths = token_byte_lengths(get_encoder()).to(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()

    step = 0
    latest = out_dir / "latest.pt"
    if latest.exists():
        # Load on CPU: set_rng_state needs a CPU tensor, and load_state_dict
        # moves weights and optimizer state to the parameters' device.
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        train_loader.load_state_dict(saved["train_loader"])
        torch.set_rng_state(saved["cpu_rng"])
        step = saved["step"]
        print(f"resumed from step {step}", flush=True)
    else:
        (out_dir / "config.json").write_text(json.dumps(
            {"model": asdict(cfg), "args": vars(args), "grad_accum": grad_accum,
             "params": sum(p.numel() for p in model.parameters())}, indent=2))

    step_model = torch.compile(model) if args.compile else model
    flops_token = flops_per_token(model, cfg)
    metrics = (out_dir / "metrics.jsonl").open("a")

    def log(record):
        print(json.dumps(record), flush=True)
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()

    def run_eval():
        val = evaluate(model, val_loader, byte_lengths, args.eval_tokens, args.micro_batch,
                       device, autocast)
        # Held-in windows from the first training shard, for the train/val gap.
        held_in = ShardLoader(args.data_dir, "train", cfg.block_size)
        train_eval = evaluate(model, held_in, byte_lengths, args.eval_tokens, args.micro_batch,
                              device, autocast)
        log({"step": step, "tokens_seen": step * args.batch_tokens,
             "val_loss": val["loss"], "val_bpb": val["bpb"],
             "train_loss_fixed": train_eval["loss"], "train_bpb_fixed": train_eval["bpb"]})
        return val

    def save():
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "train_loader": train_loader.state_dict(), "cpu_rng": torch.get_rng_state(),
                 "step": step, "config": asdict(cfg)}
        torch.save(state, out_dir / "latest.pt.tmp")
        (out_dir / "latest.pt.tmp").replace(latest)

    started = time.perf_counter()
    window_start, window_tokens = time.perf_counter(), 0
    if step == 0:
        run_eval()
    val = None
    while step < cfg.max_steps:
        lr = m.get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        for _ in range(grad_accum):
            x, y = train_loader.batch(args.micro_batch, device)
            with autocast:
                _, loss = step_model(x, y)
            # Average over micro-batches: the gradient equals one big batch's.
            (loss / grad_accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_tokens += args.batch_tokens

        if step % args.log_interval == 0 or step == cfg.max_steps:
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            tokens_per_s = window_tokens / elapsed
            log({"step": step, "lr": lr, "train_loss": (loss_sum / grad_accum).item(),
                 "grad_norm": grad_norm.item(), "tokens_per_s": round(tokens_per_s),
                 "mfu": tokens_per_s * flops_token / A100_BF16_FLOPS if cuda else None})
            window_start, window_tokens = time.perf_counter(), 0
        if step % args.eval_interval == 0 or step == cfg.max_steps:
            val = run_eval()
        if args.max_seconds and time.perf_counter() - started > args.max_seconds:
            break
    save()
    metrics.close()
    done = step >= cfg.max_steps
    summary = {"step": step, "max_steps": cfg.max_steps, "done": done,
               "tokens_seen": step * args.batch_tokens, "lr": args.lr,
               "val_bpb": val["bpb"] if val else None}
    if done:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--total-tokens", type=int, default=200_000)
    p.add_argument("--batch-tokens", type=int, default=4096, help="tokens per optimizer update")
    p.add_argument("--micro-batch", type=int, default=8, help="sequences per forward pass")
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--eval-tokens", type=int, default=16_384)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=0, help="stop and save after this long")
    p.add_argument("--train-limit-tokens", type=int, default=0,
                   help="BREAK IT: train only on the first N tokens (repeats data)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
