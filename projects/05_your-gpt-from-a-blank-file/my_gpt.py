"""Our Chapter 5 build, filled in step by step.

Run from the repo root using the project environment:
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/my_gpt.py
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/my_gpt.py --sample-batch
    .venv/bin/python projects/05_your-gpt-from-a-blank-file/my_gpt.py --forward-batch

Training: --train --steps 3000 --warmup-steps 300
Continue: --train --resume /path/to/checkpoint.pt --steps 10000
On resume, --steps means additional updates; the original LR schedule is retained.
"""

# 1. Imports
import argparse
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# 2. Transformer config (small starting values, adjustable as we build)
@dataclass
class TransformerConfig:
    vocab_size: int = 512
    block_size: int = 64
    batch_size: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 20
    grad_clip: float = 1.0
    device: str = "cpu"
    mlp_type: str = "gelu"
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    rope_base: float = 10000.0
    attention_impl: str = "manual"  # manual, sdpa, or forced flash
    mhc_streams: int = 1             # 1 retains the original residual block
    mhc_sinkhorn_iters: int = 64


# 3. Simple byte-pair encoding (BPE) tokenizer
class BPETokenizer:
    """Start with 256 byte tokens; repeatedly merge the most common pair.

    Split text into words, punctuation, and whitespace first. Merges stay
    inside those pieces. Counting unique pieces makes training on books cheap.
    All UTF-8 bytes are supported, including bytes absent from training text.
    """

    pattern = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)

    def __init__(self):
        self.merges = {}  # (left token, right token) -> new token ID
        self.vocab = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def merge_pair(tokens, pair, new_id):
        result = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return tuple(result)

    def train(self, text, vocab_size=512):
        if vocab_size < 256:
            raise ValueError("Byte-level BPE needs at least 256 tokens.")
        self.__init__()
        pieces = Counter(
            tuple(piece.encode("utf-8")) for piece in self.pattern.findall(text)
        )
        while len(self.vocab) < vocab_size:
            counts = Counter()
            for tokens, frequency in pieces.items():
                for pair in zip(tokens, tokens[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            new_id = len(self.vocab)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            updated = Counter()
            for tokens, frequency in pieces.items():
                updated[self.merge_pair(tokens, pair, new_id)] += frequency
            pieces = updated
            if (new_id + 1) % 64 == 0:
                print(f"BPE vocabulary: {new_id + 1}/{vocab_size}", flush=True)

    def encode(self, text):
        cache = {}
        result = []
        for piece in self.pattern.findall(text):
            if piece not in cache:
                tokens = tuple(piece.encode("utf-8"))
                while len(tokens) > 1:
                    # Lower IDs are earlier merges: apply learned ranks in order.
                    pair = min(
                        zip(tokens, tokens[1:]),
                        key=lambda p: self.merges.get(p, float("inf")),
                    )
                    if pair not in self.merges:
                        break
                    tokens = self.merge_pair(tokens, pair, self.merges[pair])
                cache[piece] = tokens
            result.extend(cache[piece])
        return result

    def decode(self, ids):
        # Arbitrary model samples can end in an incomplete UTF-8 sequence.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for a, b, idx in payload["merges"]:
            tokenizer.merges[(a, b)] = idx
            tokenizer.vocab[idx] = tokenizer.vocab[a] + tokenizer.vocab[b]
        return tokenizer


# 4. Data loading and train/validation split
def load_text_splits(path, train_fraction=0.9):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    text = Path(path).read_text(encoding="utf-8")
    split = int(len(text) * train_fraction)
    if not 0 < split < len(text):
        raise ValueError("Need enough text for nonempty train and validation splits.")
    # Split BEFORE training BPE so held-out text cannot affect learned merges.
    return text[:split], text[split:]


# 5. Batch sampler
def get_batch(split, datasets, cfg, generator=None):
    """Pick random contiguous windows; targets are shifted one token ahead.

    datasets holds CPU torch.long tensors under 'train' and 'val'.
    Each call samples fresh starts, with replacement, within the chosen split.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'.")
    if cfg.block_size < 1 or cfg.batch_size < 1:
        raise ValueError("block_size and batch_size must be positive.")
    data = datasets[split]
    if data.ndim != 1 or data.dtype != torch.long or data.device.type != "cpu":
        raise ValueError("Each dataset must be a 1D CPU torch.long tensor.")
    if len(data) <= cfg.block_size:
        raise ValueError("Need at least block_size + 1 tokens for inputs and targets.")

    # randint's upper bound is exclusive. Include the LAST valid start, N-T-1.
    starts = torch.randint(
        len(data) - cfg.block_size, (cfg.batch_size,), generator=generator,
    )
    x = torch.stack([data[i:i + cfg.block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in starts.tolist()])
    return x.to(cfg.device), y.to(cfg.device)


def sample_saved_batches(output_dir, cfg):
    datasets = {
        split: torch.tensor(
            json.loads((output_dir / f"{split}_ids.json").read_text(encoding="utf-8")),
            dtype=torch.long,
        )
        for split in ("train", "val")
    }
    generator = torch.Generator().manual_seed(42)
    for split in ("train", "val"):
        x, y = get_batch(split, datasets, cfg, generator)
        assert torch.equal(x[:, 1:], y[:, :-1])
        print(f"{split}: x={list(x.shape)}, y={list(y.shape)}, dtype={x.dtype}")
        print(f"  First input IDs:  {x[0, :8].tolist()}")
        print(f"  First target IDs: {y[0, :8].tolist()}")
        print("  One-token shift: PASS")

# 6. Causal multi-head self-attention
def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Precompute fixed RoPE angles for every position and coordinate pair."""
    if seq_len < 1 or head_dim < 2 or head_dim % 2 or base <= 0:
        raise ValueError("RoPE needs a positive length/base and an even head dimension.")
    pair_ids = torch.arange(head_dim // 2, dtype=torch.float32)
    inv_freq = base ** (-2 * pair_ids / head_dim)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos[None, None], sin[None, None]


def apply_rope(x, cos, sin):
    """Rotate adjacent feature pairs; x has shape [B, heads, T, head_dim]."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE needs an even head dimension.")
    pair = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.n_heads < 1 or cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by a positive n_heads.")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        if cfg.position_encoding not in ("learned", "rope"):
            raise ValueError("position_encoding must be 'learned' or 'rope'.")
        self.position_encoding = cfg.position_encoding
        if cfg.attention_impl not in ("manual", "sdpa", "flash"):
            raise ValueError("attention_impl must be 'manual', 'sdpa', or 'flash'.")
        self.attention_impl = cfg.attention_impl
        self.dropout_p = cfg.dropout
        if self.position_encoding == "rope":
            cos, sin = build_rope_cache(cfg.block_size, self.head_dim, cfg.rope_base)
            # Fixed tables move with model.to(device), but stay out of checkpoints.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B, T, C] -> [B, heads, T, head_dim]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.position_encoding == "rope":
            cos = self.rope_cos[:, :, :T].to(dtype=q.dtype)
            sin = self.rope_sin[:, :, :T].to(dtype=q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.attention_impl == "manual":
            scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            # Position t can see positions 0..t, never future tokens.
            scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            out = weights @ v
        elif self.attention_impl == "flash":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            # Fail loudly if this shape/dtype/GPU cannot run the fused kernel.
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    dropout_p=self.dropout_p if self.training else 0.0,
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.output_dropout(self.proj(out))


# 7. Feed-forward network and pre-norm transformer block
def make_norm(cfg):
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(cfg.d_model)
    if cfg.norm_type == "rmsnorm":
        return nn.RMSNorm(cfg.d_model, eps=1e-6)
    raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'.")


class SwiGLU(nn.Module):
    """Gated MLP, sized to roughly match the parameter count of a 4d GELU MLP."""

    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.mlp_type == "gelu":
            # Applied independently to each token: C -> 4C -> C.
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model),
                nn.GELU(),
                nn.Linear(4 * cfg.d_model, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
        elif cfg.mlp_type == "swiglu":
            self.net = SwiGLU(cfg)
        else:
            raise ValueError("mlp_type must be 'gelu' or 'swiglu'.")

    def forward(self, x):
        return self.net(x)


def sinkhorn_doubly_stochastic(logits, iterations=20):
    """Nonnegative near-doubly-stochastic [*, n, n] mixing in FP32."""
    matrix = torch.exp(logits.float() - logits.float().amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True)
    return matrix


_compiled_sinkhorn = None


def compiled_sinkhorn_doubly_stochastic(logits, iterations):
    """Compile the fixed-shape GPU Sinkhorn loop once per process."""
    global _compiled_sinkhorn
    if _compiled_sinkhorn is None:
        _compiled_sinkhorn = torch.compile(
            sinkhorn_doubly_stochastic, fullgraph=True, dynamic=False,
        )
    return _compiled_sinkhorn(logits, iterations)


class MHCRoute(nn.Module):
    """One mHC residual edge over n parallel [C] streams per token.

    X'[i] = sum_j H_res[i,j] X[j] + H_post[i] F(sum_j H_pre[j] X[j]).
    The coefficients vary by token, and Sinkhorn constrains the skip mixing.
    """

    def __init__(self, cfg, norm, layer):
        super().__init__()
        n, c = cfg.mhc_streams, cfg.d_model
        self.n = n
        self.iterations = cfg.mhc_sinkhorn_iters
        self.block_size = cfg.block_size
        self.routing_norm = nn.RMSNorm(n * c, eps=1e-6)
        self.routing_proj = nn.Linear(n * c, n * n + 2 * n, bias=False)
        self.routing_alpha = nn.Parameter(torch.full((n * n + 2 * n,), 0.01))
        self.pre_bias = nn.Parameter(torch.full((n,), -math.log(n - 1)))
        self.post_bias = nn.Parameter(torch.zeros(n))
        self.res_bias = nn.Parameter(torch.eye(n).mul(4.0))
        self.residual_mix_logit = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.norm = norm
        self.layer = layer
        self.record_stats = False
        self.last_stats = None

    def forward(self, streams):
        b, t, n, c = streams.shape
        flat = streams.reshape(b, t, n * c)
        coefficients = self.routing_proj(self.routing_norm(flat)) * self.routing_alpha
        pre_delta, post_delta, res_delta = coefficients.split((n, n, n * n), dim=-1)
        pre = torch.sigmoid(pre_delta + self.pre_bias)
        post = 2 * torch.sigmoid(post_delta + self.post_bias)
        # Unbounded learned logits made a 20-step Sinkhorn matrix almost
        # column-stochastic but not row-stochastic in a long trial run. Keep
        # logits in a compact range and preserve an identity-heavy skip.
        res_logits = 2 * torch.tanh(
            (res_delta.reshape(b, t, n, n) + self.res_bias) / 2
        )
        sinkhorn = (
            compiled_sinkhorn_doubly_stochastic
            if streams.device.type == "cuda" and t == self.block_size
            else sinkhorn_doubly_stochastic
        )
        stochastic = sinkhorn(
            res_logits,
            self.iterations,
        )
        mix_strength = torch.sigmoid(self.residual_mix_logit)
        identity = torch.eye(n, device=streams.device, dtype=stochastic.dtype)
        mixing = ((1 - mix_strength) * identity + mix_strength * stochastic).to(
            dtype=streams.dtype
        )
        if self.record_stats:
            with torch.no_grad():
                stream_rms = streams.float().square().mean(dim=(0, 1, 3)).sqrt()
                self.last_stats = {
                    "stream_rms": stream_rms.tolist(),
                    "pre_mean": pre.float().mean().item(),
                    "post_mean": post.float().mean().item(),
                    "res_diag_mean": mixing.diagonal(dim1=-2, dim2=-1).float().mean().item(),
                    "mix_strength": mix_strength.item(),
                    "row_error": (mixing.float().sum(-1) - 1).abs().max().item(),
                    "col_error": (mixing.float().sum(-2) - 1).abs().max().item(),
                }
        selected = torch.einsum("...n,...nc->...c", pre, streams)
        updated = self.layer(self.norm(selected))
        skip = torch.einsum("...ij,...jc->...ic", mixing, streams)
        return skip + post[..., :, None] * updated[..., None, :]


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.ffn = FeedForward(cfg)
        self.mhc = cfg.mhc_streams > 1
        if self.mhc:
            self.attn_route = MHCRoute(cfg, self.ln1, self.attn)
            self.ffn_route = MHCRoute(cfg, self.ln2, self.ffn)
            # These modules now belong to the routes; avoid duplicate names.
            del self.ln1, self.attn, self.ln2, self.ffn

    def forward(self, x):
        if self.mhc:
            return self.ffn_route(self.attn_route(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# 8. GPT: connect the pieces into a single forward pass
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.mhc_streams < 1 or (cfg.mhc_streams > 1 and cfg.mhc_sinkhorn_iters < 1):
            raise ValueError("mHC needs positive stream count and Sinkhorn iterations.")
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = (
            nn.Embedding(cfg.block_size, cfg.d_model)
            if cfg.position_encoding == "learned" else None
        )
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # One shared matrix for token lookup and projection back to vocabulary.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 1 <= idx.shape[1] <= self.cfg.block_size:
            raise ValueError("idx must have shape [B, T] with 1 <= T <= block_size.")
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(idx.shape[1], device=idx.device)
            # [B, T, C] + [T, C]; position embeddings broadcast across the batch.
            x = x + self.position_embedding(positions)
        if self.cfg.mhc_streams > 1:
            x = x.unsqueeze(-2).expand(-1, -1, self.cfg.mhc_streams, -1)
        for block in self.blocks:
            x = block(x)
        if self.cfg.mhc_streams > 1:
            x = x.sum(dim=-2)
        logits = self.lm_head(self.final_norm(x))  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Sample one next token at a time, keeping only the latest context window."""
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("Provide a nonempty [B, T] prompt.")
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative; temperature must be positive.")
        was_training = self.training
        self.eval()  # Disable dropout while generating.
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.block_size:])
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx
        finally:
            self.train(was_training)


def forward_saved_batch(output_dir, cfg):
    tokenizer = BPETokenizer.load(output_dir / "tokenizer.json")
    cfg.vocab_size = len(tokenizer.vocab)
    data = torch.tensor(
        json.loads((output_dir / "train_ids.json").read_text(encoding="utf-8")),
        dtype=torch.long,
    )
    x, y = get_batch("train", {"train": data}, cfg, torch.Generator().manual_seed(42))
    torch.manual_seed(42)
    model = GPT(cfg).to(cfg.device)
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"Token embedding table: {list(model.token_embedding.weight.shape)}")
    if model.position_embedding is not None:
        print(f"Position embedding table: {list(model.position_embedding.weight.shape)}")
    else:
        print(f"Position encoding: RoPE (base={cfg.rope_base:g})")
    print(f"Blocks: {cfg.n_layers}; heads: {cfg.n_heads}; head dimension: {cfg.d_model // cfg.n_heads}")
    print(f"Input: {list(x.shape)} -> embeddings: {[ *x.shape, cfg.d_model ]}")
    print(f"Vocabulary logits: {list(logits.shape)}")
    print(f"Untrained next-token loss: {loss.item():.4f}")
    print("Forward pass only; no optimizer update performed.")

# 9. Optimizer
def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)


def init_scaled_residual_projections(model):
    """Chapter 6: make each residual branch's initial output smaller."""
    scale = 0.02 / math.sqrt(2 * model.cfg.n_layers)
    with torch.no_grad():
        for block in model.blocks:
            attn = block.attn_route.layer if block.mhc else block.attn
            ffn = block.ffn_route.layer if block.mhc else block.ffn
            nn.init.normal_(attn.proj.weight, mean=0.0, std=scale)
            ffn_output = (
                ffn.net.down_proj if isinstance(ffn.net, SwiGLU)
                else ffn.net[2]
            )
            nn.init.normal_(ffn_output.weight, mean=0.0, std=scale)


def configure_decay_groups(model, weight_decay=0.1):
    """Chapter 6: decay matrices, but exclude biases and norm parameters."""
    decay, no_decay, seen = [], [], set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen:
                continue  # The token embedding and LM head have a tied weight.
            seen.add(id(param))
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or name.endswith("bias") or param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)
    if seen != {id(param) for param in model.parameters()}:
        raise AssertionError("Every trainable parameter must appear in exactly one group.")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# 10. Learning-rate schedule
def get_lr(step, cfg):
    """step is the zero-based optimizer update index.

    Warm up linearly to peak LR; cosine decay reaches min_lr at the last update.
    """
    if not 0 <= cfg.warmup_steps < cfg.max_steps or cfg.max_steps < 2:
        raise ValueError("Need max_steps >= 2 and 0 <= warmup_steps < max_steps.")
    if step < 0 or not 0 <= cfg.min_lr <= cfg.learning_rate:
        raise ValueError("Invalid step or learning-rate bounds.")
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps - 1:
        return cfg.min_lr
    peak_step = max(0, cfg.warmup_steps - 1)
    progress = (step - peak_step) / (cfg.max_steps - 1 - peak_step)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (
        cfg.learning_rate - cfg.min_lr
    )


# 11. Training loop and logging
def train_saved_tokens(output_dir, cfg, resume=None):
    requested_steps = cfg.max_steps
    checkpoint = None
    start_step = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        # Keep the original schedule horizon: after it ends, get_lr stays at min_lr.
        saved_cfg = TransformerConfig(**checkpoint["config"])
        saved_cfg.device = cfg.device
        saved_cfg.eval_interval = cfg.eval_interval
        saved_cfg.eval_steps = cfg.eval_steps
        cfg = saved_cfg
        start_step = checkpoint["step"]
    if requested_steps < 1:
        raise ValueError("Training steps must be positive.")
    stop_step = start_step + requested_steps
    get_lr(0, cfg)  # Validate the schedule before creating a run.
    if cfg.eval_interval < 1 or cfg.eval_steps < 1:
        raise ValueError("Evaluation interval and evaluation steps must be positive.")
    tokenizer = BPETokenizer.load(output_dir / "tokenizer.json")
    if checkpoint is not None:
        merges = [[a, b, idx] for (a, b), idx in tokenizer.merges.items()]
        if merges != checkpoint["tokenizer_merges"]:
            raise ValueError("Saved dataset tokenizer differs from checkpoint tokenizer.")
    cfg.vocab_size = len(tokenizer.vocab)
    datasets = {
        split: torch.tensor(
            json.loads((output_dir / f"{split}_ids.json").read_text(encoding="utf-8")),
            dtype=torch.long,
        ) for split in ("train", "val")
    }
    torch.manual_seed(42)
    model = GPT(cfg).to(cfg.device)
    optimizer = make_optimizer(model, cfg)
    train_rng = torch.Generator().manual_seed(42)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "train_rng" in checkpoint:
            train_rng.set_state(checkpoint["train_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
        else:
            # Legacy checkpoints did not store RNG states: use a new stream.
            train_rng.manual_seed(42 + start_step)
            torch.manual_seed(42 + start_step)
    run_dir = output_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "run.json").write_text(json.dumps({
        "resume": str(Path(resume).resolve()) if resume else None,
        "start_step": start_step, "stop_step": stop_step,
        "schedule": "original warmup/cosine, then constant min_lr",
    }, indent=2))
    logger = logging.getLogger("my_gpt.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(run_dir / "training.log")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    try:
        logger.info("Parameters=%s | device=%s",
                    f"{sum(p.numel() for p in model.parameters()):,}", cfg.device)
        if checkpoint is not None:
            logger.info("Resuming model AND optimizer at step %d; stop=%d; next LR=%.6f",
                        start_step, stop_step, get_lr(start_step, cfg))
            if "train_rng" not in checkpoint:
                logger.info("Legacy checkpoint has no RNG state; continuing with a new random stream.")
        logger.info("Original schedule: warmup=%d | peak_lr=%.6f | min_lr=%.6f | horizon=%d",
                    cfg.warmup_steps, cfg.learning_rate, cfg.min_lr, cfg.max_steps)
        initial = estimate_loss(model, datasets, cfg)
        best_val = initial["val"]
        save_checkpoint(run_dir / "best.pt", model, optimizer, start_step, tokenizer, train_rng)
        logger.info("Before training | train=%.4f val=%.4f", initial["train"], initial["val"])
        start = time.perf_counter()
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
            metrics.write(json.dumps({"step": start_step, **initial}) + "\n")
            model.train()
            for step in range(start_step, stop_step):
                lr = get_lr(step, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, y = get_batch("train", datasets, cfg, train_rng)
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at update {step + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip, error_if_nonfinite=True,
                )
                optimizer.step()
                row = {
                    "step": step + 1, "lr": lr, "batch_loss": loss.item(),
                    "grad_norm_before_clip": grad_norm.item(),
                    "elapsed_seconds": time.perf_counter() - start,
                }
                if (step + 1) % cfg.eval_interval == 0 or step == stop_step - 1:
                    row.update(estimate_loss(model, datasets, cfg))
                    save_checkpoint(run_dir / "latest.pt", model, optimizer, step + 1, tokenizer, train_rng)
                    if row["val"] < best_val:
                        best_val = row["val"]
                        save_checkpoint(run_dir / "best.pt", model, optimizer, step + 1, tokenizer, train_rng)
                    logger.info("Step %d/%d | lr=%.6f | batch=%.4f train=%.4f val=%.4f | grad=%.3f | %.1fs",
                                step + 1, stop_step, lr, row["batch_loss"],
                                row["train"], row["val"], row["grad_norm_before_clip"],
                                row["elapsed_seconds"])
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
        save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, stop_step, tokenizer, train_rng)
        sample = sample_text(model, tokenizer, cfg)
        (run_dir / "sample.txt").write_text(sample, encoding="utf-8")
        logger.info("Sample after %d updates: %r", stop_step, sample)
        logger.info("Run saved: %s", run_dir.resolve())
        return run_dir
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


# 12. Validation evaluation
@torch.no_grad()
def estimate_loss(model, datasets, cfg):
    if cfg.eval_steps < 1:
        raise ValueError("eval_steps must be positive.")
    was_training = model.training
    model.eval()
    try:
        result = {}
        for seed, split in enumerate(("train", "val"), start=100):
            # Same evaluation windows every time; independent of training sampling.
            rng = torch.Generator().manual_seed(seed)
            losses = []
            for _ in range(cfg.eval_steps):
                x, y = get_batch(split, datasets, cfg, rng)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[split] = sum(losses) / len(losses)
        return result
    finally:
        model.train(was_training)


# 13. Sample generation
def sample_text(model, tokenizer, cfg, prompt="Harry", max_new_tokens=64):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=cfg.device)
    generated = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8)
    return tokenizer.decode(generated[0].tolist())


# 14. Atomic checkpoints, including CPU RNG states for reproducible continuation
def save_checkpoint(path, model, optimizer, step, tokenizer, train_rng=None):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "config": asdict(model.cfg),
        "tokenizer_merges": [[a, b, idx] for (a, b), idx in tokenizer.merges.items()],
    }
    if train_rng is not None:
        payload.update(train_rng=train_rng.get_state(), torch_rng=torch.get_rng_state())
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Chapter 5: our step-by-step GPT")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--train", action="store_true", help="Train on the saved BPE token IDs.",
    )
    modes.add_argument(
        "--sample-batch", action="store_true",
        help="Load saved token IDs and sample train/val batches without retraining BPE.",
    )
    modes.add_argument(
        "--forward-batch", action="store_true",
        help="Run an untrained GPT forward pass on a saved training batch.",
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path.home() / "Desktop/NanoGPT/data/harry_potter.txt",
    )
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path, help="Restore checkpoint; --steps is ADDITIONAL updates.")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for this small model.")
    parser.add_argument("--position-encoding", choices=("learned", "rope"), default="learned")
    parser.add_argument("--norm-type", choices=("layernorm", "rmsnorm"), default="layernorm")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "outputs/my_gpt",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = TransformerConfig(
        vocab_size=args.vocab_size, max_steps=args.steps, warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval, eval_steps=args.eval_steps,
        position_encoding=args.position_encoding, norm_type=args.norm_type,
    )
    if args.train:
        train_saved_tokens(args.output_dir, cfg, resume=args.resume)
        return
    if args.sample_batch:
        sample_saved_batches(args.output_dir, cfg)
        return
    if args.forward_batch:
        forward_saved_batch(args.output_dir, cfg)
        return
    print(f"Loading text: {args.data.resolve()}", flush=True)
    train_text, val_text = load_text_splits(args.data)
    print(f"Characters: train={len(train_text):,}, val={len(val_text):,}", flush=True)
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, cfg.vocab_size)
    cfg.vocab_size = len(tokenizer.vocab)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    assert tokenizer.decode(train_ids) == train_text
    assert tokenizer.decode(val_ids) == val_text
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "tokenizer.json"
    tokenizer.save(path)
    # Keep the encoded datasets ready for the batch sampler.
    for name, token_ids in (("train", train_ids), ("val", val_ids)):
        token_path = args.output_dir / f"{name}_ids.json"
        token_path.write_text(json.dumps(token_ids), encoding="utf-8")
        assert json.loads(token_path.read_text(encoding="utf-8")) == token_ids
        print(f"Saved {name} token IDs: {token_path}", flush=True)
    restored = BPETokenizer.load(path)
    example = "Harry opened the door."
    ids = restored.encode(example)
    assert ids == tokenizer.encode(example)
    assert restored.decode(ids) == example
    print(f"Tokens: train={len(train_ids):,}, val={len(val_ids):,}")
    print(f"Vocabulary: {cfg.vocab_size}; full train/val round-trip: PASS")
    print(f"Example: {example!r}")
    print(f"Token IDs: {ids}")
    print(f"Token bytes: {[restored.vocab[i] for i in ids]}")
    print(f"Next-token input:  {ids[:-1]}")
    print(f"Next-token target: {ids[1:]}")
    print(f"Tokenizer saved: {path}")
    print("Tokenization complete. Use --sample-batch or --forward-batch for the next stages.")


if __name__ == "__main__":
    main()
