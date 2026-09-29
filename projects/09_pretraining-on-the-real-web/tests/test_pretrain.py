"""Chapter 9 pipeline: shards, memmap loader, bpb, grad accumulation, resume."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import pretrain  # noqa: E402
import pretrain_data as pdata  # noqa: E402
from prepare_fineweb import prepare  # noqa: E402

DOCS = [f"Document {i}: the owl carried letter number {i} across the lake at dawn."
        for i in range(400)]


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    prepare(DOCS, out, max_tokens=10**9, shard_tokens=3000, val_tokens=1000, batch_docs=64)
    return out


def test_shards_roundtrip_and_eot(shards):
    manifest = json.loads((shards / "manifest.json").read_text())
    assert manifest["shards"][0]["split"] == "val"
    assert manifest["shards"][0]["tokens"] == 1000
    assert all(s["tokens"] == 3000 for s in manifest["shards"][1:-1])
    tokens = np.concatenate([np.fromfile(shards / s["file"], dtype=np.uint16)
                             for s in manifest["shards"]])
    assert len(tokens) == manifest["total_tokens"]
    assert (tokens == pdata.EOT).sum() == len(DOCS)  # one boundary per document
    enc = pdata.get_encoder()
    first = tokens[:np.argmax(tokens == pdata.EOT)].tolist()
    assert enc.decode(first) == DOCS[0]


def test_loader_windows_are_shifted_and_deterministic(shards):
    a = pdata.ShardLoader(shards, "train", seq_len=32, seed=7)
    b = pdata.ShardLoader(shards, "train", seq_len=32, seed=7)
    x, y = a.batch(4)
    x2, y2 = b.batch(4)
    assert x.shape == y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])  # targets are inputs shifted by one
    assert torch.equal(x, x2) and torch.equal(y, y2)
    state = a.state_dict()
    after = a.batch(4)[0]
    a.load_state_dict(state)
    assert torch.equal(a.batch(4)[0], after)
    fixed = [w[0] for w in pdata.ShardLoader(shards, "val", 32).fixed_windows(96, 2)]
    assert sum(len(w) for w in fixed) == 3


def test_token_bytes_and_bpb():
    enc = pdata.get_encoder()
    lengths = pdata.token_byte_lengths(enc)
    assert lengths.shape == (pdata.PADDED_VOCAB_SIZE,)
    assert lengths[pdata.EOT] == 0 and lengths[pdata.VOCAB_SIZE:].sum() == 0
    text = "Hello, wonderful world!"
    assert lengths[enc.encode_ordinary(text)].sum() == len(text.encode())
    assert pdata.bits_per_byte(8 * math.log(2), 8) == pytest.approx(1.0)


def test_grad_accumulation_equals_big_batch():
    torch.manual_seed(0)
    model = pretrain.m.GPT(pretrain.m.TransformerConfig(
        vocab_size=50, block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0))
    x = torch.randint(0, 50, (8, 8))
    y = torch.randint(0, 50, (8, 8))
    model(x, y)[1].backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for xs, ys in zip(x.chunk(4), y.chunk(4)):
        (model(xs, ys)[1] / 4).backward()
    for g, p in zip(big, model.parameters()):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-5)


def _args(shards, out, **kw):
    argv = ["--data-dir", str(shards), "--out-dir", str(out), "--preset", "tiny",
            "--device", "cpu", "--total-tokens", str(8 * 1024), "--batch-tokens", "1024",
            "--micro-batch", "8", "--warmup-steps", "2", "--eval-interval", "4",
            "--eval-tokens", "256", "--log-interval", "2"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return pretrain.parse_args(argv)


def test_resume_matches_uninterrupted_run(shards, tmp_path):
    full = pretrain.train(_args(shards, tmp_path / "full"))
    assert full["done"] and full["step"] == 8
    first = pretrain.train(_args(shards, tmp_path / "split", max_seconds=1e-9))
    assert not first["done"] and first["step"] == 1
    second = pretrain.train(_args(shards, tmp_path / "split"))
    assert second["done"] and second["step"] == 8
    a = torch.load(tmp_path / "full" / "latest.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "latest.pt", weights_only=False)["model"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], msg=name)
    assert second["val_bpb"] == pytest.approx(full["val_bpb"])
    evals = [json.loads(line) for line in (tmp_path / "full" / "metrics.jsonl").open()
             if "val_bpb" in line]
    assert [e["step"] for e in evals] == [0, 4, 8]


def test_train_limit_repeats_one_slice(shards):
    loader = pdata.ShardLoader(shards, "train", seq_len=16, seed=0, limit_tokens=200)
    assert len(loader.arrays) == 1 and len(loader.arrays[0]) == 200
    first = np.fromfile(shards / "train_0001.bin", dtype=np.uint16)[:200]
    x, _ = loader.batch(32)
    allowed = {tuple(first[i:i + 16]) for i in range(len(first) - 16)}
    assert all(tuple(row.tolist()) in allowed for row in x)
