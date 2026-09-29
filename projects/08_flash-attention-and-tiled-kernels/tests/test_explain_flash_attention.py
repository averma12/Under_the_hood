"""Tests for explain_flash_attention.py: every numeric claim in the explainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load("project_08_explain", "explain_flash_attention.py")


def _qkv(T: int = 96, d: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(T, d, generator=g, dtype=torch.float64) for _ in range(3))


@pytest.mark.parametrize("block", [1, 3, 7, 32, 1000])
def test_online_softmax_matches_torch(block: int) -> None:
    x = torch.randn(50, dtype=torch.float64) * 10
    assert torch.allclose(fx.softmax_online(x, block), torch.softmax(x, dim=0))
    assert torch.allclose(fx.softmax_three_pass(x), torch.softmax(x, dim=0))


@pytest.mark.parametrize("block", [1, 16, 33, 96])
def test_streaming_row_matches_naive(block: int) -> None:
    Q, K, V = _qkv()
    ref = fx.naive_attention(Q, K, V)
    for i in (0, 17, 95):
        out = fx.attention_row_streaming(Q[i], K, V, block=block)
        assert torch.allclose(out, ref[i], atol=1e-12)


def test_missing_rescale_is_wrong() -> None:
    Q, K, V = _qkv()
    K[80] = Q[0] * 3  # the max score arrives late, forcing a rescale
    ref = fx.naive_attention(Q[:1], K, V)[0]
    bad = fx.attention_row_streaming(Q[0], K, V, block=16, rescale=False)
    assert (bad - ref).abs().max() > 1e-2


@pytest.mark.parametrize("splits", [1, 2, 5, 12])
def test_split_kv_any_order_matches(splits: int) -> None:
    Q, K, V = _qkv()
    ref = fx.naive_attention(Q[:1], K, V)[0]
    for reverse in (False, True):
        out = fx.split_kv_attention(Q[0], K, V, splits, reverse=reverse)
        assert torch.allclose(out, ref, atol=1e-12)


def test_merge_is_associative() -> None:
    Q, K, V = _qkv()
    a, b, c = (fx.partial_state(Q[0], k, v) for k, v in zip(K.chunk(3), V.chunk(3)))
    left = fx.merge_states(fx.merge_states(a, b), c)
    right = fx.merge_states(a, fx.merge_states(b, c))
    for x, y in zip(left, right):
        assert torch.allclose(x, y)


def test_backward_from_lse_matches_autograd() -> None:
    Q, K, V = (t.clone().requires_grad_(True) for t in _qkv(T=40, d=16))
    dO = torch.randn(40, 16, dtype=torch.float64)
    fx.naive_attention(Q, K, V).backward(dO)
    dQ, dK, dV = fx.flash_backward_from_lse(Q.detach(), K.detach(), V.detach(), dO)
    assert torch.allclose(dQ, Q.grad)
    assert torch.allclose(dK, K.grad)
    assert torch.allclose(dV, V.grad)


def test_traffic_model_orders() -> None:
    N, d = 4096, 64
    naive = fx.hbm_bytes_naive(N, d)
    hi = fx.hbm_bytes_flash(N, d, block_q=128)
    lo = fx.hbm_bytes_flash(N, d, kv_from_l2=True)
    assert lo < hi < naive
    flops = fx.attention_flops(N, d)
    ridge = fx.A100["tensor_fp16_flops"] / fx.A100["hbm_bytes_per_s"]
    assert flops / naive < ridge  # naive is memory-bound
    assert flops / lo > ridge  # ideal flash is compute-bound
    assert flops / naive == pytest.approx(d / 2, rel=0.05)
    assert flops / lo == pytest.approx(N / 2, rel=0.01)
