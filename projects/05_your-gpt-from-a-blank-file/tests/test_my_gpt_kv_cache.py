"""KV-cache decoding and split-KV (Flash-Decoding) attention match full recompute."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MODEL_PATH = Path(__file__).resolve().parents[1] / "my_gpt.py"
spec = importlib.util.spec_from_file_location("my_gpt_kv_cache_test", MODEL_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def config(**kwargs):
    values = dict(vocab_size=40, block_size=12, batch_size=2, d_model=32,
                  n_heads=4, n_layers=2, dropout=0.1)
    values.update(kwargs)
    return module.TransformerConfig(**values)


VARIANTS = [
    dict(),
    dict(position_encoding="rope"),
    dict(attention_impl="sdpa", mlp_type="swiglu", norm_type="rmsnorm"),
    dict(mhc_streams=4, mlp_type="swiglu", norm_type="rmsnorm", mhc_sinkhorn_iters=8),
]


@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("variant", VARIANTS, ids=["learned", "rope", "sdpa", "mhc"])
def test_incremental_logits_match_full_forward(variant, splits):
    torch.manual_seed(0)
    model = module.GPT(config(**variant)).eval()
    idx = torch.randint(0, 40, (2, 12))
    with torch.no_grad():
        full, _ = model(idx)
        cache = module.KVCache(model.cfg, decode_splits=splits)
        steps = [model(idx[:, :5], kv_cache=cache)[0]]  # prefill a 5-token prompt
        steps += [model(idx[:, t:t + 1], kv_cache=cache)[0] for t in range(5, 12)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache.length == 12


@pytest.mark.parametrize("L,splits", [(1, 1), (7, 3), (16, 4), (5, 8), (33, 5)])
def test_split_kv_matches_softmax_attention(L, splits):
    torch.manual_seed(L)
    q = torch.randn(2, 3, 1, 8)
    k, v = torch.randn(2, 3, L, 8), torch.randn(2, 3, L, 8)
    k[:, :, -1] = q[:, :, 0] * 4  # largest score in the last chunk forces a rescale
    expected = F.scaled_dot_product_attention(q, k, v)
    got = module.split_kv_decode_attention(q, k, v, splits)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("prompt_len", [1, 4, 12, 20])
def test_generate_with_cache_matches_recompute(prompt_len):
    """Same seed, same samples, including after the window starts sliding."""
    torch.manual_seed(1)
    model = module.GPT(config(position_encoding="rope"))
    prompt = torch.randint(0, 40, (2, prompt_len))
    outputs = []
    for kwargs in (dict(use_kv_cache=False), dict(), dict(decode_splits=2)):
        torch.manual_seed(123)
        outputs.append(model.generate(prompt, 15, temperature=0.8, **kwargs))
    assert outputs[0].shape == (2, prompt_len + 15)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
    assert model.training  # generate restores the previous mode


def test_cache_rejects_misuse():
    model = module.GPT(config()).eval()
    cache = module.KVCache(model.cfg)
    with torch.no_grad():
        model(torch.zeros(1, 3, dtype=torch.long), kv_cache=cache)
        with pytest.raises(ValueError, match="one token at a time"):
            model(torch.zeros(1, 2, dtype=torch.long), kv_cache=cache)
        for _ in range(9):
            model(torch.zeros(1, 1, dtype=torch.long), kv_cache=cache)
        with pytest.raises(ValueError, match="full"):
            model(torch.zeros(1, 1, dtype=torch.long), kv_cache=cache)
    with pytest.raises(ValueError):
        module.KVCache(model.cfg, decode_splits=0)
