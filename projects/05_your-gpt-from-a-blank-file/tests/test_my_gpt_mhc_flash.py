"""mHC mixing and fused-SDPA wiring on a small CPU fixture."""

import importlib.util
import sys
from pathlib import Path

import torch


MODEL_PATH = Path(__file__).resolve().parents[1] / "my_gpt.py"
spec = importlib.util.spec_from_file_location("my_gpt_mhc_test", MODEL_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def config(**kwargs):
    values = dict(vocab_size=32, block_size=8, batch_size=2, d_model=16,
                  n_heads=4, n_layers=2, dropout=0, mlp_type="swiglu",
                  norm_type="rmsnorm", mhc_streams=4)
    values.update(kwargs)
    return module.TransformerConfig(**values)


def test_sinkhorn_is_nearly_doubly_stochastic():
    raw = torch.randn(2, 3, 4, 4)
    matrix = module.sinkhorn_doubly_stochastic(raw)
    assert matrix.min() >= 0
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 3, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2, 3, 4), atol=1e-6, rtol=0)


def test_mhc_forward_backward_and_optimizer_groups():
    model = module.GPT(config())
    module.init_scaled_residual_projections(model)
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x.roll(1, dims=-1))
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    loss.backward()
    route = model.blocks[0].attn_route
    assert route.routing_proj.weight.grad is not None
    assert torch.isfinite(route.res_bias.grad).all()
    groups = module.configure_decay_groups(model)
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))
    assert any(param is route.routing_alpha for param in groups[1]["params"])


def test_route_matches_explicit_stream_equation():
    route = module.MHCRoute(config(), torch.nn.Identity(), torch.nn.Identity())
    with torch.no_grad():
        route.routing_proj.weight.zero_()
        route.pre_bias.copy_(torch.tensor([-1.0, -0.5, 0.0, 0.5]))
        route.post_bias.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0]))
    streams = torch.randn(1, 2, 4, 16)
    actual = route(streams)
    pre = torch.sigmoid(route.pre_bias)
    post = 2 * torch.sigmoid(route.post_bias)
    base = module.sinkhorn_doubly_stochastic(
        2 * torch.tanh(route.res_bias / 2), route.iterations,
    )
    strength = torch.sigmoid(route.residual_mix_logit)
    mixing = (1 - strength) * torch.eye(4) + strength * base
    expected = torch.zeros_like(streams)
    for token in range(2):
        selected = sum(pre[j] * streams[0, token, j] for j in range(4))
        for i in range(4):
            expected[0, token, i] = sum(
                mixing[i, j] * streams[0, token, j] for j in range(4)
            ) + post[i] * selected
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sdpa_matches_manual_attention_without_dropout():
    torch.manual_seed(11)
    manual = module.MultiHeadAttention(config(attention_impl="manual"))
    fused_api = module.MultiHeadAttention(config(attention_impl="sdpa"))
    fused_api.load_state_dict(manual.state_dict())
    manual.eval()
    fused_api.eval()
    x = torch.randn(2, 8, 16)
    torch.testing.assert_close(manual(x), fused_api(x), atol=2e-6, rtol=2e-5)


def test_mhc_routes_cannot_see_future_tokens():
    model = module.GPT(config(attention_impl="sdpa")).eval()
    original = torch.randint(0, 32, (1, 8))
    changed = original.clone()
    changed[:, 5:] = torch.randint(0, 32, (1, 3))
    with torch.no_grad():
        a, _ = model(original)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=2e-6, rtol=2e-5)


def test_routing_stays_doubly_stochastic_with_extreme_learned_logits():
    route = module.MHCRoute(config(), torch.nn.Identity(), torch.nn.Identity())
    route.record_stats = True
    with torch.no_grad():
        route.routing_proj.weight.normal_(0, 4)
        route.routing_alpha.fill_(10)
        route.res_bias.normal_(0, 20)
        route(torch.randn(2, 8, 4, 16))
    assert route.last_stats["row_error"] < 1e-3
    assert route.last_stats["col_error"] < 1e-5
