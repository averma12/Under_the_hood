"""RMSNorm wiring in the Harry Potter BPE GPT."""

import importlib.util
import sys
from pathlib import Path

import torch


MODEL_PATH = Path(__file__).resolve().parents[1] / "my_gpt.py"
spec = importlib.util.spec_from_file_location("my_gpt_norm_test", MODEL_PATH)
assert spec is not None and spec.loader is not None
my_gpt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = my_gpt
spec.loader.exec_module(my_gpt)


def _config(norm_type: str):
    return my_gpt.TransformerConfig(
        vocab_size=32, block_size=8, d_model=16, n_heads=4,
        n_layers=2, dropout=0.0, mlp_type="swiglu", norm_type=norm_type,
    )


def test_rmsnorm_normalizes_without_mean_centering():
    norm = my_gpt.make_norm(_config("rmsnorm"))
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0] * 4]])
    y = norm(x)
    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(y.square().mean(dim=-1), torch.ones(1, 1), atol=1e-6, rtol=0)
    assert y.mean() > 0  # Unlike LayerNorm, RMSNorm does not subtract the mean.


def test_rmsnorm_model_forward_and_optimizer_groups():
    model = my_gpt.GPT(_config("rmsnorm"))
    assert isinstance(model.blocks[0].ln1, torch.nn.RMSNorm)
    assert isinstance(model.blocks[0].ln2, torch.nn.RMSNorm)
    assert isinstance(model.final_norm, torch.nn.RMSNorm)
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].ln1.weight.grad is not None
    decay, no_decay = my_gpt.configure_decay_groups(model)
    norm_weights = {id(module.weight) for module in model.modules() if isinstance(module, torch.nn.RMSNorm)}
    assert norm_weights <= {id(p) for p in no_decay["params"]}
    assert not norm_weights & {id(p) for p in decay["params"]}


def test_old_config_defaults_to_layernorm_and_shared_weights_match():
    old_config = vars(_config("layernorm")).copy()
    del old_config["norm_type"]
    assert my_gpt.TransformerConfig(**old_config).norm_type == "layernorm"
    torch.manual_seed(42)
    ln_model = my_gpt.GPT(_config("layernorm"))
    torch.manual_seed(42)
    rms_model = my_gpt.GPT(_config("rmsnorm"))
    for name, weight in ln_model.state_dict().items():
        if "ln1." not in name and "ln2." not in name and "final_norm." not in name:
            torch.testing.assert_close(weight, rms_model.state_dict()[name], rtol=0, atol=0)
