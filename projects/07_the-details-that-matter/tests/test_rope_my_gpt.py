"""Correctness checks for the RoPE option in our Harry Potter GPT."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[2] / "05_your-gpt-from-a-blank-file" / "my_gpt.py"
spec = importlib.util.spec_from_file_location("chapter5_my_gpt_rope_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestRoPE(unittest.TestCase):
    def test_rotation_preserves_norm_and_position_zero(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 11, 8)
        cos, sin = module.build_rope_cache(11, 8)
        rotated = module.apply_rope(x, cos, sin)
        torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
        torch.testing.assert_close(
            rotated.float().norm(dim=-1), x.float().norm(dim=-1), atol=1e-6, rtol=1e-6
        )

    def test_attention_score_depends_on_relative_offset(self):
        torch.manual_seed(2)
        q, k = torch.randn(8), torch.randn(8)
        cos, sin = module.build_rope_cache(20, 8)

        def score(q_position, k_position):
            q_rot = module.apply_rope(q[None, None, None], cos[:, :, q_position:q_position + 1], sin[:, :, q_position:q_position + 1])
            k_rot = module.apply_rope(k[None, None, None], cos[:, :, k_position:k_position + 1], sin[:, :, k_position:k_position + 1])
            return (q_rot * k_rot).sum()

        torch.testing.assert_close(score(2, 7), score(12, 17), atol=1e-6, rtol=1e-6)

    def test_forward_backward_and_causal_mask(self):
        torch.manual_seed(3)
        cfg = module.TransformerConfig(
            vocab_size=64, block_size=16, d_model=32, n_heads=4,
            n_layers=2, dropout=0.0, mlp_type="swiglu", position_encoding="rope",
        )
        model = module.GPT(cfg)
        self.assertIsNone(model.position_embedding)
        self.assertFalse(any("rope_cos" in key or "rope_sin" in key for key in model.state_dict()))
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (2, 8, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            altered = idx.clone()
            altered[:, 5:] = (altered[:, 5:] + 1) % cfg.vocab_size
            altered_logits, _ = model(altered)
        torch.testing.assert_close(logits[:, :5], altered_logits[:, :5], atol=1e-6, rtol=1e-6)
        long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 4))
        generated = model.generate(long_prompt, 3)
        self.assertEqual(generated.shape, (1, cfg.block_size + 7))

    def test_existing_learned_position_config_still_loads(self):
        cfg = module.TransformerConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
        original = module.GPT(cfg)
        saved = original.state_dict()
        restored = module.GPT(module.TransformerConfig(**{
            "vocab_size": 32, "block_size": 8, "d_model": 16, "n_heads": 2, "n_layers": 1,
        }))
        restored.load_state_dict(saved)
        self.assertIsNotNone(restored.position_embedding)


if __name__ == "__main__":
    unittest.main()
