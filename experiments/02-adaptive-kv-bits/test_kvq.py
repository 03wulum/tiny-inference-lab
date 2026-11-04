"""Correctness checks for the KV quantization hook (~10 s, CPU, tiny random model, no downloads).
Skipped when torch/transformers are not installed."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import kvq  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    m = LlamaForCausalLM(cfg).eval()
    kvq.install(m)
    ids = torch.randint(0, 100, (3, 60))
    mask = torch.ones_like(ids)
    mask[2, 40:] = 0
    with torch.no_grad():
        kvq.KVPlan.mode, kvq.KVPlan.capture_attn = "fp", True
        fp = m(input_ids=ids, attention_mask=mask).logits
        kvq.KVPlan.capture_attn = False
    return m, ids, mask, fp, kvq.KVPlan.captured


def run(m, ids, mask, bits):
    kvq.KVPlan.mode, kvq.KVPlan.bits = "bits", bits
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask).logits
    kvq.KVPlan.mode = "fp"
    return out


def test_16_bits_is_exact(setup):
    m, ids, mask, fp, _ = setup
    assert torch.equal(run(m, ids, mask, kvq.static_bits(mask, 16)), fp)


def test_error_grows_as_bits_shrink(setup):
    m, ids, mask, fp, _ = setup
    errs = [float((run(m, ids, mask, kvq.static_bits(mask, b)) - fp).abs().mean()) for b in (8, 4, 2)]
    assert errs[0] < errs[1] < errs[2]


def test_fake_quant_levels():
    x = torch.randn(1, 1, 5, 32)
    q = kvq.fake_quant(x, torch.full((1, 5), 2))
    assert all(len(torch.unique(q[0, 0, t])) <= 4 for t in range(5))


def test_captured_attention_is_causal_and_normalised(setup):
    *_, attn = setup
    assert torch.allclose(attn[0].sum(-1), torch.ones_like(attn[0].sum(-1)), atol=1e-4)
    assert float(attn[0, :, 0, 1:].abs().max()) == 0.0  # token 0 cannot see the future


def test_adaptive_respects_budget_and_keeps_sink(setup):
    m, ids, mask, fp, attn = setup
    f = kvq.token_features(fp, ids, attn, mask)
    bits = kvq.adaptive_bits(f, mask, 3.8)
    for i in range(3):
        n = int(mask[i].sum())
        assert float(bits[i, :n].float().mean()) <= 3.8 + 1e-6
        assert int(bits[i, 0]) == 16
        assert set(bits[i, :n].tolist()) <= {2, 4, 8, 16}


def test_random_policy_same_budget(setup):
    *_, mask, fp, attn = setup
    r = kvq.random_bits(mask, 3.8)
    assert abs(kvq.mean_bits(r, mask) - kvq.mean_bits(kvq.random_bits(mask, 3.8, seed=1), mask)) < 1e-6
