"""
Per-token mixed-precision KV-cache quantization, simulated inside a Hugging Face model.

Paper under test: "Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight
On-Device LLMs", arXiv:2604.04722 (2026).

How the simulation works
------------------------
We register a custom attention function with transformers. It receives the post-RoPE keys
and values (exactly what a KV cache stores), fake-quantizes every token's K and V to that
token's bit-width, then runs ordinary SDPA attention. Bits are chosen per token by a
*policy* before the forward pass. This is the standard "fake quant" protocol used by
KIVI/KVQuant-style accuracy evaluations; it measures accuracy, not latency.

Quantizer: per-token, per-head asymmetric min/max integer quantization with an fp16 scale
and zero point per group of `group` channels (KIVI-style). 16 bits = left untouched.
"""
from __future__ import annotations

import math

import numpy as np
import torch

BIT_OPTIONS = (2, 4, 8, 16)


# --------------------------------------------------------------------------------------
# Quantizers
# --------------------------------------------------------------------------------------
def fake_quant(x: torch.Tensor, bits: torch.Tensor, group: int = 32) -> torch.Tensor:
    """x: [batch, heads, seq, dim]; bits: [batch, seq] ints in BIT_OPTIONS.
    Returns x quantized/dequantized per (batch, head, token, channel-group)."""
    b, h, s, d = x.shape
    g = min(group, d)
    xf = x.float().reshape(b, h, s, d // g, g)
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    levels = (2.0 ** bits.float() - 1).view(b, 1, s, 1, 1)
    scale = ((hi - lo) / levels).clamp_min(1e-8).half().float()  # stored as fp16
    lo = lo.half().float()
    q = torch.clamp(torch.round((xf - lo) / scale), torch.zeros_like(levels), levels)
    out = (q * scale + lo).reshape(b, h, s, d)
    keep = (bits >= 16).view(b, 1, s, 1)
    return torch.where(keep, x.float(), out).to(x.dtype)


class TurboQuantTorch:
    """TurboQuant-mse from experiment 01, ported to torch: random rotation + Lloyd-Max
    codebook for the coordinate distribution of a random unit vector."""

    def __init__(self, d: int, bits: int, seed: int = 0, device="cpu"):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "01-turboquant-kv-cache"))
        from turboquant import lloyd_max_codebook, random_rotation  # noqa: E402
        rng = np.random.default_rng((0x7B0, seed))
        self.rot = torch.tensor(random_rotation(d, rng), dtype=torch.float32, device=device)
        c, bnd = lloyd_max_codebook(d, bits)
        self.c = torch.tensor(c, dtype=torch.float32, device=device)
        self.bnd = torch.tensor(bnd, dtype=torch.float32, device=device)
        self.bits = bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).half().float()
        y = (xf / n.clamp_min(1e-8)) @ self.rot.T
        y = self.c[torch.bucketize(y.contiguous(), self.bnd)]
        return ((y @ self.rot) * n).to(x.dtype)


# --------------------------------------------------------------------------------------
# Attention hook: quantize K/V with whatever the current "plan" says
# --------------------------------------------------------------------------------------
class KVPlan:
    """Global, set before each forward pass."""
    mode = "fp"             # "fp" | "bits" | "turbo"
    bits = None             # [batch, seq] tensor for mode == "bits"
    turbo = None            # TurboQuantTorch for mode == "turbo"
    capture_attn = False    # record last-layer attention probabilities (for features)
    captured = None
    n_layers = 0


def kvq_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if KVPlan.mode == "bits":
        key = fake_quant(key, KVPlan.bits)
        value = fake_quant(value, KVPlan.bits)
    elif KVPlan.mode == "turbo":
        key, value = KVPlan.turbo(key), KVPlan.turbo(value)
    if KVPlan.capture_attn and module.layer_idx == KVPlan.n_layers - 1:
        rep = query.shape[1] // key.shape[1]
        k = key.repeat_interleave(rep, dim=1)
        logits = (query.float() @ k.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attention_mask[..., : k.shape[-2]], float("-inf"))
        elif attention_mask is not None:
            logits = logits + attention_mask[..., : k.shape[-2]].float()
        else:  # transformers may pass no mask and rely on is_causal
            q_len, k_len = logits.shape[-2:]
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=logits.device).tril(k_len - q_len)
            logits = logits.masked_fill(~causal, float("-inf"))
        KVPlan.captured = logits.softmax(-1)  # [batch, heads, q, k]
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def install(model):
    from transformers import AttentionInterface
    AttentionInterface.register("kvq", kvq_attention)
    model.config._attn_implementation = "kvq"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "kvq"
    KVPlan.n_layers = model.config.num_hidden_layers


# --------------------------------------------------------------------------------------
# Per-token features (the paper's s_t = [H_t, R_t, V_t, C_t]) and bit policies
# --------------------------------------------------------------------------------------
def token_features(logits, input_ids, attn, mask):
    """logits [b,s,V] from an fp pass, attn [b,h,s,s] last-layer probs, mask [b,s].
    H: entropy of the next-token distribution at t (uncertainty)
    R: rarity, -log p(x_t | x_<t), smoothed with a 3-token moving average
    V: variance across heads of the attention mass token t receives (paper: attention variance)
    C: confidence, max next-token probability at t
    Returns [b, s, 4] float tensor."""
    lp = logits.float().log_softmax(-1)
    p = lp.exp()
    H = -(p * lp).sum(-1)
    C = p.max(-1).values
    R = torch.zeros_like(H)
    R[:, 1:] = -lp[:, :-1].gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    R[:, 0] = R[:, 1:].mean(-1)
    R = torch.nn.functional.avg_pool1d(R[:, None], 3, 1, 1, count_include_pad=False)[:, 0]
    received = attn.sum(-2)                 # [b,h,s] attention each key token receives
    V = received.var(1)                     # variance across heads
    f = torch.stack([H, R, V, C], -1)
    return f * mask[..., None]


FRAC_8BIT = 0.10  # share of tokens that get 8 bits, if the budget allows


def adaptive_bits(features, mask, target_bits=3.8):
    """Rank tokens by an importance score built from the paper's features and give the most
    important ones more bits, at a fixed average budget. High uncertainty (H), rarity (R)
    and attention variance (V) and low confidence (C) => more bits. Features are z-scored
    within each sequence. The first token (attention sink) always keeps 16 bits."""
    z = (features - features.mean(1, keepdim=True)) / (features.std(1, keepdim=True) + 1e-6)
    score = z[..., 0] + z[..., 1] + z[..., 2] - z[..., 3]
    return _assign_by_rank(score, mask, target_bits)


def random_bits(mask, target_bits=3.8, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    score = torch.rand(mask.shape, generator=g).to(mask.device)
    return _assign_by_rank(score, mask, target_bits)


def _assign_by_rank(score, mask, target_bits):
    """Every token starts at 2 bits, the first token (attention sink) at 16. The remaining
    budget (target_bits x tokens, counted exactly, sink included) upgrades the highest-scoring
    tokens: first up to FRAC_8BIT of them to 8 bits, then as many as fit to 4 bits.
    So mean_bits(...) <= target_bits for every sequence."""
    bits = torch.full(score.shape, 2, dtype=torch.int64, device=score.device)
    for i in range(score.shape[0]):
        idx = torch.nonzero(mask[i]).squeeze(-1)
        n = len(idx)
        bits[i, idx[0]] = 16
        budget = target_bits * n - 16 - 2 * (n - 1)
        rest = idx[1:][torch.argsort(score[i, idx[1:]], descending=True)]
        n8 = int(min(round(FRAC_8BIT * n), max(budget, 0) // 6))
        bits[i, rest[:n8]] = 8
        budget -= 6 * n8
        n4 = int(min(len(rest) - n8, max(budget, 0) // 2))
        bits[i, rest[n8:n8 + n4]] = 4
    return bits


def static_bits(mask, b):
    return torch.full(mask.shape, b, dtype=torch.int64, device=mask.device)


def mean_bits(bits, mask):
    return float((bits * mask).sum() / mask.sum())
