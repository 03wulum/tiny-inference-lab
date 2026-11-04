# 02 · Don't Waste Bits: details

**Paper:** *Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs.* [arXiv:2604.04722](https://arxiv.org/abs/2604.04722), 2026.
**Why it matters on-device:** if a few tokens need high precision and most can go to 2 bits, a phone could hold a much longer context in the same memory.
**Size budget:** nothing trained, nothing kept. Temporary downloads are SmolLM-135M + 360M (~1 GB, HF cache) and HellaSwag validation (11 MB, `data/`). `run.py --clean` removes the data.

## What the paper claims

For SmolLM-360M on HellaSwag: FP16 41.50%, static 4-bit KV 33.60%, adaptive 41.20%, plus 17.75% lower latency than static. The adaptive controller is a small MLP over four per-token features, s_t = [entropy H, rarity R, attention variance V, confidence C], choosing among {2, 4, 8, 16} bits.

**Why test it:** a 7.9-point loss from 4-bit KV is far larger than the near-lossless 4-bit KV results in KIVI and KVQuant. If a reasonable static 4-bit baseline is near-lossless, the adaptive gain has nothing to recover.

## Method

- **Simulation:** a custom transformers attention function fake-quantizes the post-RoPE K and V of every token at that token's bit-width, then runs SDPA. This is the standard accuracy protocol for KV quantization. It does not measure latency, so the paper's latency claim is out of scope.
- **Static baseline:** per-token, per-head asymmetric min/max with an fp16 scale + zero point per 32-channel group (KIVI-style group size).
- **Adaptive policy:** the paper's four features, computed from an FP16 pass (H = next-token entropy, R = smoothed −log p of the token, V = variance across heads of the last-layer attention the token receives, C = max next-token probability). Tokens are ranked by z(H) + z(R) + z(V) − z(C). The first token (attention sink) keeps 16 bits, the top 10% get 8 bits, then as many as the budget allows get 4 bits, and the rest get 2 bits. The budget is exactly ≤ 3.8 average payload bits. *Deviation:* the paper's MLP is underspecified (loss weights α, β, γ, the confidence feature, and training labels are not given), so I used a budgeted ranking over the same features.
- **Eval:** HellaSwag validation (first 2,000), zero-shot, lm-eval-harness preprocessing and `acc` / `acc_norm`. Differences are paired with a bootstrap 95% CI.

## Controls

- **Negative control (N1):** static 2-bit everywhere should lose clearly. If it doesn't, the eval can't discriminate.
- **Ablation:** random per-token bits at the same 3.8-bit budget. Do the features matter, or only the budget?
- **Bridge to 01:** TurboQuant 3-bit (from experiment 01) on the real KV cache. Reported, not judged.
- **Perturbation:** two model sizes (135M, 360M). Every criterion must hold on both.

## Results

Pending the full run.

## Where it runs

The full run uses GitHub's free CPU runners via the manual `full-run` workflow: 2 models × 8 shards of 250 questions each, merged by `run.py --aggregate`. On CPU the "FP16" baseline actually runs in fp32, which is a slightly stronger reference than fp16, and the quantized variants are computed from the same fp32 tensors. The random-policy ablation's seeds depend on the shard layout.
