# 01 · TurboQuant: compress the KV cache to ~3 bits with no calibration

**Goal:** check whether [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) delivers its proven error bounds, and whether that beats standard int quantization on an outlier-heavy KV cache.

**Verdict: INCONCLUSIVE** (4/5 criteria pass). Every claim from the paper reproduced. My application test (C5) passed on 5 of 6 KV caches and failed on 1.
✨ **INTRIGUING:** the random rotation is a lottery. On some caches, which rotation you draw moves the attention error as much as the bit width does.

[Details](docs/DETAILS.md) · [Plain-language explainer](EXPLAINER.md) · [All numbers](results/details.md)

## Results

| # | Criterion | Measured | Threshold | Result |
|---|---|---|---|---|
| C1 ★ | MSE within the paper's proven bound, 1-4 bits | worst 0.89× bound | ≤ 1.0× | ✅ PASS |
| C2 | MSE matches the paper's reported numbers | worst 13% off | ≤ 15% | ✅ PASS |
| C3 | "prod" variant gives unbiased inner products | slope off by 0.002 | ≤ 0.02 | ✅ PASS |
| C4 | 1-bit MSE variant shrinks inner products by 2/π | off by 0.003 | ≤ 0.02 | ✅ PASS |
| C5 | 3-bit KV cache: attention error ≤ ½ of int quantization | worst 1.05× (5 of 6 caches pass) | ≤ 0.5× | ❌ FAIL |
| N1 | *Negative control:* no rotation stays within the bound | 7.3× bound | should FAIL | ✅ failed as expected |

★ = core criterion. Each value is the worst case across dimensions (64/128/1024), seeds, and data draws. Criteria are in [`criteria.py`](criteria.py). They were written down after a first run, with thresholds taken from the paper.

## Run it

```bash
pip install numpy matplotlib pytest
python run.py           # ~50 s on a laptop CPU → PASS/FAIL table + results/verdict.json
python run.py --quick   # ~5 s
python -m pytest -q     # 10 correctness checks, ~2 s
```
