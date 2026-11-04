# 03 · Best-of-k rotations: details

**Follows:** [experiment 01](../../01-turboquant-kv-cache), criterion C5 (failed on 1 of 6 caches) and its ✨ note that the rotation is a lottery.
**Size budget:** NumPy only, < 300 MB RAM, nothing downloaded. A chosen rotation costs d² floats per layer (64 KB at d=128, fp32).

## Method

- Same 6 synthetic outlier-heavy KV caches as experiment 01 (identical generator and seeds), d = 128, TurboQuant-mse at 3 bits on keys.
- For each cache, 5 independent repeats. Each repeat draws k = 8 rotations (seeds 1000·r + c).
- **Selection signal:** attention KL on 64 calibration queries drawn with a different seed from the 256 evaluation queries.
- **Reported:** `random` (candidate 0, i.e. experiment 01's setup), `best` and `worst` by the calibration signal, and `oracle` (best judged on the evaluation queries, an unreachable upper bound).

## Controls

- **Negative control:** worst-of-8 by the same signal must fail C1. It does, at 3.35×, so the signal carries information.
- **Baseline:** per-token min/max int quantization at the same 3 bits, as in experiment 01.
- **Perturbations:** 2 context lengths × 3 data draws × 5 independent candidate sets. C1 is judged on the worst of all 30.

## Results

| | worst ratio to int quant | mean KL |
|---|---|---|
| single random rotation (exp 01) | 1.14× | 0.43 |
| **best of 8 (calibration)** | **0.46×** | **0.26** |
| oracle best of 8 | 0.34× | 0.20 |
| worst of 8 (negative control) | 3.35× | 0.94 |

Full per-cache table: [`results/details.md`](../results/details.md).

## What I found

1. **A small calibration set transfers.** 64 queries were enough to avoid the bad draws on all 30 repeat × cache combinations, without seeing the evaluation queries.
2. **There is still headroom.** The oracle's mean KL is about 22% lower than the calibrated pick's, so a better signal (more calibration queries, or k = 16) would likely gain more.
3. **The selection sometimes picks a slightly worse candidate than the first one** (e.g. 4096-token cache 0, repeat 0: random 0.077, best 0.118). The aim is to cut the tail, and the tail is gone.

## Limitations and next steps

- Synthetic KV caches. The real test is per-layer selection on SmolLM's actual KV cache, which belongs with experiment 02's GPU run.
- Only keys are selected, because attention probabilities depend only on K. Value rotations affect the output but not the KL used here.
