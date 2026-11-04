# 03 · Best-of-8 rotations: removing TurboQuant's unlucky draws

**Goal:** experiment 01 found that a single random rotation can be an unlucky draw. Test whether trying 8 rotations and keeping the best one, judged on 64 held-out calibration queries, fixes that cheaply.

**Verdict: PASS** (3/3). The worst-case attention error went from 1.14× int quantization (a single random rotation) to 0.46× (best of 8). That clears experiment 01's failed bar on all 6 caches, and selection takes 0.24 s.

[Details](docs/DETAILS.md) · [Plain-language explainer](EXPLAINER.md) · [All numbers](results/details.md)

## Results

| # | Criterion | Measured | Threshold | Result |
|---|---|---|---|---|
| C1 ★ | Best-of-8 attention KL ≤ 0.5× int quantization on all 6 caches | worst 0.46× | ≤ 0.5× | ✅ PASS |
| C2 | Best-of-8 mean KL ≤ 0.7× a single random rotation | 0.61× | ≤ 0.7× | ✅ PASS |
| C3 | Selection ≤ 1 s per cache on one CPU core | 0.24 s | ≤ 1 s | ✅ PASS |
| N1 | *Negative control:* worst-of-8 also meets 0.5× | 3.35× | should FAIL | ✅ failed as expected |

★ = core criterion. 6 KV caches (the same ones as experiment 01) × 5 independent draws of 8 candidates. Criteria were [committed](criteria.py) before the first run.

## Run it

```bash
python run.py           # ~8 s on a laptop CPU
python run.py --quick   # ~2 s
```
