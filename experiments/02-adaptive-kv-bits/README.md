# 02 · Don't Waste Bits: does per-token KV precision beat plain 4-bit?

**Goal:** check the claim in [Don't Waste Bits!](https://arxiv.org/abs/2604.04722) (2026) that static 4-bit KV quantization costs SmolLM-360M ~8 HellaSwag points, and that choosing each token's precision (2/4/8/16 bits) wins them back.

**Verdict: pending.** Criteria were committed before the first full run. The full run needs a GPU and is queued on Fortran.

[Details](docs/DETAILS.md) · [Criteria](criteria.py)

## Results

| # | Criterion | Measured | Threshold | Result |
|---|---|---|---|---|
| C1 ★ | Static 4-bit KV loses ≥ 5 points vs FP16 (paper: 7.9) | — | ≥ 5 | pending |
| C2 | Adaptive (≤ 4 avg bits) within 0.5 points of FP16 (paper: 0.3) | — | ≤ 0.5 | pending |
| C3 | Adaptive beats static 4-bit by ≥ 2 points (paper: 7.6) | — | ≥ 2 | pending |
| N1 | *Negative control:* static 2-bit stays within 0.5 points | — | should FAIL | pending |

★ = core criterion. Each criterion must hold on both SmolLM-360M and SmolLM-135M, on 2,000 HellaSwag questions, using paired differences.

## Run it

```bash
pip install -r requirements.txt   # see the file for the torch/CUDA line
python run.py            # both models, 2,000 questions; ~10 min on a GPU
python run.py --quick    # SmolLM-135M, 40 questions; ~1-2 min on a CPU
python -m pytest -q      # 6 checks on a tiny random model, no downloads
python run.py --clean    # delete downloaded data
```
