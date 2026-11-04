# 04 · Pointer memory: a tiny network that stores how ideas connect, not the facts

**Goal:** test whether a ~0.3M-parameter router that only knows *which pointer to follow* (facts live in a constant-time dictionary) beats a same-size model that memorises facts, on multi-hop knowledge questions, in microseconds.

**Verdict: PENDING** (criteria pre-registered 2026-11-04; training queued for Fortran's GPU)

[Plain-language explainer](EXPLAINER.md) · [Method and limitations](docs/DETAILS.md)

## Pre-registered criteria

| # | Criterion | Threshold |
|---|---|---|
| C1 ★ | Beats the same-size no-memory model on 2- and 3-hop (MetaQA Hits@1, worst of 3 seeds) | ≥ +20 pts |
| C2 | Accuracy at published-system level on 1 / 2 / 3 hops | ≥ 95 / 94 / 85 |
| C3 | One question end to end on a CPU, batch 1 (median) | ≤ 500 µs |
| C4 | Rewire 10% of movies' facts: accuracy kept with zero retraining | ≥ 90% |
| C5 | Weights + store ≤ 50 MB; GPU training ≤ 10 min | both |
| N1 | *Negative control:* scrambled pointers reach 94 on 2-hop | should FAIL |

★ = core criterion. Thresholds and their sources are in [`criteria.py`](criteria.py).

## Run it

```bash
pip install numpy pytest
python run.py --quick   # ~30 s: evaluates the committed weights on 600 questions
python run.py           # full MetaQA test sets (43k questions)
python -m pytest -q     # correctness checks on a hand-made knowledge base
python run.py --clean   # delete data/ (MetaQA download, ~68 MB)
# training (GPU only, ~5 min):  pip install torch && python train.py && python bench_gpu.py
```
