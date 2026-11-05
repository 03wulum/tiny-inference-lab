# 05 · Pointer memory at 1-5 hops vs every rival type (follow-up to 04)

**Goal:** make experiment 04 conclusive. The loop fix is built in from the start and the routers are retrained from scratch. The test now runs at five complexity levels (1-5 hops, where 4 and 5 are never trained on) against three kinds of no-memory model. It also finds which router flavor is enough to beat each rival at each level.

**Verdict: PENDING** (criteria pre-registered 2026-11-05; GPU run queued on Fortran)

[Plain-language explainer](EXPLAINER.md) · [Method and limitations](docs/DETAILS.md) · [Experiment 04](../04-pointer-memory)

## Pre-registered criteria

| # | Criterion | Threshold |
|---|---|---|
| C1 ★ | Fixed-hop router + loop fix, MetaQA Hits@1 at 1 / 2 / 3 hops (worst of 3 seeds) | ≥ 95 / 94 / 85 |
| C2 | Beats every no-memory rival (transformer 1x and 4x, ComplEx KGE 1x) at every trained level | ≥ +20 pts |
| C3 | Recurrent router on 4- and 5-hop questions it never trained on (worst seed) | ≥ 80% |
| C4 | One question end to end on a CPU, batch 1 (median) | ≤ 500 µs |
| C5 | Weights + store; GPU training | ≤ 50 MB; ≤ 10 min |
| N1 | *Negative control:* scrambled pointers on MetaQA 2-hop | should FAIL (< 94) |

★ = core criterion. Thresholds and their sources: [`criteria.py`](criteria.py).

## Run it

```bash
pip install numpy pytest
python run.py --quick     # ~1 min on the committed weights
python run.py             # full: 39k MetaQA + 7.5k compositional questions
python -m pytest -q       # store, generator, torch<->NumPy parity (parity needs torch)
python run.py --clean
# training (GPU only, ~6 min):  pip install torch && python train.py
```
