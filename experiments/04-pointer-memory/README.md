# 04 · Pointer memory: a tiny network that stores how ideas connect, not the facts

**Goal:** test whether a 0.34M-parameter router that only knows *which pointer to follow* (facts live in a constant-time dictionary) beats a same-size model that memorises facts, on multi-hop knowledge questions, in microseconds.

**Verdict: INCONCLUSIVE** (3/5 criteria pass). The core claim holds by a wide margin: 52 to 88 points better than a no-memory model of the same size, and edits to the facts work with no retraining. Two criteria missed. 3-hop scored 82.8 against a target of 85, and CPU latency was 549 µs against a target of 500.
✨ **INTRIGUING:** the 3-hop miss came from one walking rule, not from the router. When the walk can't loop back through the question's own entity, the same routers, with no retraining, score 98.9-100% on 3-hop (found after the pre-registered run). A router trained on just 1% of the questions does as well as one trained on all of them.

[Plain-language explainer](EXPLAINER.md) · [Method and limitations](docs/DETAILS.md) · [All numbers](results/details.md)

## Results

| # | Criterion | Measured | Threshold | Result |
|---|---|---|---|---|
| C1 ★ | Beats the same-size no-memory model on 2- and 3-hop | +52.7 pts (worst case: 3-hop, 82.8 vs 30.1) | ≥ +20 | ✅ PASS |
| C2 | Hits@1 at published-system level, 1 / 2 / 3 hops | 95.4 / 97.6 / **82.8** | ≥ 95 / 94 / 85 | ❌ FAIL |
| C3 | One question end to end on a CPU, batch 1 | 549 µs median | ≤ 500 µs | ❌ FAIL |
| C4 | Rewire 10% of movies' facts, no retraining | 99.6% → 99.6% (ratio 1.00) | ≥ 0.90 | ✅ PASS |
| C5 | Weights + store; GPU training | 21 MB; 291 s | ≤ 50 MB; ≤ 10 min | ✅ PASS |
| N1 | *Negative control:* scrambled pointers | 0.4% on 2-hop | should FAIL | ✅ failed as expected |

★ = core criterion. Accuracy is the worst of 3 seeds on the full MetaQA test sets (39k questions). Criteria were written in [`criteria.py`](criteria.py) before training.

| MetaQA Hits@1 (1 / 2 / 3-hop) | Size | Score |
|---|---|---|
| **Pointer router + store (ours, worst seed)** | 0.34M params + 4.2 MB | 95.4 / 97.6 / 82.8 |
| ↳ same routers, non-backtracking walk (post hoc) | same | 95.4 / 97.6 / 98.9 |
| No-memory model, same bytes | 1.45M params | 6.8 / 10.0 / 30.1 |
| No-memory model, 4x bytes | 5.7M params | 16.2 / 10.8 / 34.3 |
| ChatGPT, no knowledge base ([StructGPT](https://github.com/RUCAIBox/StructGPT) paper) | ~100 GB class | – / 31.0 / 43.2 |
| ChatGPT + StructGPT knowledge-base tool | ~100 GB class | – / 93.9 / 80.2 |
| TransferNet (closest prior method) | BERT-size | 97.5 / 100 / 100 |

**What the abstract doesn't tell you:**
- The no-memory models barely learned the facts they were shown: 7% of training facts recalled at 1x size, 26% at 4x, after the same GPU time. Memorising 135k facts into weights is the hard part, and pointers skip it.
- On one question at a time, the CPU (0.55 ms) beat the RTX 4080 SUPER (3.5 ms). That's because each GPU kernel launch cost 17 µs, and the GPU was shared with another job. Where the GPU does reach nanoseconds is bulk lookups: 0.08 ns per pointer across 16M lookups.
- A router that never saw a 3-hop question gets 0% on 3-hop. It learned "how many hops" as a fixed choice, not as a count it can extend.

## Run it

```bash
pip install numpy pytest
python run.py --quick     # ~30 s: evaluates the committed weights on 600 questions
python run.py             # full MetaQA test sets, ~3 min on a CPU
python explore_walk.py    # the post-hoc non-backtracking re-score, ~1 min
python -m pytest -q       # 12 correctness checks on a hand-made knowledge base
python run.py --clean     # delete data/ (MetaQA download)
# training (GPU only, ~5 min):  pip install torch && python train.py && python bench_gpu.py
```
