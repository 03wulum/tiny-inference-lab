# Experiment 05: method, controls, limitations

## What changed from experiment 04 (all fixed before this run)

- **Loop fix in the method.** `soft_follow(..., avoid_topic=True)` in training, and `store.answer(..., avoid_topic=True)` at inference. The topic entity is removed after every hop.
- **Complexity levels.** MetaQA test (1/2/3 hops) plus compositional questions from `synth.py` at 1-5 hops over the same KB. These are typed walks (movie → attribute → movie ...) with one phrase per pointer type and 1-50 gold answers, and gold = the non-backtracking walk. Topics with id % 10 == 0 are test-only. Training uses 20k compositional questions per hop for hops 1-3 only.
- **Rivals** (no memory; all see the QA training set plus every fact as a pseudo-question; byte budgets matched to the fixed-hop router + store in fp32):
  - `mem_1x`, `mem_4x`: the transformer memoriser from experiment 04.
  - `kge_1x`: ComplEx entity embeddings with a question encoder that outputs one complex relation vector, scored over all entities (EmbedKGQA-style, trained jointly from scratch).
- **Router flavors:** `ptr` (exp 04 architecture), `rec` (recurrent: conv + transformer encoder with no absolute positions, one GRU hop step with coverage-penalised attention, a linear distance-to-entity term, and a halting head fed by "unused relation words remaining"; d=112 so it fits the byte budget), `bow` (bag of words), `tiny` (d=32, 1 layer), `ptr_0.1pct` (0.1% of the questions).

## Budgets

Training is one process with 12 runs and ~370 s total (hard cap 600 s). Per-run budgets are in `train.py`'s `RUNS`. Weights are stored as fp16 `.npz`, about 22 MB total plus the 4.2 MB store.

## Controls

- N1: the scrambled store (same keys and out-degree, shuffled targets) with `ptr_s0`.
- `ptr_untrained`: a random router with the real store.
- `ptr_plainwalk`: the same `ptr_s0` weights, answering with the loop fix switched off. This shows what the fix is worth.
- 3 seeds for `ptr` and `rec`; criteria use the worst.

## Limitations

- The compositional questions are template-built, so they test composition and length rather than language variety. MetaQA covers the natural-language side, up to 3 hops.
- Entity linking is still free (the name is in brackets).
- `ptr` has 3 hop slots by design, so it cannot answer 4-5 hops. Only `rec` is judged there.
- Latency depends on the machine; C4 uses the machine that runs `run.py`.
