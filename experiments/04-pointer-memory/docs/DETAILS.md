# Experiment 04: method, controls, limitations

## Hypothesis

A router that holds **no facts**, only a policy for which typed pointer to follow next, plus a
constant-time pointer store, answers multi-hop knowledge questions far better than a no-memory
model of the same total size, runs in microseconds per question on a CPU, and can be updated by
editing the store with no retraining.

## Data

[MetaQA](https://github.com/yuyuz/MetaQA) (Zhang et al., AAAI 2018; Creative Commons) on WikiMovies
(CC-BY 3.0): 134,741 facts, 43,234 entities, 9 relations. Questions: 1-hop 96k/10k/10k,
2-hop 119k/15k/15k, 3-hop 114k/14k/14k (train/dev/test). Downloaded (~68 MB) from the HF mirror
`camazlucas/MetaQA` at a pinned commit into git-ignored `data/`.

## Models

| | What it stores | Params | Input |
|---|---|---|---|
| **Pointer store** | every fact, both directions (267k pointers), CSR | 0 (4.2 MB of int32) | (entity, pointer type) |
| **Router** | how to connect: hop count + pointer type per hop | ~0.3M (2-layer transformer, d=128) | question with the entity blanked as `<E>` |
| **No-memory 1x** | everything in weights: tied entity table + encoder | same fp32 bytes as router + store | question + topic entity embedding |
| **No-memory 4x** | same, 4x the bytes | | |

The no-memory widths are chosen at train time to match bytes (see `weights/train_log.json`).

**Router training (weak supervision).** Loss = binary cross-entropy between gold answers and a
*soft* walk: start at the topic entity, at hop t spread mass along every edge weighted by the
router's probability for that edge's pointer type, clamp at 1, mix hops 1-3 by the predicted hop
count. Only (question, answer set) pairs are used, never gold relation paths.

**Inference (hard pointers).** argmax hop count, argmax pointer type per hop, then exact CSR
lookups. Score = number of paths reaching an entity. Hits@1 = top entity (excluding the topic) is a
gold answer.

**No-memory training.** Same GPU-time budget. Training set = all QA training questions **plus one
pseudo-question per (entity, pointer type) key in the store** (`<fact> <r:directed_by> <E>`), so
it sees every fact. Loss = marginal log-likelihood of the gold set under a softmax over entities.

## Budgets

Training: one process on Fortran's RTX 4080 SUPER, ~5 min total (3 router seeds x 40 s, two
exploratory routers x 25 s, two no-memory models x 60 s), hard cap 10 min. Kept: fp16 weights
(`weights/*.npz`) + the 4.2 MB store, ≤ 50 MB.

## Controls

- **N1 random pointers (negative control, must FAIL):** same router, the store's targets shuffled
  (same keys, same out-degree). Tests that accuracy comes from the memory's content.
- **No memory:** same router, empty store (answers impossible; sanity check that it reports 0).
- **Untrained router + real store:** the memory alone, without learned connection weights.
- **3 seeds:** accuracy criteria use the worst seed.

Exploratory (reported, not in the verdict): a router trained only on 1- and 2-hop questions,
evaluated on 3-hop; a router trained on 1% of the questions.

## Knowledge-edit test (C4)

1. Take 1,500 random 1-hop and 1,500 random 2-hop test questions; keep those whose gold answer set
   is produced by exactly one pointer path (search over all 18 / 324 paths).
2. Rewire 10% of movies: each of their `directed_by / written_by / starred_actors / release_year /
   in_language / has_genre` facts points at a random entity from that relation's range.
3. New gold = that path on the edited store. Keep questions whose answer changed.
4. Score the pointer model with the edited store (router unchanged) against the new answers, and the
   no-memory model (unchanged) against both new and old answers.

## Limitations

- MetaQA marks the topic entity in brackets, so entity linking is a hash lookup. Real text needs NER.
- MetaQA's question templates are few (≤ 100 distinct words), which favours a small router.
- The 2-hop and 3-hop test answers in MetaQA are generated from the same KB, so a perfect path
  gives a perfect answer; noise mostly comes from 1-hop (WikiMovies) labels.
- Latency is machine-dependent; the C3 number is from the machine that ran `run.py`.
- The no-memory baseline gets the same GPU time, not unlimited time; its train-set fit is reported
  so capacity limits and under-training can be told apart.

## Post-hoc analysis (after the pre-registered run; does not change the verdict)

All three seeds landed at 82.8-83.8% on 3-hop, so the error looked systematic. Grouping the misses by
question template showed they are almost all "share actors with" or "starred by X actors" questions.
For these, the router's pointer sequence is correct, but the walk can pass back through the topic
entity, and those paths dominate the path-count ranking. MetaQA's answers come from walks that don't
revisit the topic. `explore_walk.py` re-scores the same weights with `avoid_topic=True` (drop the topic
after every hop): 3-hop worst seed 98.9% (seeds 98.9 / 100 / 100), 1-hop and 2-hop unchanged, 1%-data router
99.6%. Results in `results/nonbacktracking.json`. A follow-up experiment should pre-register the
non-backtracking walk.

GPU latency was measured while another job shared the GPU (launch 17 µs is ~3x a quiet GPU), so treat
`results/latency_gpu.json` as an upper bound.
