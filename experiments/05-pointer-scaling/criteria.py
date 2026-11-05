"""
Pre-registered pass criteria for experiment 05 (written 2026-11-05, before any training run).

Follow-up to experiment 04 (INCONCLUSIVE 3/5). Changes, all decided BEFORE this run:
  * The loop fix is part of the method: walks never pass back through the topic entity, in training
    (soft walk) and at inference. Fresh training from scratch, new seeds' weights.
  * Complexity levels: MetaQA 1/2/3 hops (real questions) + compositional questions over the same KB
    at 1/2/3/4/5 hops (synth.py). Training sees 1-3 hops only; test topics never appear in training.
  * Rival model types at every level: no-memory transformer at 1x and 4x bytes, and a
    knowledge-graph-embedding model (ComplEx, EmbedKGQA-style) at 1x bytes.
  * Router flavors: fixed-hop (exp 04 + fix), recurrent (one shared hop step + halting), order-blind
    (bag of words), tiny (18k params).

Accuracy criteria use the WORST of 3 seeds. Hits@1 throughout.
Thresholds for C1 are exp 04's C2 thresholds, unchanged (GraftNet 97.0/94.8/77.7,
ChatGPT+StructGPT 93.9 (2-hop) / 80.2 (3-hop); TransferNet 97.5/100/100).
"""
from __future__ import annotations

THRESH = {1: 95.0, 2: 94.0, 3: 85.0}
INTRIGUING = ""   # set only after the run, if something beyond expectation shows up (never changes the verdict)

CRITERIA = [
    dict(id="C1", core=True, name="Fixed-hop router + loop fix reaches >= 95 / 94 / 85 on MetaQA",
         metric="c1_min_margin_pts", op=">=", threshold=0.0,
         source="Exp 04's C2 bars, unchanged; worst of 3 fresh seeds"),
    dict(id="C2", name="Beats every no-memory rival at every trained level by >= 20 pts",
         metric="c2_min_gap_pts", op=">=", threshold=20.0,
         source="Rivals: transformer 1x and 4x bytes, ComplEx KGE 1x; levels MetaQA 1-3 + compositional 1-3"),
    dict(id="C3", name="Recurrent router: >= 80% on 4- and 5-hop questions it never trained on",
         metric="c3_rec_min_45", op=">=", threshold=80.0,
         source="Length generalisation; exp 04's fixed router scored 0% one level past its training"),
    dict(id="C4", name="One question end to end in <= 500 us on a CPU (batch 1, fixed-hop router)",
         metric="c4_e2e_us", op="<=", threshold=500.0,
         source="Exp 04 measured 549 us on a shared machine; same bar"),
    dict(id="C5", name="Budget: kept weights + store <= 50 MB, GPU training <= 10 min",
         metric="c5_budget_ok", op=">=", threshold=1.0, source="Project rules"),
]

NEGATIVE_CONTROL = dict(
    id="N1", name="Scrambled pointers reach 94 on MetaQA 2-hop",
    metric="n1_random_2hop", op=">=", threshold=94.0, expect="FAIL",
    source="Same router, store targets shuffled; must collapse")


def _check(value, op, threshold):
    return value <= threshold if op == "<=" else value >= threshold


def evaluate(metrics: dict) -> dict:
    rows = []
    for c in CRITERIA:
        v = metrics[c["metric"]]
        rows.append({**c, "value": v, "result": "PASS" if _check(v, c["op"], c["threshold"]) else "FAIL"})
    n = NEGATIVE_CONTROL
    v = metrics[n["metric"]]
    neg = {**n, "value": v, "result": "PASS" if _check(v, n["op"], n["threshold"]) else "FAIL"}
    neg["as_expected"] = neg["result"] == n["expect"]
    passed = sum(r["result"] == "PASS" for r in rows)
    core_ok = all(r["result"] == "PASS" for r in rows if r.get("core"))
    if not neg["as_expected"]:
        verdict = "INCONCLUSIVE"
    elif not core_ok:
        verdict = "FAIL"
    elif passed == len(rows):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return dict(verdict=verdict, passed=passed, total=len(rows), criteria=rows, negative_control=neg,
                registered="2026-11-05, before the first training run")
