"""
Pre-registered pass criteria for experiment 03 (written 2026-11-04, before any run).

Follow-up to experiment 01, whose C5 failed on 1 of 6 KV caches because a single random
rotation can be an unlucky draw (spread of attention KL across rotations > its mean).
Hypothesis: drawing k = 8 candidate rotations and keeping the one with the lowest attention
KL on 64 *calibration* queries (held out from the evaluation queries) removes that tail.

Setting = experiment 01's: synthetic outlier-heavy KV caches, d = 128, 3 bits, contexts of
1,024 and 4,096 tokens x 3 data draws = 6 caches; 256 evaluation queries per cache;
5 independent repeats of the 8-candidate draw.

Verdicts (SOP v0.6): PASS = all criteria pass and the negative control fails;
FAIL = the core criterion C1 fails; INCONCLUSIVE = mixed, or the negative control passed.
"""
from __future__ import annotations

CRITERIA = [
    dict(id="C1", core=True,
         name="Best-of-8: attention KL <= 0.5x int quantization on all 6 caches (exp 01's C5)",
         metric="c1_worst_kl_ratio", op="<=", threshold=0.5,
         source="Experiment 01 C5 threshold; worst over caches and repeats"),
    dict(id="C2", name="Best-of-8 mean KL <= 0.7x a single random rotation's mean KL",
         metric="c2_kl_vs_random", op="<=", threshold=0.7,
         source="Needs a material gain, not noise; mean over caches and repeats"),
    dict(id="C3", name="Selection costs <= 1 s per cache on one CPU core",
         metric="c3_select_seconds", op="<=", threshold=1.0,
         source="Must be cheap enough to run once per layer at model load"),
]

NEGATIVE_CONTROL = dict(
    id="N1", name="Worst-of-8 (same signal, inverted) also meets C1's 0.5x on all caches",
    metric="n1_worst_kl_ratio", op="<=", threshold=0.5, expect="FAIL",
    source="If picking the worst candidate still passes, the selection signal carries no information")


def _ok(v, op, t):
    return v <= t if op == "<=" else v >= t


def evaluate(metrics: dict) -> dict:
    rows = [{**c, "value": metrics[c["metric"]],
             "result": "PASS" if _ok(metrics[c["metric"]], c["op"], c["threshold"]) else "FAIL"}
            for c in CRITERIA]
    n = NEGATIVE_CONTROL
    neg = {**n, "value": metrics[n["metric"]]}
    neg["result"] = "PASS" if _ok(neg["value"], n["op"], n["threshold"]) else "FAIL"
    neg["as_expected"] = neg["result"] == n["expect"]
    passed = sum(r["result"] == "PASS" for r in rows)
    if not neg["as_expected"]:
        verdict = "INCONCLUSIVE"
    elif not all(r["result"] == "PASS" for r in rows if r.get("core")):
        verdict = "FAIL"
    elif passed == len(rows):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return dict(verdict=verdict, passed=passed, total=len(rows), criteria=rows,
                negative_control=neg, registered="pre-registered 2026-11-04, before the first run")
