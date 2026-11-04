"""
Pass criteria for experiment 01, with thresholds taken from the paper or from the baseline.

Honesty note: experiment 01 was built before the SOP existed, so these criteria were written
down after a first run (retro-registered, 2026-11-04). The thresholds come from the
paper's theorems and reported numbers, not from our results. From experiment 02 on, criteria
are committed before the first full run.

Each criterion must hold on EVERY perturbation (several dimensions, seeds, and data draws)
to PASS. The negative control is a variant that SHOULD fail. If it passes, the test cannot
tell a working method from a broken one and the verdict is INCONCLUSIVE.
Verdicts (SOP v0.6): PASS = every criterion passes and the negative control fails;
FAIL = the core criterion (C1) fails; INCONCLUSIVE = mixed, or the negative control passed.
"""
from __future__ import annotations

import math

PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
UPPER = {b: math.sqrt(3) * math.pi / 2 * 4.0 ** (-b) for b in PAPER_MSE}  # Theorem 1
LOWER = {b: 4.0 ** (-b) for b in PAPER_MSE}                               # Theorem 3

CRITERIA = [
    dict(id="C1", core=True, name="MSE within the paper's proven bounds (1-4 bits)",
         metric="c1_worst_ratio_to_upper", op="<=", threshold=1.0,
         source="Thm 1: MSE <= (sqrt3*pi/2)*4^-b; worst case over bits, dims, seeds"),
    dict(id="C2", name="MSE matches the paper's reported values",
         metric="c2_worst_rel_err", op="<=", threshold=0.15,
         source="Paper reports 0.36/0.117/0.03/0.009 (rounded); allow +-15%"),
    dict(id="C3", name="TurboQuant-prod inner products are unbiased",
         metric="c3_worst_slope_err", op="<=", threshold=0.02,
         source="Thm 2: E[<y, x_hat>] = <y, x>; |slope-1| over bits 1-4"),
    dict(id="C4", name="1-bit MSE quantizer shrinks inner products by 2/pi",
         metric="c4_shrink_err", op="<=", threshold=0.02,
         source="Paper sec. 3.2: bias factor 2/pi = 0.637 at b=1"),
    dict(id="C5", name="3-bit KV cache: attention error at most half of per-token int quantization",
         metric="c5_worst_kl_ratio", op="<=", threshold=0.5,
         source="Baseline: per-token min/max (KIVI-style) at equal or more bits/coord; "
                "worst case over 3 data draws x 2 context lengths"),
]

NEGATIVE_CONTROL = dict(
    id="N1", name="No random rotation stays within the MSE bound on outlier-heavy keys",
    metric="n1_ratio_to_upper", op="<=", threshold=1.0, expect="FAIL",
    source="Same codebook without the rotation: the guarantee no longer applies")


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
    # SOP v0.6 verdicts
    if not neg["as_expected"]:
        verdict = "INCONCLUSIVE"   # the test can't tell working from broken
    elif not core_ok:
        verdict = "FAIL"
    elif passed == len(rows):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"   # mixed results
    return dict(verdict=verdict, passed=passed, total=len(rows), criteria=rows,
                negative_control=neg, registered="retro (2026-11-04), thresholds from paper")
