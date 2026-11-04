"""
Pre-registered pass criteria for experiment 02 (written 2026-11-04, before any full run).

Paper: "Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs",
arXiv:2604.04722. Headline (SmolLM-360M, HellaSwag): FP16 41.50%, static 4-bit KV 33.60%,
adaptive 41.20%, i.e. static 4-bit loses 7.9 points and adaptive recovers to within 0.3.

Metric: HellaSwag zero-shot accuracy (lm-eval-harness `acc` formulation), first N
validation examples. Every criterion must hold on BOTH models (SmolLM-360M, the paper's
headline, and SmolLM-135M) to PASS. Differences are paired (same questions).

Verdicts (SOP v0.6): PASS = all criteria pass and the negative control fails;
FAIL = the core criterion C1 fails; INCONCLUSIVE = mixed, or the negative control passed.
"""
from __future__ import annotations

CRITERIA = [
    dict(id="C1", core=True, name="Static 4-bit KV loses >= 5 points vs FP (paper: 7.9)",
         metric="c1_static4_drop", op=">=", threshold=5.0,
         source="Paper Table 3: 41.50 -> 33.60 on SmolLM-360M; min over models"),
    dict(id="C2", name="Adaptive (<= 4 avg bits) within 0.5 points of FP (paper: 0.3)",
         metric="c2_adaptive_gap", op="<=", threshold=0.5,
         source="Paper abstract; max over models"),
    dict(id="C3", name="Adaptive beats static 4-bit by >= 2 points (paper: 7.6)",
         metric="c3_adaptive_gain", op=">=", threshold=2.0,
         source="Paper abstract; min over models"),
]

NEGATIVE_CONTROL = dict(
    id="N1", name="Static 2-bit KV everywhere stays within 0.5 points of FP",
    metric="n1_static2_gap", op="<=", threshold=0.5, expect="FAIL",
    source="2-bit plain min/max KV is known to be lossy; if this passes the eval can't discriminate")


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
                negative_control=neg, registered="pre-registered 2026-11-04, before the first full run")
