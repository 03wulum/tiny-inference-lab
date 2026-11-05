"""
Pre-registered pass criteria for experiment 04 (written 2026-11-04, before any training run).

Hypothesis: a ~0.3M-parameter router that stores NO facts, only "which pointer to follow next",
plus a constant-time pointer store (the knowledge base as a CSR dictionary), answers multi-hop
questions far better than a no-memory model of the same total size, stays microsecond-fast,
and can be updated by editing the store instead of retraining.

Data: MetaQA (Zhang et al., AAAI 2018) test sets, Hits@1. 1-hop 9.9k, 2-hop 14.9k, 3-hop 14.3k questions.
Accuracy criteria use the WORST of 3 training seeds.

Published reference points (Hits@1, 1/2/3-hop), from the papers' tables:
  KV-MemNN 95.8 / 25.1 / 10.1 (MetaQA paper) · GraftNet 97.0 / 94.8 / 77.7 · EmbedKGQA 97.5 / 98.8 / 94.8
  NSM 97.1 / 99.9 / 98.9 · TransferNet 97.5 / 100 / 100
  ChatGPT, no KB (StructGPT paper): 2-hop 31.0, 3-hop 43.2 · ChatGPT + StructGPT: 93.9 / 80.2
"""
from __future__ import annotations

# C2 thresholds: at least GraftNet-level on 1/2-hop, and clearly above GraftNet and
# ChatGPT+StructGPT on 3-hop (77.7 / 80.2).
THRESH = {1: 95.0, 2: 94.0, 3: 85.0}

# ✨ tag added after the pre-registered run (SOP: a tag never changes the verdict)
INTRIGUING = ("Post hoc: a non-backtracking walk lifts the same routers to 98.9-100% on 3-hop with no retraining; a router trained on 1% of the questions matches the full-data router.")

CRITERIA = [
    dict(id="C1", core=True, name="Beats the same-size no-memory model by >= 20 pts on 2- and 3-hop",
         metric="c1_min_gap_pts", op=">=", threshold=20.0,
         source="No-memory baseline gets the same bytes as router + store (fp32), sees every fact, "
                "same GPU time; worst seed, min over 2-hop and 3-hop"),
    dict(id="C2", name="Accuracy at published-system level (>= 95 / 94 / 85 Hits@1)",
         metric="c2_min_margin_pts", op=">=", threshold=0.0,
         source="GraftNet 97.0/94.8/77.7, ChatGPT+StructGPT 93.9 (2-hop) / 80.2 (3-hop); worst seed"),
    dict(id="C3", name="One question end to end in <= 500 us on a CPU (batch 1)",
         metric="c3_e2e_us", op="<=", threshold=500.0,
         source="Obi's target: fast enough to feed a main model; median over 1,000 2-hop questions"),
    dict(id="C4", name="Edit 10% of movies' facts: >= 90% of accuracy kept, no retraining",
         metric="c4_edit_ratio", op=">=", threshold=0.90,
         source="Hits@1 vs the NEW answers after the edit / Hits@1 before the edit, same questions"),
    dict(id="C5", name="Budget: kept weights + store <= 50 MB, GPU training <= 10 min",
         metric="c5_budget_ok", op=">=", threshold=1.0,
         source="Project rules (Obi, 2026-11-04): 50 MB kept, ~5 min training, never > 10"),
]

NEGATIVE_CONTROL = dict(
    id="N1", name="Random pointers (same store shape, shuffled targets) reach 94 on 2-hop",
    metric="n1_random_2hop", op=">=", threshold=94.0, expect="FAIL",
    source="If scrambled pointers still score, the router is not using the memory")


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
                registered="2026-11-04, before the first training run")
