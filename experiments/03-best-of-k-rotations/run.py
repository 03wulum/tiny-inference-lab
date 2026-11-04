"""
Experiment 03: can picking the best of k random rotations remove TurboQuant's unlucky draws?

  python run.py           # 6 caches x 5 repeats x 8 candidates, ~1-2 min on a laptop CPU
  python run.py --quick   # 1 cache x 2 repeats, a few seconds

Reuses experiment 01's quantizer and its synthetic KV caches (same seeds, so the caches are
identical to the ones that produced exp 01's C5 failure).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "01-turboquant-kv-cache"))  # append: our own criteria.py must win
from turboquant import TurboQuantMSE, UniformMinMax  # noqa: E402

from criteria import evaluate  # noqa: E402  (this experiment's criteria.py)

D, BITS, K_CAND = 128, 3, 8


# Same generator and seeds as experiment 01 (run.py: synthetic_kv, seed 3000 + data_seed).
def synthetic_kv(n_tok, d, n_q, rng):
    K = rng.standard_normal((n_tok, d))
    ch = rng.choice(d, size=4, replace=False)
    K[:, ch] = K[:, ch] * 4 + rng.choice([-10, 10], size=4)
    K *= rng.lognormal(0, 0.3, size=(n_tok, 1))
    V = rng.standard_normal((n_tok, d)) * rng.lognormal(0, 0.3, size=(n_tok, 1))
    Q = 1.5 * K[rng.integers(0, n_tok, size=n_q)] + rng.standard_normal((n_q, d)) * 2.0
    return Q, K, V


def calib_queries(K, n, seed):
    """Held-out queries from the same recipe, drawn with a different seed."""
    rng = np.random.default_rng(6000 + seed)
    return 1.5 * K[rng.integers(0, len(K), size=n)] + rng.standard_normal((n, K.shape[1])) * 2.0


def attn_probs(Q, K):
    L = Q @ K.T / math.sqrt(Q.shape[1])
    P = np.exp(L - L.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def kl(P, Ph):
    return float(np.mean(np.sum(P * (np.log(P + 1e-30) - np.log(Ph + 1e-30)), axis=1)))


def run_cache(n_tok, ds, repeats, n_calib=64):
    Q, K, _ = synthetic_kv(n_tok, D, 256, np.random.default_rng(3000 + ds))
    Qc = calib_queries(K, n_calib, ds)
    P, Pc = attn_probs(Q, K), attn_probs(Qc, K)
    kl_int = kl(P, attn_probs(Q, UniformMinMax(D, BITS)(K)))
    out = []
    for r in range(repeats):
        t = time.perf_counter()
        cands = []
        for c in range(K_CAND):
            Kh = TurboQuantMSE(D, BITS, seed=1000 * r + c)(K)
            cands.append((kl(Pc, attn_probs(Qc, Kh)), Kh))   # selection signal: calib queries only
        sel_s = time.perf_counter() - t
        calib = [x for x, _ in cands]
        best, worst = int(np.argmin(calib)), int(np.argmax(calib))
        ev = [kl(P, attn_probs(Q, Kh)) for _, Kh in cands]   # evaluation: held-out queries
        out.append(dict(context=n_tok, data_seed=ds, repeat=r, kl_int=kl_int,
                        kl_random=ev[0], kl_best=ev[best], kl_worst=ev[worst],
                        kl_oracle=min(ev), select_s=sel_s))
    return out


def md_table(rows, cols, fmt):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return out + "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_dir = HERE / "results" / ("quick" if a.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    caches = [(1024, 0)] if a.quick else [(c, s) for c in (1024, 4096) for s in (0, 1, 2)]
    rows = [row for c, s in caches for row in run_cache(c, s, 2 if a.quick else 5)]

    metrics = dict(
        c1_worst_kl_ratio=max(r["kl_best"] / r["kl_int"] for r in rows),
        c2_kl_vs_random=float(np.mean([r["kl_best"] for r in rows]) / np.mean([r["kl_random"] for r in rows])),
        c3_select_seconds=max(r["select_s"] for r in rows),
        n1_worst_kl_ratio=max(r["kl_worst"] / r["kl_int"] for r in rows),
    )
    v = evaluate(metrics)
    runtime = time.perf_counter() - t0

    print(f"\nExperiment 03 · best-of-{K_CAND} rotations  ({'quick' if a.quick else 'full'}, {runtime:.1f}s)\n")
    for r in v["criteria"] + [v["negative_control"]]:
        tag = r["result"] + (" (expected FAIL ✓)" if r.get("expect") and r["as_expected"] else
                             " (expected FAIL ✗)" if r.get("expect") else "")
        print(f"{r['id']:<4}{r['name']:<84}{r['value']:>7.3f}  {r['op']} {r['threshold']:<4} {tag}")
    print(f"\nVERDICT: {v['verdict']}  ({v['passed']}/{v['total']} criteria passed)\n")

    (out_dir / "verdict.json").write_text(json.dumps(
        {"experiment": "03-best-of-k-rotations", "follows": "01-turboquant-kv-cache", "quick": a.quick,
         "runtime_s": round(runtime, 1), **v, "metrics": metrics}, indent=2, default=float))
    for r in rows:
        r["ratio_best"], r["ratio_random"] = r["kl_best"] / r["kl_int"], r["kl_random"] / r["kl_int"]
    md = (f"# Experiment 03 · full results\n\n_{'quick' if a.quick else 'full'} run, {runtime:.1f}s. "
          f"Verdict **{v['verdict']}** ({v['passed']}/{v['total']}). Generated by `run.py`; do not edit._\n\n"
          "KL = KL(fp attention || 3-bit attention) on 256 held-out evaluation queries. `best`/`worst` = "
          "candidate picked by lowest/highest KL on 64 separate calibration queries; `random` = the first "
          "candidate (one random rotation, as in exp 01); `oracle` = best candidate judged on the "
          "evaluation queries themselves (an upper bound, not achievable in practice).\n\n")
    md += md_table(rows, ["context", "data_seed", "repeat", "kl_int", "kl_random", "kl_best", "kl_worst",
                          "kl_oracle", "ratio_random", "ratio_best", "select_s"],
                   {k: "{:.4f}" for k in ["kl_int", "kl_random", "kl_best", "kl_worst", "kl_oracle"]}
                   | {"ratio_random": "{:.2f}", "ratio_best": "{:.2f}", "select_s": "{:.3f}"})
    md += "\n## Criteria\n\n" + md_table(
        v["criteria"] + [v["negative_control"]], ["id", "name", "value", "op", "threshold", "result"],
        {"value": "{:.3f}"})
    (out_dir / "details.md").write_text(md)


if __name__ == "__main__":
    main()
