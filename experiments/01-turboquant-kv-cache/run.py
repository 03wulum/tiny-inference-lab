"""
Experiment 01: does TurboQuant deliver what the paper claims, on a laptop CPU?

  python run.py            # full run, ~1 min on a laptop CPU
  python run.py --quick    # fewer sizes/seeds, a few seconds

Prints a PASS/FAIL table and verdict. Writes results/verdict.json, results/details.md and
plots (if matplotlib is installed). Criteria live in criteria.py.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from criteria import LOWER, PAPER_MSE, UPPER, evaluate
from turboquant import TurboQuantMSE, TurboQuantProd, UniformMinMax

OUT = Path(__file__).parent / "results"


def unit_vectors(n, d, rng):
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def mse(X, q):
    return float(np.mean(np.sum((X - q(X)) ** 2, axis=1)))


# ---- Claims 1-2: distortion vs the paper's bounds and reported numbers ----------------
def distortion(n, dims, seeds):
    rows = []
    for d in dims:
        for s in seeds:
            X = unit_vectors(n, d, np.random.default_rng(1000 + s))
            for b in [1, 2, 3, 4]:
                m = mse(X, TurboQuantMSE(d, b, seed=s))
                rows.append(dict(d=d, seed=s, bits=b, mse=m, lower=LOWER[b], upper=UPPER[b],
                                 paper=PAPER_MSE[b], ratio_upper=m / UPPER[b]))
    return rows


# ---- Claims 3-4: inner-product bias ---------------------------------------------------
def bias(n, d, seeds):
    rows = []
    for s in seeds:
        rng = np.random.default_rng(2000 + s)
        X = unit_vectors(n, d, rng)
        Y = 0.6 * X + 0.4 * unit_vectors(n, d, rng)  # correlated, so <x,y> is not ~0
        true = np.sum(X * Y, axis=1)
        for b in [1, 2, 3, 4]:
            for name, q in [("TurboQuant-mse", TurboQuantMSE(d, b, seed=s)),
                            ("TurboQuant-prod", TurboQuantProd(d, b, seed=s))]:
                est = np.sum(q(X) * Y, axis=1)
                rows.append(dict(method=name, seed=s, bits=b,
                                 slope=float(np.dot(est, true) / np.dot(true, true)),
                                 rmse=float(np.sqrt(np.mean((est - true) ** 2)))))
    return rows


# ---- Claim 5: attention over a compressed, outlier-heavy KV cache ---------------------
def synthetic_kv(n_tok, d, n_q, rng):
    """Keys with a few persistent high-magnitude channels (as in real LLM key caches,
    cf. KIVI / KVQuant), varying token norms, and queries that attend sharply."""
    K = rng.standard_normal((n_tok, d))
    ch = rng.choice(d, size=4, replace=False)
    K[:, ch] = K[:, ch] * 4 + rng.choice([-10, 10], size=4)
    K *= rng.lognormal(0, 0.3, size=(n_tok, 1))
    V = rng.standard_normal((n_tok, d)) * rng.lognormal(0, 0.3, size=(n_tok, 1))
    Q = 1.5 * K[rng.integers(0, n_tok, size=n_q)] + rng.standard_normal((n_q, d)) * 2.0
    return Q, K, V


def attention(Q, K, V):
    L = Q @ K.T / math.sqrt(Q.shape[1])
    P = np.exp(L - L.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P, P @ V


def attn_metrics(P, O, Ph, Oh):
    kl = float(np.mean(np.sum(P * (np.log(P + 1e-30) - np.log(Ph + 1e-30)), axis=1)))
    t, h = np.argsort(-P, 1)[:, :8], np.argsort(-Ph, 1)[:, :8]
    top8 = float(np.mean([len(set(a) & set(b)) / 8 for a, b in zip(t, h)]))
    cos = float(np.mean(np.sum(O * Oh, 1) / (np.linalg.norm(O, axis=1) * np.linalg.norm(Oh, axis=1))))
    return kl, top8, cos


METHODS = {
    "Uniform min/max (per-token)": lambda d, b, s: (UniformMinMax(d, b), UniformMinMax(d, b)),
    "TurboQuant-mse K+V": lambda d, b, s: (TurboQuantMSE(d, b, seed=s), TurboQuantMSE(d, b, seed=100 + s)),
    "TurboQuant-prod K, mse V": lambda d, b, s: (TurboQuantProd(d, b, seed=s), TurboQuantMSE(d, b, seed=100 + s)),
}


def kv_attention(contexts, d, n_q, data_seeds, rot_seeds):
    rows = []
    for n_tok, ds in [(c, s) for c in contexts for s in data_seeds]:
        Q, K, V = synthetic_kv(n_tok, d, n_q, np.random.default_rng(3000 + ds))
        P, O = attention(Q, K, V)
        for b in [2, 3, 4]:
            for name, make in METHODS.items():
                seeds = [0] if name.startswith("Uniform") else rot_seeds
                ms = []
                for s in seeds:
                    qk, qv = make(d, b, s)
                    ms.append(attn_metrics(P, O, *attention(Q, qk(K), qv(V))))
                ms = np.array(ms)
                rows.append(dict(context=n_tok, data_seed=ds, method=name, bits=b, bits_per_coord=qk.bits_per_coord(),
                                 compression=16 / qk.bits_per_coord(), kl=ms[:, 0].mean(),
                                 kl_std=ms[:, 0].std(), top8=ms[:, 1].mean(), out_cos=ms[:, 2].mean()))
    return rows


# ---- Negative control: same codebook, no rotation, on outlier-heavy keys ---------------
def negative_control(n_tok, d, bits=3):
    _, K, _ = synthetic_kv(n_tok, d, 1, np.random.default_rng(4000))
    U = K / np.linalg.norm(K, axis=1, keepdims=True)
    with_rot = mse(U, TurboQuantMSE(d, bits, seed=0))
    no_rot = mse(U, TurboQuantMSE(d, bits, rotate=False))
    return dict(bits=bits, mse_rotated=with_rot, mse_no_rotation=no_rot, upper=UPPER[bits])


def throughput(n, d):
    X = np.random.default_rng(5).standard_normal((n, d))
    rows = []
    for name, q in [("Uniform min/max", UniformMinMax(d, 3)), ("TurboQuant-mse 3b", TurboQuantMSE(d, 3)),
                    ("TurboQuant-prod 3b", TurboQuantProd(d, 3))]:
        q.quantize(X[:100])
        t = time.perf_counter()
        q.quantize(X)
        rows.append(dict(method=name, vec_per_s=n / (time.perf_counter() - t)))
    return rows


# ---- reporting --------------------------------------------------------------------------
def md_table(rows, cols, fmt):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n"
    return out


def plots(dist, kv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    sub = [r for r in dist if r["d"] == 128 and r["seed"] == dist[0]["seed"]]
    bits = [r["bits"] for r in sub]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(bits, [r["upper"] for r in sub], "--", color="#999", label="paper upper bound")
    ax.semilogy(bits, [r["lower"] for r in sub], ":", color="#999", label="lower bound (any quantizer)")
    ax.semilogy(bits, [r["mse"] for r in sub], "o-", color="#2a6fdb", label="TurboQuant measured")
    ax.semilogy(bits, [r["paper"] for r in sub], "x", color="#d9534f", ms=9, label="paper reported")
    ax.set(xlabel="bits per coordinate", ylabel="MSE (unit vectors)", xticks=bits,
           title="Distortion vs bounds (d=128)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "claim1_mse.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, c in zip(METHODS, ["#999", "#2a6fdb", "#1a9e5c"]):
        rs = [r for r in kv if r["method"] == name]
        rs = [dict(bits=b, top8=np.mean([r["top8"] for r in rs if r["bits"] == b])) for b in (2, 3, 4)]
        ax.plot([r["bits"] for r in rs], [r["top8"] for r in rs], "o-", color=c, label=name)
    ax.set(xlabel="bits per coordinate", ylabel="top-8 attention recall", xticks=[2, 3, 4],
           ylim=(0, 1.02), title="Attention fidelity, outlier-heavy KV cache (mean of draws)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "claim3_attention.png", dpi=150); plt.close(fig)
    return ["claim1_mse.png", "claim3_attention.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = a.quick
    global OUT
    if q:  # never overwrite the committed full-run results with a smoke run
        OUT = OUT / "quick"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    n = 4_000 if q else 20_000
    dist = distortion(n, [128] if q else [64, 128, 1024], [0] if q else [0, 1, 2])
    bia = bias(n, 128, [0] if q else [0, 1, 2])
    kv = kv_attention([1024] if q else [1024, 4096], 128, 64 if q else 256,
                      [0] if q else [0, 1, 2], [0, 1] if q else [0, 1, 2, 3, 4])
    neg = negative_control(1024 if q else 4096, 128)
    tp = throughput(n, 128)

    # ---- metrics the criteria are judged on (worst case over all perturbations) ----
    m128 = {b: np.mean([r["mse"] for r in dist if r["d"] == 128 and r["bits"] == b]) for b in PAPER_MSE}
    kv3 = {}
    for r in kv:
        if r["bits"] == 3:
            kv3.setdefault((r["context"], r["data_seed"]), {})[r["method"]] = r["kl"]
    metrics = dict(
        c1_worst_ratio_to_upper=max(r["ratio_upper"] for r in dist),
        c2_worst_rel_err=max(abs(m128[b] - PAPER_MSE[b]) / PAPER_MSE[b] for b in PAPER_MSE),
        c3_worst_slope_err=max(abs(r["slope"] - 1) for r in bia if r["method"] == "TurboQuant-prod"),
        c4_shrink_err=max(abs(r["slope"] - 2 / math.pi) for r in bia
                          if r["method"] == "TurboQuant-mse" and r["bits"] == 1),
        c5_worst_kl_ratio=max(v["TurboQuant-mse K+V"] / v["Uniform min/max (per-token)"] for v in kv3.values()),
        n1_ratio_to_upper=neg["mse_no_rotation"] / neg["upper"],
    )
    c5_ratios = {k: v["TurboQuant-mse K+V"] / v["Uniform min/max (per-token)"] for k, v in kv3.items()}
    metrics["c5_draws_passing"] = f"{sum(x <= 0.5 for x in c5_ratios.values())}/{len(c5_ratios)}"
    v = evaluate(metrics)
    tq3 = [r for r in kv if r["bits"] == 3 and r["method"] == "TurboQuant-mse K+V"]
    spread = max(r["kl_std"] / r["kl"] for r in tq3)
    if spread > 0.7:
        v["intriguing"] = (f"The random rotation is a lottery: on one KV cache the attention error's spread "
                           f"across rotations was {spread:.1f}x its mean, so picking the rotation may matter "
                           f"more than the bit width.")
    runtime = time.perf_counter() - t0

    # ---- console + verdict.json ----
    print(f"\nExperiment 01 · TurboQuant  ({'quick' if q else 'full'} run, {runtime:.1f}s)\n")
    print(f"{'':4}{'criterion':<72}{'value':>9}  {'threshold':>10}  result")
    for r in v["criteria"] + [v["negative_control"]]:
        tag = r["result"] + (" (expected FAIL ✓)" if r.get("expect") and r["as_expected"] else
                             " (expected FAIL ✗)" if r.get("expect") else "")
        print(f"{r['id']:<4}{r['name']:<72}{r['value']:>9.3f}  {r['op']} {r['threshold']:<7}  {tag}")
    print(f"\nC5 per KV cache: {metrics['c5_draws_passing']} draws within threshold")
    print(f"\nVERDICT: {v['verdict']}  ({v['passed']}/{v['total']} criteria passed)")
    if "intriguing" in v:
        print(f"✨ INTRIGUING: {v['intriguing']}")
    print()
    (OUT / "verdict.json").write_text(json.dumps(
        {"experiment": "01-turboquant-kv-cache", "paper": "arXiv:2504.19874", "quick": q,
         "runtime_s": round(runtime, 1), **v, "metrics": metrics}, indent=2, default=float))

    # ---- details.md: every number behind the verdict ----
    f4 = "{:.4f}"
    md = f"# Experiment 01 · full results\n\n_{'quick' if q else 'full'} run, {runtime:.1f}s. "
    md += f"Verdict **{v['verdict']}** ({v['passed']}/{v['total']}). Generated by `run.py`; do not edit._\n\n"
    md += "## Criteria\n\n" + md_table(
        v["criteria"] + [v["negative_control"]], ["id", "name", "value", "op", "threshold", "result", "source"],
        {"value": "{:.4f}"})
    md += "\n## 1 · MSE vs bounds (unit vectors, every dim and seed)\n\n" + md_table(
        dist, ["d", "seed", "bits", "mse", "paper", "lower", "upper", "ratio_upper"],
        {"mse": f4, "lower": f4, "upper": f4, "ratio_upper": "{:.3f}"})
    md += "\n## 2 · Inner-product bias (slope 1.0 = unbiased; 2/pi = 0.637)\n\n" + md_table(
        bia, ["method", "seed", "bits", "slope", "rmse"], {"slope": "{:.3f}", "rmse": f4})
    md += ("\n## 3 · Attention on a compressed, outlier-heavy KV cache\n\n"
           "1,024- and 4,096-token caches, d=128, 3 data draws each. kl = KL(fp attention || quantized), mean over random rotations; "
           "top8 = overlap of the 8 most-attended tokens; out_cos = cosine of attention outputs. "
           "bits_per_coord includes norms/scales; fp16 = 16.\n\n") + md_table(
        kv, ["context", "data_seed", "method", "bits", "bits_per_coord", "compression", "kl", "kl_std", "top8", "out_cos"],
        {"bits_per_coord": "{:.2f}", "compression": "{:.1f}x", "kl": f4, "kl_std": f4,
         "top8": "{:.3f}", "out_cos": f4})
    md += "\n## Negative control · no rotation, outlier-heavy keys, 3 bits\n\n" + md_table(
        [neg], ["bits", "mse_rotated", "mse_no_rotation", "upper"],
        {"mse_rotated": f4, "mse_no_rotation": f4, "upper": f4})
    md += "\n## Throughput (one CPU core, d=128)\n\n" + md_table(tp, ["method", "vec_per_s"], {"vec_per_s": "{:,.0f}"})
    for p in plots(dist, kv):
        md += f"\n![{p}]({p})\n"
    (OUT / "details.md").write_text(md)
    old = OUT / "results.md"
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    main()
