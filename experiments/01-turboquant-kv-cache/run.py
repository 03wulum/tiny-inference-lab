"""
Experiment 01: does TurboQuant deliver what the paper claims, on a laptop CPU?

  python run.py            # full run, ~1 minute on a laptop CPU
  python run.py --quick    # smaller sizes, a few seconds

Writes results/results.md (and results/*.png if matplotlib is installed).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from turboquant import (
    TurboQuantMSE,
    TurboQuantProd,
    UniformMinMax,
    mse_lower_bound,
    mse_upper_bound,
)

OUT = Path(__file__).parent / "results"
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}  # values reported in the paper


def unit_vectors(n, d, rng):
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Claim 1: MSE distortion sits between the information-theoretic lower bound 4^-b and the
#          paper's upper bound (sqrt(3)*pi/2) * 4^-b  (~2.7x the optimum, any b, any input).
# --------------------------------------------------------------------------------------
def claim1_mse(n, dims, rng):
    rows = []
    for d in dims:
        X = unit_vectors(n, d, rng)
        for b in [1, 2, 3, 4]:
            tq = TurboQuantMSE(d, b, seed=d + b)
            mse = float(np.mean(np.sum((X - tq(X)) ** 2, axis=1)))
            rows.append(dict(d=d, bits=b, mse=mse, lower=mse_lower_bound(b),
                             upper=mse_upper_bound(b), paper=PAPER_MSE[b],
                             ratio=mse / mse_lower_bound(b)))
    return rows


# --------------------------------------------------------------------------------------
# Claim 2: the MSE quantizer is biased for inner products (shrinks them; at 1 bit by 2/pi),
#          while TurboQuant-prod is unbiased.
# --------------------------------------------------------------------------------------
def claim2_bias(n, d, rng):
    X = unit_vectors(n, d, rng)
    # queries correlated with keys so inner products are not all ~0
    Y = 0.6 * X + 0.4 * unit_vectors(n, d, rng)
    true = np.sum(X * Y, axis=1)
    rows = []
    for b in [1, 2, 3, 4]:
        for name, q in [("TurboQuant-mse", TurboQuantMSE(d, b, seed=7)),
                        ("TurboQuant-prod", TurboQuantProd(d, b, seed=7))]:
            est = np.sum(q(X) * Y, axis=1)
            slope = float(np.dot(est, true) / np.dot(true, true))  # 1.0 == unbiased
            rows.append(dict(method=name, bits=b, slope=slope,
                             bias=float(np.mean(est - true)),
                             rmse=float(np.sqrt(np.mean((est - true) ** 2)))))
    return rows


# --------------------------------------------------------------------------------------
# Claim 3 (application): a KV cache compressed to ~3 bits keeps attention nearly intact,
#          even when keys have the large outlier channels real LLMs exhibit, without any
#          calibration. Compared against the standard per-token min/max integer quantizer.
# --------------------------------------------------------------------------------------
def synthetic_kv(n_tok, d, n_q, rng):
    """Keys with a handful of persistent high-magnitude channels (as observed in real
    LLM key caches, cf. KIVI / KVQuant), varying per-token norms, and queries that
    attend sharply to a few positions."""
    K = rng.standard_normal((n_tok, d))
    outlier_ch = rng.choice(d, size=4, replace=False)
    K[:, outlier_ch] = K[:, outlier_ch] * 4 + rng.choice([-10, 10], size=4)
    K *= rng.lognormal(0, 0.3, size=(n_tok, 1))
    V = rng.standard_normal((n_tok, d)) * rng.lognormal(0, 0.3, size=(n_tok, 1))
    targets = rng.integers(0, n_tok, size=n_q)
    Q = 1.5 * K[targets] + rng.standard_normal((n_q, d)) * 2.0
    return Q, K, V


def attention(Q, K, V):
    logits = Q @ K.T / math.sqrt(Q.shape[1])
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P, P @ V


def claim3_attention(n_tok, d, n_q, rng, n_seeds=5):
    """Averaged over `n_seeds` random rotations (TurboQuant's only randomness) since a
    single draw is noisy at this scale."""
    Q, K, V = synthetic_kv(n_tok, d, n_q, rng)
    P, O = attention(Q, K, V)
    top_true = np.argsort(-P, axis=1)[:, :8]
    rows = []
    for b in [2, 3, 4]:
        configs = {
            "Uniform min/max (per-token)": lambda s: (UniformMinMax(d, b), UniformMinMax(d, b)),
            "TurboQuant-mse K+V": lambda s: (TurboQuantMSE(d, b, seed=s), TurboQuantMSE(d, b, seed=100 + s)),
            "TurboQuant-prod K, mse V": lambda s: (TurboQuantProd(d, b, seed=s), TurboQuantMSE(d, b, seed=100 + s)),
        }
        for name, make in configs.items():
            seeds = [0] if name.startswith("Uniform") else range(n_seeds)
            m = {"kl": [], "top8": [], "out_cos": []}
            for s in seeds:
                qk, qv = make(s)
                Ph, Oh = attention(Q, qk(K), qv(V))
                m["kl"].append(np.mean(np.sum(P * (np.log(P + 1e-30) - np.log(Ph + 1e-30)), axis=1)))
                top_hat = np.argsort(-Ph, axis=1)[:, :8]
                m["top8"].append(np.mean([len(set(a) & set(h)) / 8 for a, h in zip(top_true, top_hat)]))
                m["out_cos"].append(np.mean(np.sum(O * Oh, 1) /
                                            (np.linalg.norm(O, axis=1) * np.linalg.norm(Oh, axis=1))))
            bits_k = qk.bits_per_coord()
            rows.append(dict(method=name, bits=b, bits_per_coord=bits_k, compression=16 / bits_k,
                             kl=float(np.mean(m["kl"])), kl_std=float(np.std(m["kl"])),
                             top8=float(np.mean(m["top8"])), out_cos=float(np.mean(m["out_cos"])),
                             seeds=len(seeds)))
    return rows


# --------------------------------------------------------------------------------------
# Practical cost: quantization throughput on CPU (vectors/s) - it is "online" and cheap.
# --------------------------------------------------------------------------------------
def throughput(n, d, rng):
    X = rng.standard_normal((n, d))
    rows = []
    for name, q in [("Uniform min/max", UniformMinMax(d, 3)),
                    ("TurboQuant-mse 3b", TurboQuantMSE(d, 3)),
                    ("TurboQuant-prod 3b", TurboQuantProd(d, 3))]:
        q.quantize(X[:100])  # warm-up
        t = time.perf_counter()
        q.quantize(X)
        dt = time.perf_counter() - t
        rows.append(dict(method=name, vec_per_s=n / dt))
    return rows


# --------------------------------------------------------------------------------------
def md_table(rows, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n"
                   for r in rows)
    return head + body


def maybe_plot(c1, c3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    d0 = c1[0]["d"]
    sub = [r for r in c1 if r["d"] == d0]
    bits = [r["bits"] for r in sub]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(bits, [r["upper"] for r in sub], "--", color="#999", label="paper upper bound")
    ax.semilogy(bits, [r["lower"] for r in sub], ":", color="#999", label="lower bound (any quantizer)")
    ax.semilogy(bits, [r["mse"] for r in sub], "o-", color="#2a6fdb", label="TurboQuant measured")
    ax.semilogy(bits, [r["paper"] for r in sub], "x", color="#d9534f", ms=9, label="paper reported")
    ax.set_xlabel("bits per coordinate"); ax.set_ylabel("MSE (unit vectors)")
    ax.set_xticks(bits); ax.set_title(f"Claim 1: distortion vs bounds (d={d0})")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "claim1_mse.png", dpi=150); files.append("claim1_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, c in [("Uniform min/max (per-token)", "#999"), ("TurboQuant-mse K+V", "#2a6fdb"),
                    ("TurboQuant-prod K, mse V", "#1a9e5c")]:
        rs = [r for r in c3 if r["method"] == name]
        ax.plot([r["bits"] for r in rs], [r["top8"] for r in rs], "o-", color=c, label=name)
    ax.set_xlabel("bits per coordinate"); ax.set_ylabel("top-8 attention recall")
    ax.set_xticks([2, 3, 4]); ax.set_ylim(0, 1.02)
    ax.set_title("Claim 3: attention fidelity, outlier-heavy KV cache")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "claim3_attention.png", dpi=150); files.append("claim3_attention.png")
    plt.close(fig)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = 4_000 if a.quick else 20_000
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    c1 = claim1_mse(n, [128] if a.quick else [128, 1024], rng)
    c2 = claim2_bias(n, 128, rng)
    c3 = claim3_attention(1024 if a.quick else 4096, 128, 64 if a.quick else 256, rng)
    tp = throughput(n, 128, rng)
    plots = maybe_plot(c1, c3)

    f3, f4 = "{:.3f}", "{:.4f}"
    md = "# Experiment 01 results\n\n"
    md += f"_seed={a.seed}, n={n}, quick={a.quick}, runtime={time.perf_counter() - t0:.1f}s_\n\n"
    md += "## Claim 1: MSE distortion is within ~2.7x of the information-theoretic optimum\n\n"
    md += md_table(c1, ["d", "bits", "mse", "paper", "lower", "upper", "ratio"],
                   {"mse": f4, "lower": f4, "upper": f4, "paper": "{}", "ratio": "{:.2f}x"})
    md += "\n## Claim 2: MSE quantizer is biased for inner products; prod variant is unbiased\n\n"
    md += "slope = 1.0 means unbiased. Theory says MSE at 1 bit shrinks inner products by 2/pi = 0.637.\n\n"
    md += md_table(c2, ["method", "bits", "slope", "bias", "rmse"],
                   {"slope": f3, "bias": "{:+.4f}", "rmse": f4})
    md += "\n## Claim 3: attention on a compressed, outlier-heavy KV cache\n\n"
    md += "fp16 baseline = 16 bits/coord. bits_per_coord includes per-vector norms/scales. "
    md += "kl = KL(true attention || quantized attention), mean (and std) over 5 random rotations; "
    md += "top8 = overlap of the 8 most-attended tokens; out_cos = cosine of attention outputs.\n\n"
    md += md_table(c3, ["method", "bits", "bits_per_coord", "compression", "kl", "kl_std", "top8", "out_cos"],
                   {"bits_per_coord": "{:.2f}", "compression": "{:.1f}x", "kl": f4, "kl_std": f4,
                    "top8": f3, "out_cos": f4})
    md += "\n## Quantization throughput (single CPU process, d=128, 3 bits)\n\n"
    md += md_table(tp, ["method", "vec_per_s"], {"vec_per_s": "{:,.0f}"})
    for p in plots:
        md += f"\n![{p}]({p})\n"
    (OUT / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
