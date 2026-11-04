"""
Experiment 02: does static 4-bit KV really cost ~8 HellaSwag points on SmolLM, and does
per-token adaptive precision win them back? (arXiv:2604.04722)

  python run.py                 # both models, 2,000 questions; ~10 min on a GPU
  python run.py --quick         # SmolLM-135M, 40 questions; ~1 min on a CPU (CI smoke run)
  python run.py --clean         # delete downloaded data (models live in the HF cache)

Downloads: HellaSwag validation (MIT, ~11 MB, into data/) and SmolLM-135M/360M
(Apache-2.0, ~1 GB total, HF cache). Nothing is trained.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

import kvq
from criteria import evaluate

HERE = Path(__file__).parent
DATA = HERE / "data"
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
MODELS = {"SmolLM-360M": "HuggingFaceTB/SmolLM-360M", "SmolLM-135M": "HuggingFaceTB/SmolLM-135M"}
TARGET_BITS = 3.8
METHODS = ["FP16", "static-8", "static-4", "static-2", f"adaptive-{TARGET_BITS}",
           f"random-{TARGET_BITS}", "turboquant-3"]


# ---- data: HellaSwag, preprocessed exactly like lm-evaluation-harness ------------------
def _pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(n):
    DATA.mkdir(exist_ok=True)
    f = DATA / "hellaswag_val.jsonl"
    if not f.exists():
        urllib.request.urlretrieve(HELLASWAG_URL, f)
    out = []
    for line in f.read_text().splitlines()[:n]:
        d = json.loads(line)
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        out.append(dict(query=_pre(d["activity_label"] + ": " + ctx),
                        choices=[_pre(e) for e in d["endings"]], label=int(d["label"])))
    return out


# ---- scoring ---------------------------------------------------------------------------
def encode_batch(tok, examples, device):
    """One row per (example, choice): context tokens + continuation tokens, right-padded."""
    seqs, spans = [], []
    for ex in examples:
        ctx = tok(ex["query"], add_special_tokens=False).input_ids
        for c in ex["choices"]:
            cont = tok(" " + c, add_special_tokens=False).input_ids
            seqs.append(ctx + cont)
            spans.append((len(ctx), len(ctx) + len(cont), len(c)))
    L = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device), spans


def continuation_logprob(logits, ids, spans):
    lp = logits.float().log_softmax(-1)
    out = []
    for i, (a, b, nchar) in enumerate(spans):
        tgt = ids[i, a:b]
        out.append((float(lp[i, a - 1: b - 1].gather(-1, tgt[:, None]).sum()), nchar))
    return out


def evaluate_model(name, n, batch, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=dtype).to(device).eval()
    kvq.install(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    turbo = kvq.TurboQuantTorch(head_dim, 3, device=device)
    data = load_hellaswag(n)
    correct = {m: [] for m in METHODS}
    correct_norm = {m: [] for m in METHODS}
    bits_used = {m: [] for m in METHODS}
    t0 = time.perf_counter()
    for s in range(0, len(data), batch):
        exs = data[s: s + batch]
        ids, mask, spans = encode_batch(tok, exs, device)
        labels = [e["label"] for e in exs]
        with torch.no_grad():
            kvq.KVPlan.mode, kvq.KVPlan.capture_attn = "fp", True
            fp_logits = model(input_ids=ids, attention_mask=mask).logits
            kvq.KVPlan.capture_attn = False
            feats = kvq.token_features(fp_logits, ids, kvq.KVPlan.captured, mask)
            plans = {
                "FP16": None,
                "static-8": kvq.static_bits(mask, 8),
                "static-4": kvq.static_bits(mask, 4),
                "static-2": kvq.static_bits(mask, 2),
                f"adaptive-{TARGET_BITS}": kvq.adaptive_bits(feats, mask, TARGET_BITS),
                f"random-{TARGET_BITS}": kvq.random_bits(mask, TARGET_BITS, seed=s),
                "turboquant-3": "turbo",
            }
            for m, plan in plans.items():
                if plan is None:
                    logits, b = fp_logits, 16.0
                elif isinstance(plan, str):
                    kvq.KVPlan.mode, kvq.KVPlan.turbo = "turbo", turbo
                    logits, b = model(input_ids=ids, attention_mask=mask).logits, 3.0
                else:
                    kvq.KVPlan.mode, kvq.KVPlan.bits = "bits", plan
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    b = kvq.mean_bits(plan, mask)
                kvq.KVPlan.mode = "fp"
                lp = continuation_logprob(logits, ids, spans)
                for j, lab in enumerate(labels):
                    sc = lp[4 * j: 4 * j + 4]
                    correct[m].append(int(np.argmax([x for x, _ in sc]) == lab))
                    correct_norm[m].append(int(np.argmax([x / c for x, c in sc]) == lab))
                bits_used[m].append(b)
    del model
    secs = time.perf_counter() - t0
    return dict(correct=correct, correct_norm=correct_norm,
                bits={m: float(np.mean(v)) for m, v in bits_used.items()}, seconds=secs)


# ---- statistics -----------------------------------------------------------------------
def paired_diff(a, b, n_boot=2000, seed=0):
    """Accuracy difference (a - b) in points with a paired bootstrap 95% CI."""
    a, b = np.array(a), np.array(b)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


def md_table(rows, cols):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return out + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    if a.clean:
        shutil.rmtree(DATA, ignore_errors=True)
        print("removed data/ (models stay in the shared HF cache: `huggingface-cli delete-cache`)")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    n = a.n or (40 if a.quick else 2000)
    models = ["SmolLM-135M"] if a.quick else list(MODELS)
    out = HERE / "results" / ("quick" if a.quick else "")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    res = {m: evaluate_model(m, n, a.batch if device == "cuda" else 4, device, dtype) for m in models}

    rows, per_model = [], {}
    for m, r in res.items():
        fp = r["correct"]["FP16"]
        pm = {}
        for meth in METHODS:
            acc = 100 * np.mean(r["correct"][meth])
            d, lo, hi = paired_diff(r["correct"][meth], fp)
            pm[meth] = dict(acc=acc, diff=d)
            rows.append(dict(model=m, method=meth, avg_bits=f"{r['bits'][meth]:.2f}", acc=f"{acc:.2f}",
                             acc_norm=f"{100 * np.mean(r['correct_norm'][meth]):.2f}",
                             vs_FP16=f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"))
        g, glo, ghi = paired_diff(r["correct"][f"adaptive-{TARGET_BITS}"], r["correct"]["static-4"])
        pm["gain"] = (g, glo, ghi)
        per_model[m] = pm

    ad = f"adaptive-{TARGET_BITS}"
    metrics = dict(
        c1_static4_drop=min(-p["static-4"]["diff"] for p in per_model.values()),
        c2_adaptive_gap=max(abs(p[ad]["diff"]) for p in per_model.values()),
        c3_adaptive_gain=min(p["gain"][0] for p in per_model.values()),
        n1_static2_gap=max(abs(p["static-2"]["diff"]) for p in per_model.values()),
    )
    v = evaluate(metrics)
    runtime = time.perf_counter() - t0

    print(f"\nExperiment 02 · Don't Waste Bits  ({'quick' if a.quick else 'full'}, n={n}, {device}, {runtime:.0f}s)\n")
    print(md_table(rows, ["model", "method", "avg_bits", "acc", "acc_norm", "vs_FP16"]))
    for r in v["criteria"] + [v["negative_control"]]:
        tag = r["result"] + (" (expected FAIL ✓)" if r.get("expect") and r["as_expected"] else
                             " (expected FAIL ✗)" if r.get("expect") else "")
        print(f"{r['id']:<4}{r['name']:<66}{r['value']:>8.2f}  {r['op']} {r['threshold']:<5} {tag}")
    print(f"\nVERDICT: {v['verdict']}  ({v['passed']}/{v['total']} criteria passed)\n")

    (out / "verdict.json").write_text(json.dumps(
        {"experiment": "02-adaptive-kv-bits", "paper": "arXiv:2604.04722", "quick": a.quick, "n": n,
         "device": device, "runtime_s": round(runtime, 1), **v, "metrics": metrics,
         "adaptive_vs_static4": {m: p["gain"] for m, p in per_model.items()}}, indent=2, default=float))
    md = (f"# Experiment 02 · full results\n\n_{'quick' if a.quick else 'full'} run, n={n} HellaSwag "
          f"validation questions per model, {device}, {runtime:.0f}s. Verdict **{v['verdict']}** "
          f"({v['passed']}/{v['total']}). Generated by `run.py`; do not edit._\n\n"
          "acc = lm-eval `acc`; vs_FP16 = paired difference in points with bootstrap 95% CI. "
          "avg_bits = payload bits per cached value, as in the paper. Int methods also store an fp16 "
          "scale + zero point per 32-channel group (+1 bit/value); TurboQuant stores one fp16 norm "
          "per head vector (+0.25 bit/value at head_dim 64).\n\n")
    md += md_table(rows, ["model", "method", "avg_bits", "acc", "acc_norm", "vs_FP16"])
    md += "\n## Adaptive vs static 4-bit (paired, points)\n\n"
    for m, p in per_model.items():
        g, lo, hi = p["gain"]
        md += f"- {m}: {g:+.2f} [{lo:+.2f}, {hi:+.2f}]\n"
    md += "\n## Criteria\n\n" + md_table(
        [{**r, "value": f"{r['value']:.2f}"} for r in v["criteria"] + [v["negative_control"]]],
        ["id", "name", "value", "op", "threshold", "result", "source"])
    (out / "details.md").write_text(md)


if __name__ == "__main__":
    main()
