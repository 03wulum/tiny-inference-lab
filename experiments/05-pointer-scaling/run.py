"""
Experiment 05: pointer memory at 1-5 hops vs every no-memory rival type, with the loop fix built in.

  python run.py            # full test sets, ~4-6 min on a laptop CPU
  python run.py --quick    # 150-200 questions per level, ~1 min (CI smoke run)
  python run.py --clean    # delete data/ and caches
  python run.py --weights weights/smoke   # evaluate other weights (results go to results/quick/)

Evaluates weights/ written by train.py (GPU-only). Prints the PASS/FAIL table, the level x model
table, and which router flavor beats which rival at each level. Writes results/verdict.json and
results/details.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

import data as D
import synth
from criteria import INTRIGUING, THRESH, evaluate
from models_np import NpBowRouter, NpKGEQA, NpParamQA, NpRecRouter, NpRouter, load
from store import PointerStore

HERE = Path(__file__).parent
W = HERE / "weights"
MAX_LEN = 48
SEEDS = ("s0", "s1", "s2")
# simplest first: used to answer "which flavor is enough to beat this rival at this level?"
FLAVOR_ORDER = ["tiny", "bow", "ptr_plainwalk", "ptr", "rec"]
RIVALS = ["mem_1x", "mem_4x", "kge_1x"]


def levels(qs_meta, qs_syn):
    return {**{f"MetaQA {h}": v for h, v in qs_meta.items()}, **{f"Compositional {h}": v for h, v in qs_syn.items()}}


def hits_router(router, store, qs, vocab, T, avoid=True):
    if not qs:
        return 0.0
    paths = []
    for i in range(0, len(qs), 1024):
        paths += router.paths(vocab.batch(qs[i:i + 1024], MAX_LEN), T)
    ok = [store.answer(q["topic"], p, avoid_topic=avoid) in set(q["answers"]) for q, p in zip(qs, paths)]
    return 100.0 * float(np.mean(ok))


def hits_rival(model, qs, vocab):
    if not qs:
        return 0.0
    ok = []
    for i in range(0, len(qs), 256):
        b = qs[i:i + 256]
        pred = model.answer(vocab.batch(b, MAX_LEN), np.array([q["topic"] for q in b]))
        ok += [int(p) in set(q["answers"]) for p, q in zip(pred, b)]
    return 100.0 * float(np.mean(ok))


def latency(router, store, qs, vocab, reps):
    rng = np.random.default_rng(0)
    sample = [qs[i] for i in rng.permutation(len(qs))[:reps]]
    ts = []
    for q in sample:
        t = time.perf_counter()
        ids = np.array([vocab.encode(D.tokenize(q["q"]), MAX_LEN)])
        store.answer(q["topic"], router.paths(ids)[0], avoid_topic=True)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6, float(np.percentile(ts, 90)) * 1e6


def clean():
    freed = 0
    for p in [HERE / "data", HERE / "__pycache__", HERE / ".pytest_cache", W / "smoke", HERE / "results/quick"]:
        if p.exists():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
    print(f"cleaned {freed / 1e6:.1f} MB")


def main():
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--weights", default=str(W))
    a = ap.parse_args()
    if a.clean:
        return clean()
    W = Path(a.weights).resolve()
    out = HERE / "results" / ("quick" if a.quick or W != (HERE / "weights").resolve() else "")
    out.mkdir(parents=True, exist_ok=True)
    if not (W / "train_log.json").exists():
        (out / "verdict.json").write_text(json.dumps(dict(verdict="PENDING", passed=0, total=5), indent=1))
        print("No trained weights yet (weights/train_log.json missing). Verdict: PENDING")
        return
    log = json.loads((W / "train_log.json").read_text())
    vocab = D.Vocab(log["vocab"])
    slot = vocab.stoi[D.ENTITY_SLOT]
    t0 = time.time()

    root = D.download([f for f in D.FILES if "test" in f or "kb" in f])
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    rng = np.random.default_rng(0)
    meta = {h: D.load_questions(kb, h, "test", root) for h in (1, 2, 3)}
    if a.quick:
        meta = {h: [v[i] for i in rng.permutation(len(v))[:200]] for h, v in meta.items()}
    syn = {h: synth.generate(kb, store, h, 150 if a.quick else 1500, "test") for h in (1, 2, 3, 4, 5)}
    L = levels(meta, syn)

    def ld(name):
        return load(W / f"{name}.npz") if (W / f"{name}.npz").exists() else None

    seeds = SEEDS[:1] if a.quick else SEEDS
    acc, flavors = {}, {}
    for s in seeds:
        if ld(f"ptr_{s}") is not None:
            flavors[f"ptr_{s}"] = (NpRouter(ld(f"ptr_{s}")), 3, True)
        if ld(f"rec_{s}") is not None:
            flavors[f"rec_{s}"] = (NpRecRouter(ld(f"rec_{s}"), slot), 5, True)
    flavors["ptr_plainwalk"] = (flavors["ptr_s0"][0], 3, False)        # same weights, no loop fix at inference
    for name in ("bow", "tiny", "ptr_0.1pct", "ptr_untrained"):
        Wt = ld(name)
        if Wt is not None:
            flavors[name] = ((NpBowRouter(Wt) if name == "bow" else NpRouter(Wt)), 3, True)
    for name, (r, T, avoid) in flavors.items():
        acc[name] = {lv: hits_router(r, store, qs, vocab, T, avoid) for lv, qs in L.items()}
        print(f"  {name:<14}", {k: round(v, 1) for k, v in acc[name].items()}, flush=True)
    for name in RIVALS:
        Wt = ld(name)
        if Wt is not None:
            m = NpKGEQA(Wt) if name.startswith("kge") else NpParamQA(Wt, slot)
            acc[name] = {lv: hits_rival(m, qs, vocab) for lv, qs in L.items()}
            print(f"  {name:<14}", {k: round(v, 1) for k, v in acc[name].items()}, flush=True)
    r0 = flavors["ptr_s0"][0]
    acc["scrambled"] = {lv: hits_router(r0, store.permuted(0), qs, vocab, 3) for lv, qs in L.items()}

    # worst-of-seeds rows for the two seeded flavors
    for fam in ("ptr", "rec"):
        rows = [acc[f"{fam}_{s}"] for s in seeds if f"{fam}_{s}" in acc]
        acc[fam] = {lv: min(r[lv] for r in rows) for lv in L}

    # which (simplest) flavor beats each rival by >= 10 pts at each level?
    beats = {lv: {rv: next((f for f in FLAVOR_ORDER if f in acc and acc[f][lv] >= acc[rv][lv] + 10), "none")
                  for rv in RIVALS if rv in acc} for lv in L}

    lat = {}
    lat["ptr_e2e_us_median"], lat["ptr_e2e_us_p90"] = latency(r0, store, meta[2], vocab, 100 if a.quick else 1000)
    if "tiny" in flavors:
        lat["tiny_e2e_us_median"], _ = latency(flavors["tiny"][0], store, meta[2], vocab, 100 if a.quick else 1000)
    print("  latency", lat)

    trained = [f"MetaQA {h}" for h in (1, 2, 3)] + [f"Compositional {h}" for h in (1, 2, 3)]
    kept_mb = (sum(f.stat().st_size for f in W.glob("*.npz")) + store.nbytes) / 1e6
    metrics = dict(
        c1_min_margin_pts=min(acc["ptr"][f"MetaQA {h}"] - THRESH[h] for h in (1, 2, 3)),
        c2_min_gap_pts=min(acc["ptr"][lv] - acc[rv][lv] for lv in trained for rv in RIVALS if rv in acc),
        c3_rec_min_45=min(acc["rec"]["Compositional 4"], acc["rec"]["Compositional 5"]),
        c4_e2e_us=lat["ptr_e2e_us_median"],
        c5_budget_ok=float(kept_mb <= 50 and log["total_train_seconds"] <= 600),
        n1_random_2hop=acc["scrambled"]["MetaQA 2"],
    )
    v = evaluate(metrics)
    v.update(intriguing=INTRIGUING, quick=a.quick, accuracy=acc, beats=beats, latency=lat, kept_mb=kept_mb,
             train_seconds=log["total_train_seconds"], train_device=log["device"], dims=log["dims"],
             params={k: r["params"] for k, r in log["runs"].items()},
             n_questions={lv: len(q) for lv, q in L.items()}, eval_seconds=time.time() - t0)
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    (out / "details.md").write_text(details_md(v, list(L)))
    print()
    print(f"{'id':<4} {'criterion':<66} {'value':>9} {'threshold':>11}  result")
    for c in v["criteria"] + [v["negative_control"]]:
        print(f"{c['id']:<4} {c['name'][:66]:<66} {c['value']:>9.2f} {c['op'] + ' ' + str(c['threshold']):>11}  {c['result']}")
    print(f"\nVERDICT: {v['verdict']} ({v['passed']}/{v['total']})  eval {v['eval_seconds']:.0f}s")


NAMES = {"ptr": "Pointer, fixed-hop + loop fix (worst of 3)", "rec": "Pointer, recurrent (worst of 3)",
         "ptr_plainwalk": "Pointer, fixed-hop, no loop fix", "bow": "Pointer, order-blind router",
         "tiny": "Pointer, tiny router (18k params)", "ptr_0.1pct": "Pointer, 0.1% of training data",
         "ptr_untrained": "Pointer, untrained router (control)", "scrambled": "Scrambled pointers (control)",
         "mem_1x": "No memory: transformer, same bytes", "mem_4x": "No memory: transformer, 4x bytes",
         "kge_1x": "No memory: ComplEx KG embeddings, same bytes"}


def details_md(v, lvs):
    hdr = "| model | " + " | ".join(lv.replace("Compositional", "Comp.") for lv in lvs) + " |"
    L = ["# Experiment 05: all numbers", "",
         f"Questions per level: {v['n_questions']}. Train: {v['train_device']}, {v['train_seconds']:.0f} s total. "
         f"Eval {v['eval_seconds']:.0f} s.", "", "## Hits@1 (%) by level", "", hdr,
         "|---" * (len(lvs) + 1) + "|"]
    order = ["ptr", "rec", "ptr_plainwalk", "bow", "tiny", "ptr_0.1pct", "mem_1x", "mem_4x", "kge_1x",
             "ptr_untrained", "scrambled"]
    for k in order + [k for k in v["accuracy"] if k not in order and not k.endswith(("_s0", "_s1", "_s2"))]:
        if k in v["accuracy"]:
            L.append(f"| {NAMES.get(k, k)} | " + " | ".join(f"{v['accuracy'][k][lv]:.1f}" for lv in lvs) + " |")
    L += ["", "Compositional 4 and 5 are never seen in training. Per-seed rows are in verdict.json.", "",
          "## Simplest pointer flavor that beats each rival by >= 10 pts", "",
          "| level | " + " | ".join(NAMES[r] for r in RIVALS if r in next(iter(v["beats"].values()))) + " |",
          "|---" * (1 + len(next(iter(v["beats"].values())))) + "|"]
    for lv, row in v["beats"].items():
        L.append(f"| {lv} | " + " | ".join(NAMES.get(f, f) for f in row.values()) + " |")
    L += ["", "Order tried: tiny -> order-blind -> fixed-hop without loop fix -> fixed-hop -> recurrent.", "",
          "## Latency (CPU, this machine)", ""] + [f"- {k}: {x:,.1f}" for k, x in v["latency"].items()]
    L += ["", "## Size", "", f"- kept weights + store: {v['kept_mb']:.1f} MB", f"- parameters: {v['params']}",
          f"- rival widths: {v['dims']}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
