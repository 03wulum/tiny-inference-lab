"""
Experiment 04: a tiny "pointer" network that stores no facts, only how to connect them.

  python run.py            # full test sets (43k questions), ~3-5 min on a laptop CPU
  python run.py --quick    # 200 questions per hop, ~30 s (CI smoke run)
  python run.py --clean    # delete downloaded data (data/) and caches

Evaluates the weights that train.py wrote to weights/ (training is GPU-only, see train.py).
Downloads MetaQA (~68 MB, CC licensed) into data/. Prints the PASS/FAIL table and writes
results/verdict.json and results/details.md (results/quick/ for --quick).
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
from criteria import INTRIGUING, THRESH, evaluate
from router_np import NpParamQA, NpRouter, load
from store import PointerStore, fact_queries

HERE = Path(__file__).parent
W = HERE / "weights"
MAX_LEN = 32
EDIT_FRACTION = 0.10
EDIT_RELATIONS = ["directed_by", "written_by", "starred_actors", "release_year", "in_language", "has_genre"]


# --------------------------------------------------------------------------- evaluation
def hits1_pointer(router, store, qs, vocab):
    if not qs:
        return 0.0
    paths = []
    for i in range(0, len(qs), 1024):
        paths += router.paths(vocab.batch(qs[i:i + 1024], MAX_LEN))
    ok = [store.answer(q["topic"], p) in set(q["answers"]) for q, p in zip(qs, paths)]
    return 100.0 * float(np.mean(ok))


def hits1_param(model, qs, vocab, gold_key="answers"):
    if not qs:
        return 0.0
    ok = []
    for i in range(0, len(qs), 512):
        b = qs[i:i + 512]
        pred = model.answer(vocab.batch(b, MAX_LEN), np.array([q["topic"] for q in b]))
        ok += [int(p) in set(q[gold_key]) for p, q in zip(pred, b)]
    return 100.0 * float(np.mean(ok))


# --------------------------------------------------------------------------- knowledge edits
def unique_gold_path(store, q, max_hop):
    """The single pointer path whose result equals the question's gold answer set (or None)."""
    gold = set(q["answers"])
    found, frontier = [], [((), np.array([q["topic"]], np.int32), np.ones(1, np.float32))]
    for _ in range(max_hop):
        nxt = []
        for path, ents, w in frontier:
            for d in range(D.N_DIR):
                e2, w2 = store.hop(ents, w, d)
                if len(e2):
                    nxt.append((path + (d,), e2, w2))
        frontier = nxt
    for path, ents, _ in frontier:
        if set(ents.tolist()) - {q["topic"]} == gold:
            found.append(path)
            if len(found) > 1:
                return None
    return found[0] if found else None


def edit_kb(kb, seed=0):
    """Rewire EDIT_FRACTION of movies: each of their facts in EDIT_RELATIONS now points at a random
    entity from the same relation's range (another director, another year, ...)."""
    rng = np.random.default_rng(seed)
    movies = np.unique(kb.h)
    edited = set(rng.choice(movies, int(EDIT_FRACTION * len(movies)), replace=False).tolist())
    t = kb.t.copy()
    for rname in EDIT_RELATIONS:
        r = D.RELATIONS.index(rname)
        m = (kb.r == r) & np.isin(kb.h, list(edited))
        t[m] = rng.choice(kb.t[kb.r == r], int(m.sum()))
    return PointerStore.from_kb(kb, t=t), len(edited)


def edit_test(kb, store, router, param, vocab, qs_by_hop, n_per_hop, seed=0):
    rng = np.random.default_rng(seed)
    cand = []
    for h in (1, 2):
        qs = qs_by_hop[h]
        for i in rng.permutation(len(qs))[:n_per_hop]:
            p = unique_gold_path(store, qs[i], h)
            if p is not None:
                cand.append(dict(qs[i], path=p))
    new_store, n_movies = edit_kb(kb, seed)
    changed = []
    for q in cand:
        ents, _ = new_store.follow(q["topic"], q["path"])
        new = sorted(set(ents.tolist()) - {q["topic"]})
        if new and set(new) != set(q["answers"]):
            changed.append(dict(q, new_answers=new))
    before = hits1_pointer(router, store, changed, vocab)
    after = hits1_pointer(router, new_store, [dict(q, answers=q["new_answers"]) for q in changed], vocab)
    p_after = hits1_param(param, changed, vocab, "new_answers") if param else float("nan")
    p_old = hits1_param(param, changed, vocab, "answers") if param else float("nan")
    return dict(n_candidates=len(cand), n_changed=len(changed), movies_edited=n_movies,
                pointer_before=before, pointer_after=after, ratio=after / max(before, 1e-9),
                param_after_new_facts=p_after, param_still_old_facts=p_old)


# --------------------------------------------------------------------------- latency
def latency(router, store, param, qs, vocab, reps):
    rng = np.random.default_rng(0)
    out = {}
    # 1) one pointer dereference from Python (dict-style API call: key -> slice of the flat array)
    keys = rng.integers(0, store.n * D.N_DIR, 200_000)
    ents, dirs = (keys // D.N_DIR).tolist(), (keys % D.N_DIR).tolist()
    t = time.perf_counter()
    for e, d in zip(ents, dirs):
        store.neighbours(e, d)
    out["deref_python_ns"] = (time.perf_counter() - t) / len(keys) * 1e9
    # 2) the same lookup vectorised over 1M keys: the raw memory cost per pointer
    keys = rng.integers(0, store.n * D.N_DIR, 1_000_000)
    t = time.perf_counter()
    first = store.indices[np.minimum(store.indptr[keys], len(store.indices) - 1)]
    out["deref_vectorised_ns"] = (time.perf_counter() - t) / len(keys) * 1e9
    del first
    # 3) end to end, one question at a time: tokenize -> router -> follow pointers -> answer
    sample = [qs[i] for i in rng.permutation(len(qs))[:reps]]
    times = []
    for q in sample:
        t = time.perf_counter()
        ids = np.array([vocab.encode(D.tokenize(q["q"]), MAX_LEN)])
        store.answer(q["topic"], router.paths(ids)[0])
        times.append(time.perf_counter() - t)
    out["e2e_cpu_batch1_us_median"] = statistics.median(times) * 1e6
    out["e2e_cpu_batch1_us_p90"] = float(np.percentile(times, 90)) * 1e6
    t = time.perf_counter()
    router.paths(vocab.batch(sample, MAX_LEN))
    out["router_cpu_batched_us_per_q"] = (time.perf_counter() - t) / len(sample) * 1e6
    if param is not None:
        times = []
        for q in sample[: max(20, reps // 5)]:
            t = time.perf_counter()
            ids = np.array([vocab.encode(D.tokenize(q["q"]), MAX_LEN)])
            param.answer(ids, np.array([q["topic"]]))
            times.append(time.perf_counter() - t)
        out["param_1x_cpu_batch1_us_median"] = statistics.median(times) * 1e6
    return out


# --------------------------------------------------------------------------- main
def clean():
    freed = 0
    for p in [HERE / "data", HERE / "__pycache__", HERE / ".pytest_cache", W / "smoke"]:
        if p.exists():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
    print(f"cleaned {freed / 1e6:.1f} MB")


def main():
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--weights", default=str(W), help="weights folder (default weights/)")
    a = ap.parse_args()
    if a.clean:
        return clean()
    W = Path(a.weights).resolve()
    out = HERE / "results" / ("quick" if a.quick or W != HERE / "weights" else "")
    out.mkdir(parents=True, exist_ok=True)
    logf = W / "train_log.json"
    if not logf.exists():
        v = dict(verdict="PENDING", passed=0, total=5, note="weights/ not trained yet: run train.py on a GPU")
        (out / "verdict.json").write_text(json.dumps(v, indent=1))
        print("No trained weights yet (weights/train_log.json missing). Verdict: PENDING")
        return
    log = json.loads(logf.read_text())
    vocab = D.Vocab(log["vocab"])

    t0 = time.time()
    root = D.download([f for f in D.FILES if "train" not in f and "dev" not in f])   # kb + test sets, ~11 MB
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    n = 200 if a.quick else None
    qs = {h: D.load_questions(kb, h, "test", root) for h in (1, 2, 3)}
    if n:
        rng = np.random.default_rng(0)
        qs = {h: [v[i] for i in rng.permutation(len(v))[:n]] for h, v in qs.items()}

    seeds = ["router_s0"] if a.quick else ["router_s0", "router_s1", "router_s2"]
    routers = {s: NpRouter(load(W / f"{s}.npz")) for s in seeds if (W / f"{s}.npz").exists()}
    slot = vocab.stoi[D.ENTITY_SLOT]
    params = {p: NpParamQA(load(W / f"{p}.npz"), slot) for p in ("param_1x", "param_4x") if (W / f"{p}.npz").exists()}

    acc = {}
    for s, r in routers.items():
        acc[s] = {h: hits1_pointer(r, store, qs[h], vocab) for h in (1, 2, 3)}
        print(f"  {s:<22}", acc[s])
    facts = fact_queries(store)
    facts = [facts[i] for i in np.random.default_rng(0).permutation(len(facts))[: 300 if a.quick else 3000]]
    fact_recall = {}
    for p, m in params.items():
        acc[p] = {h: hits1_param(m, qs[h], vocab) for h in (1, 2, 3)}
        fact_recall[p] = hits1_param(m, facts, vocab)
        print(f"  {p:<22}", acc[p], f"recall of training facts {fact_recall[p]:.1f}")
    r0 = routers["router_s0"]
    ctrl = {
        "random_pointers": {h: hits1_pointer(r0, store.permuted(0), qs[h], vocab) for h in (1, 2, 3)},
        "no_memory": {h: hits1_pointer(r0, PointerStore.empty(kb.n_entities), qs[h], vocab) for h in (1, 2, 3)},
    }
    if (W / "router_untrained.npz").exists():
        ctrl["untrained_router"] = {h: hits1_pointer(NpRouter(load(W / "router_untrained.npz")), store, qs[h], vocab)
                                    for h in (1, 2, 3)}
    explore = {}
    for name in ("router_12hop", "router_1pct"):
        if (W / f"{name}.npz").exists():
            explore[name] = {h: hits1_pointer(NpRouter(load(W / f"{name}.npz")), store, qs[h], vocab) for h in (1, 2, 3)}
    for k, v in {**ctrl, **explore}.items():
        print(f"  {k:<22}", v)

    edit = edit_test(kb, store, r0, params.get("param_1x"), vocab, qs, 100 if a.quick else 1500)
    print("  edit test", edit)
    lat = latency(r0, store, params.get("param_1x"), qs[2], vocab, 100 if a.quick else 1000)
    print("  latency", lat)
    gpu_lat = json.loads((HERE / "results/latency_gpu.json").read_text()) if (HERE / "results/latency_gpu.json").exists() else None

    kept_mb = (sum(f.stat().st_size for f in W.glob("*.npz")) + store.nbytes) / 1e6
    worst = {h: min(acc[s][h] for s in routers) for h in (1, 2, 3)}
    p1 = acc.get("param_1x", {1: 0, 2: 0, 3: 0})
    metrics = dict(
        c1_min_gap_pts=min(worst[h] - p1[h] for h in (2, 3)),
        c2_min_margin_pts=min(worst[h] - THRESH[h] for h in (1, 2, 3)),
        c3_e2e_us=lat["e2e_cpu_batch1_us_median"],
        c4_edit_ratio=edit["ratio"],
        c5_budget_ok=float(kept_mb <= 50 and log["total_train_seconds"] <= 600),
        n1_random_2hop=ctrl["random_pointers"][2],
    )
    v = evaluate(metrics)
    v.update(intriguing=INTRIGUING, quick=a.quick, accuracy=acc, param_fact_recall=fact_recall, worst_seed=worst, controls=ctrl, exploratory=explore, edit=edit,
             latency=lat, latency_gpu=gpu_lat, kept_mb=kept_mb, store_mb=store.nbytes / 1e6,
             train_seconds=log["total_train_seconds"], train_device=log["device"],
             params={k: r["params"] for k, r in log["runs"].items()}, param_dims=log["param_dims"],
             n_questions={h: len(qs[h]) for h in qs}, n_fact_probe=len(facts), eval_seconds=time.time() - t0)
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    (out / "details.md").write_text(details_md(v))
    print()
    print(f"{'id':<4} {'criterion':<62} {'value':>10} {'threshold':>12}  result")
    for c in v["criteria"] + [v["negative_control"]]:
        print(f"{c['id']:<4} {c['name'][:62]:<62} {c['value']:>10.2f} {c['op'] + ' ' + str(c['threshold']):>12}  {c['result']}")
    print(f"\nVERDICT: {v['verdict']} ({v['passed']}/{v['total']})  eval {v['eval_seconds']:.0f}s")


def details_md(v):
    L = ["# Experiment 04: all numbers", "",
         f"Questions per hop: {v['n_questions']}. Train device: {v['train_device']}, "
         f"total GPU training {v['train_seconds']:.0f}s. Eval took {v['eval_seconds']:.0f}s.", "",
         "## Hits@1 (%) on MetaQA test", "", "| model | 1-hop | 2-hop | 3-hop |", "|---|---|---|---|"]
    for k, a in {**v["accuracy"], **{f"control: {k}": x for k, x in v["controls"].items()},
                 **{f"exploratory: {k}": x for k, x in v["exploratory"].items()}}.items():
        L.append(f"| {k} | {a[1]:.1f} | {a[2]:.1f} | {a[3]:.1f} |")
    e = v["edit"]
    L += ["", "## Knowledge edit (no retraining)", "",
          f"{e['movies_edited']} movies rewired; {e['n_changed']} test questions whose answer changed "
          f"(of {e['n_candidates']} with a unique gold path).", "",
          f"- pointer model before edit: {e['pointer_before']:.1f}%, after edit (vs new answers): {e['pointer_after']:.1f}%",
          f"- no-memory model vs new answers: {e['param_after_new_facts']:.1f}% (still gives the old answer: {e['param_still_old_facts']:.1f}%)",
          "", "## Did the no-memory models learn the facts they were shown?", "",
          *[f"- {k}: Hits@1 on {v['n_fact_probe']} training facts (entity + pointer type -> any target): {x:.1f}%"
            for k, x in v["param_fact_recall"].items()],
          "", "## Latency (CPU, this machine)", ""]
    L += [f"- {k}: {x:,.1f}" for k, x in v["latency"].items()]
    if v.get("latency_gpu"):
        L += ["", "## Latency (GPU, from bench_gpu.py)", ""] + [f"- {k}: {x}" for k, x in v["latency_gpu"].items()]
    L += ["", "## Size", "", f"- kept weights + store: {v['kept_mb']:.1f} MB (store {v['store_mb']:.1f} MB)",
          f"- parameters: {v['params']}", f"- no-memory widths: {v['param_dims']}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
