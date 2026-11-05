"""
Train every model for experiment 05 on the GPU: one process, ~6 minutes total, hard cap 10.

  python train.py            # needs CUDA; refuses to train on CPU
  python train.py --smoke    # 3 steps per model on a tiny slice: a code check that runs anywhere

Training data for every model: MetaQA train (1-3 hops) + 20k compositional questions per hop for
1-3 hops (synth.py, topics with id % 10 != 0). The no-memory rivals also see every fact in the store.
Nobody trains on 4- or 5-hop questions. Writes fp16 weights to weights/ and weights/train_log.json.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import synth
from models_torch import BowRouter, KGEQA, ParamQA, RecRouter, Router, n_params, soft_follow, to_numpy_dict
from store import PointerStore, fact_queries

HERE = Path(__file__).parent
WEIGHTS = HERE / "weights"
MAX_LEN = 48
MAX_ANS = 256
N_SYNTH = 20000

# name, kind, data fraction, seed, seconds of GPU training
RUNS = [
    ("ptr_s0", "router", 1.0, 0, 35),       # fixed-hop router + loop fix (main flavor)
    ("ptr_s1", "router", 1.0, 1, 35),
    ("ptr_s2", "router", 1.0, 2, 35),
    ("rec_s0", "rec", 1.0, 0, 30),          # recurrent router (can run past 3 hops)
    ("rec_s1", "rec", 1.0, 1, 30),
    ("rec_s2", "rec", 1.0, 2, 30),
    ("bow", "bow", 1.0, 0, 20),             # order-blind router
    ("tiny", "tiny", 1.0, 0, 20),           # d=32, 1 layer
    ("ptr_0.1pct", "router", 0.001, 0, 15),  # exploratory: 0.1% of the questions
    ("mem_1x", "param", 1.0, 0, 40),        # no memory, same bytes as router + store
    ("mem_4x", "param", 1.0, 0, 40),        # no memory, 4x bytes
    ("kge_1x", "kge", 1.0, 0, 40),          # no memory, ComplEx KG embeddings (EmbedKGQA-style), same bytes
]
TOTAL_CAP_S = 600


def free_ram_gb():
    """Free physical RAM in GiB (Windows or Linux), None if unknown. No extra packages."""
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("tot", "avail", "tpf", "apf", "tv", "av", "aev")]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.avail / 2**30
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return None


def peak_rss_gb():
    """Peak resident memory of this process in GiB, None if unknown."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("pfc", ctypes.c_ulong)] + \
                           [(n, ctypes.c_size_t) for n in ("peak_ws", "ws", "qpp", "qp", "qpnp", "qnp", "pf", "ppf")]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.peak_ws / 2**30
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        return None


def pack(qs, vocab):
    """Questions -> compact arrays: token ids [N, MAX_LEN] int32, topic int32, answers as CSR (flat + offsets).
    Answers are capped at MAX_ANS per question, as before. Keeps host RAM to tens of MB."""
    ids = np.zeros((len(qs), MAX_LEN), dtype=np.int32)
    topic = np.empty(len(qs), dtype=np.int32)
    off = np.zeros(len(qs) + 1, dtype=np.int64)
    for i, q in enumerate(qs):
        ids[i] = vocab.encode(q["tokens"], MAX_LEN)
        topic[i] = q["topic"]
        off[i + 1] = off[i] + min(len(q["answers"]), MAX_ANS)
    flat = np.empty(int(off[-1]), dtype=np.int32)
    for i, q in enumerate(qs):
        flat[off[i]:off[i + 1]] = q["answers"][:MAX_ANS]
    return dict(ids=ids, topic=topic, off=off, flat=flat)


def to_device(p, dev):
    return dict(ids=torch.from_numpy(p["ids"]).to(dev).long(), topic=torch.from_numpy(p["topic"]).to(dev).long(),
                off=torch.from_numpy(p["off"]).to(dev), flat=torch.from_numpy(p["flat"]).to(dev).long())


def multi_hot(g, idx, n):
    """Multi-hot answer targets for rows idx, built on the GPU from the CSR answer table."""
    start, lens = g["off"][idx], g["off"][idx + 1] - g["off"][idx]
    row = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), lens)
    first = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
    col = g["flat"][torch.repeat_interleave(start, lens) + torch.arange(len(row), device=idx.device) - first]
    t = torch.zeros(len(idx), n, device=idx.device)
    t[row, col] = 1.0
    return t


def pick_dim(cls, target_bytes, n_ent, vocab, **kw):
    d = 4
    while True:
        with torch.device("meta"):
            m = cls(vocab, n_ent, d=d, **kw) if cls is ParamQA else cls(vocab, n_ent, k=d, **kw)
        if 4 * n_params(m) >= target_bytes:
            return d
        d += 4


def build(kind, name, vocab, n_ent, slot, dims):
    V = len(vocab)
    if kind == "router":
        return Router(V, max_len=MAX_LEN)
    if kind == "tiny":
        return Router(V, d=32, layers=1, heads=2, max_len=MAX_LEN)
    if kind == "bow":
        return BowRouter(V)
    if kind == "rec":
        return RecRouter(V, slot, d=112)
    if kind == "param":
        return ParamQA(V, n_ent, slot, d=dims[name], max_len=MAX_LEN)
    return KGEQA(V, n_ent, k=dims[name], max_len=MAX_LEN)


def train_one(name, kind, model, g, rows, store_t, n_ent, seconds, dev, smoke):
    pointer = kind in ("router", "tiny", "bow", "rec")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if pointer else 3e-3, weight_decay=0.0)
    N = len(rows)
    bs = 16 if smoke else (512 if pointer else 1024)
    t0, step, seen, losses = time.time(), 0, 0, []
    gen = torch.Generator(device="cpu").manual_seed(0)
    done = False
    while not done:
        perm = rows[torch.randperm(N, generator=gen).to(dev)]
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            b_ids, b_top = g["ids"][idx], g["topic"][idx]
            target = multi_hot(g, idx, n_ent)
            if pointer:
                rel_p, hop_p = model(b_ids, 3)
                score = soft_follow(b_top, rel_p, hop_p, *store_t, n_ent, avoid_topic=True)
                score = score.scatter(1, b_top[:, None], 0.0)
                loss = F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6), target, reduction="sum") / len(idx)
            else:
                logits = model(b_ids, b_top).scatter(1, b_top[:, None], -1e9)
                lse_gold = torch.logsumexp(logits.masked_fill(target == 0, -1e9), -1)
                loss = (torch.logsumexp(logits, -1) - lse_gold).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            seen += len(idx)
            losses.append(loss.item())
            if (smoke and step >= 3) or time.time() - t0 > seconds:
                done = True
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    rss = peak_rss_gb()
    print(f"  {name:<11} steps={step:<6} examples={seen:<8} loss={np.mean(losses[-50:]):.4f} {secs:.0f}s"
          f"  peak_rss={rss if rss is None else round(rss, 2)} GiB", flush=True)
    return dict(steps=step, examples=seen, final_loss=float(np.mean(losses[-50:])), seconds=secs, epochs=seen / N)


def prepare(root, smoke):
    """Load everything, pack it into compact arrays, and drop the Python question objects.
    Host RAM peaks here (~0.5 GB of Python dicts for a few seconds) and falls back once packed."""
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    split, lim = ("dev", 64) if smoke else ("train", None)
    meta = [q for h in (1, 2, 3) for q in D.load_questions(kb, h, split, root, limit=lim)]
    syn = [q for h in (1, 2, 3) for q in synth.generate(kb, store, h, 64 if smoke else N_SYNTH, "train")]
    facts = fact_queries(store)[: 64 if smoke else None]
    allq = meta + syn + facts                       # rows [0, n_qa) are questions, the rest are facts
    vocab = D.Vocab.build(allq, extra=[D.ENTITY_SLOT])
    packed = pack(allq, vocab)
    counts = dict(n_metaqa_train=len(meta), n_synth_train=len(syn), n_facts=len(facts))
    del allq, meta, syn, facts
    gc.collect()
    return kb, store, vocab, packed, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="refuse to start (and stop between models) if free RAM drops below this")
    a = ap.parse_args()
    if not a.smoke and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Training is GPU-only (project rule). Use --smoke for a code check.")
    free = free_ram_gb()
    print(f"free RAM at start: {free if free is None else round(free, 1)} GiB")
    if not a.smoke and free is not None and free < a.min_free_gb:
        raise SystemExit(f"Only {free:.1f} GiB RAM free (< {a.min_free_gb}). Not starting. Try again when the PC is quieter.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = WEIGHTS / "smoke" if a.smoke else WEIGHTS
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        root = D.download()
    except OSError:
        if not a.smoke:
            raise
        root = D.DATA
    kb, store, vocab, packed, counts = prepare(root, a.smoke)   # before CUDA init: one RAM peak at a time
    slot = vocab.stoi[D.ENTITY_SLOT]
    n_ent = kb.n_entities
    n_qa = counts["n_metaqa_train"] + counts["n_synth_train"]
    print(f"packed {len(packed['ids'])} rows, {sum(x.nbytes for x in packed.values()) / 2**20:.0f} MiB; "
          f"peak_rss so far {peak_rss_gb()} GiB", flush=True)

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        gfree, gtot = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name()} free {gfree / 2**30:.1f} / {gtot / 2**30:.1f} GiB")
    g = to_device(packed, dev)
    del packed
    store_t = tuple(torch.tensor(x.astype(np.int64), device=dev) for x in (store.e_head, store.e_dir, store.e_tail))

    with torch.device("meta"):
        router_bytes = 4 * n_params(Router(len(vocab), max_len=MAX_LEN))
    pointer_bytes = router_bytes + store.nbytes
    dims = {"mem_1x": pick_dim(ParamQA, pointer_bytes, n_ent, len(vocab), slot_id=slot, max_len=MAX_LEN),
            "mem_4x": pick_dim(ParamQA, 4 * pointer_bytes, n_ent, len(vocab), slot_id=slot, max_len=MAX_LEN),
            "kge_1x": pick_dim(KGEQA, pointer_bytes, n_ent, len(vocab), max_len=MAX_LEN)}
    log = dict(device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", torch=torch.__version__,
               vocab=vocab.itos, n_entities=n_ent, store_bytes=store.nbytes, router_bytes_fp32=router_bytes,
               pointer_system_bytes_fp32=pointer_bytes, dims=dims, **counts, runs={}, smoke=a.smoke)
    torch.manual_seed(0)
    np.savez_compressed(out_dir / "ptr_untrained.npz", **to_numpy_dict(Router(len(vocab), max_len=MAX_LEN)))

    only = set(filter(None, a.only.split(",")))
    t_all = time.time()
    for name, kind, frac, seed, secs in RUNS:
        if only and name not in only:
            continue
        if time.time() - t_all > TOTAL_CAP_S:
            print(f"  {name}: skipped, total time cap reached")
            continue
        free = free_ram_gb()
        if not a.smoke and free is not None and free < 1.5:
            print(f"  {name}: skipped, only {free:.1f} GiB RAM free")
            continue
        torch.manual_seed(seed)
        rows = np.arange(n_qa)                      # pointer flavors: questions only
        if frac < 1:
            rng = np.random.default_rng(seed)
            rows = rng.choice(n_qa, max(8, int(frac * n_qa)), replace=False)
        if kind in ("param", "kge"):
            rows = np.arange(len(g["ids"]))         # no-memory rivals also see every fact
        model = build(kind, name, vocab, n_ent, slot, dims).to(dev)
        stats = train_one(name, kind, model, g, torch.from_numpy(rows).to(dev), store_t, n_ent, secs, dev, a.smoke)
        stats.update(kind=kind, fraction=frac, seed=seed, n_train=len(rows), params=n_params(model), budget_s=secs)
        log["runs"][name] = stats
        np.savez_compressed(out_dir / f"{name}.npz", **to_numpy_dict(model))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log["total_train_seconds"] = sum(r["seconds"] for r in log["runs"].values())
    log["wall_seconds"] = time.time() - t_all
    log["peak_rss_gib"] = peak_rss_gb()
    if dev.type == "cuda":
        log["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"total training {log['total_train_seconds']:.0f}s, peak RAM {log['peak_rss_gib']} GiB -> {out_dir}")


if __name__ == "__main__":
    main()
