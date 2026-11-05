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
import json
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


def tensorize(qs, vocab, dev):
    ids = torch.tensor(vocab.batch(qs, MAX_LEN), device=dev)
    topic = torch.tensor([q["topic"] for q in qs], device=dev)
    gold = np.full((len(qs), MAX_ANS), -1, dtype=np.int64)
    for i, q in enumerate(qs):
        a = q["answers"][:MAX_ANS]
        gold[i, : len(a)] = a
    return ids, topic, torch.tensor(gold, device=dev)


def multi_hot(gold, n):
    t = torch.zeros(gold.shape[0], n + 1, device=gold.device)
    t.scatter_(1, torch.where(gold < 0, n, gold), 1.0)
    return t[:, :n]


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


def train_one(name, kind, model, batches, store_t, n_ent, seconds, dev, smoke):
    pointer = kind in ("router", "tiny", "bow", "rec")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if pointer else 3e-3, weight_decay=0.0)
    ids, topic, gold = batches
    N = len(ids)
    bs = 16 if smoke else (512 if pointer else 1024)
    t0, step, seen, losses = time.time(), 0, 0, []
    gen = torch.Generator(device="cpu").manual_seed(0)
    done = False
    while not done:
        perm = torch.randperm(N, generator=gen).to(dev)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            b_ids, b_top, b_gold = ids[idx], topic[idx], gold[idx]
            target = multi_hot(b_gold, n_ent)
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
    print(f"  {name:<11} steps={step:<6} examples={seen:<8} loss={np.mean(losses[-50:]):.4f} {secs:.0f}s", flush=True)
    return dict(steps=step, examples=seen, final_loss=float(np.mean(losses[-50:])), seconds=secs, epochs=seen / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.smoke and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Training is GPU-only (project rule). Use --smoke for a code check.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = WEIGHTS / "smoke" if a.smoke else WEIGHTS
    out_dir.mkdir(parents=True, exist_ok=True)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name()} free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")

    try:
        root = D.download()
    except OSError:
        if not a.smoke:
            raise
        root = D.DATA
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    split, lim = ("dev", 64) if a.smoke else ("train", None)
    meta = [q for h in (1, 2, 3) for q in D.load_questions(kb, h, split, root, limit=lim)]
    n_syn = 64 if a.smoke else N_SYNTH
    syn = [q for h in (1, 2, 3) for q in synth.generate(kb, store, h, n_syn, "train")]
    facts = fact_queries(store)[: 64 if a.smoke else None]
    vocab = D.Vocab.build(meta + syn + facts, extra=[D.ENTITY_SLOT])
    slot = vocab.stoi[D.ENTITY_SLOT]
    n_ent = kb.n_entities
    store_t = tuple(torch.tensor(x.astype(np.int64), device=dev) for x in (store.e_head, store.e_dir, store.e_tail))

    with torch.device("meta"):
        router_bytes = 4 * n_params(Router(len(vocab), max_len=MAX_LEN))
    pointer_bytes = router_bytes + store.nbytes
    dims = {"mem_1x": pick_dim(ParamQA, pointer_bytes, n_ent, len(vocab), slot_id=slot, max_len=MAX_LEN),
            "mem_4x": pick_dim(ParamQA, 4 * pointer_bytes, n_ent, len(vocab), slot_id=slot, max_len=MAX_LEN),
            "kge_1x": pick_dim(KGEQA, pointer_bytes, n_ent, len(vocab), max_len=MAX_LEN)}
    log = dict(device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", torch=torch.__version__,
               vocab=vocab.itos, n_entities=n_ent, store_bytes=store.nbytes, router_bytes_fp32=router_bytes,
               pointer_system_bytes_fp32=pointer_bytes, dims=dims, n_metaqa_train=len(meta),
               n_synth_train=len(syn), n_facts=len(facts), runs={}, smoke=a.smoke)
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
        torch.manual_seed(seed)
        train = meta + syn
        if frac < 1:
            rng = np.random.default_rng(seed)
            train = [train[i] for i in rng.choice(len(train), max(8, int(frac * len(train))), replace=False)]
        if kind in ("param", "kge"):
            train = train + facts
        model = build(kind, name, vocab, n_ent, slot, dims).to(dev)
        stats = train_one(name, kind, model, tensorize(train, vocab, dev), store_t, n_ent, secs, dev, a.smoke)
        stats.update(kind=kind, fraction=frac, seed=seed, n_train=len(train), params=n_params(model), budget_s=secs)
        log["runs"][name] = stats
        np.savez_compressed(out_dir / f"{name}.npz", **to_numpy_dict(model))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log["total_train_seconds"] = sum(r["seconds"] for r in log["runs"].values())
    log["wall_seconds"] = time.time() - t_all
    if dev.type == "cuda":
        log["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"total training {log['total_train_seconds']:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
