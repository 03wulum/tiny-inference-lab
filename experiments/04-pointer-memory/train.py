"""
Train every model for experiment 04 on the GPU, one process, ~5 minutes total (Obi's budget).

  python train.py            # needs CUDA; refuses to train on CPU
  python train.py --smoke    # 3 steps per model on a tiny slice: a code check that runs anywhere

Writes fp16 weights to weights/*.npz (committed, ~15 MB total) and weights/train_log.json.
run.py evaluates those files; it never trains.
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
from model_torch import ParamQA, Router, n_params, soft_follow, to_numpy_dict
from store import PointerStore, fact_queries

HERE = Path(__file__).parent
WEIGHTS = HERE / "weights"
MAX_LEN = 32
MAX_ANS = 256

# name, kind, train hops, data fraction, seed, seconds of GPU training
RUNS = [
    ("router_s0", "router", (1, 2, 3), 1.0, 0, 40),
    ("router_s1", "router", (1, 2, 3), 1.0, 1, 40),
    ("router_s2", "router", (1, 2, 3), 1.0, 2, 40),
    ("router_12hop", "router", (1, 2), 1.0, 0, 25),     # exploratory: never sees a 3-hop question
    ("router_1pct", "router", (1, 2, 3), 0.01, 0, 25),  # exploratory: 1% of the training questions
    ("param_1x", "param", (1, 2, 3), 1.0, 0, 60),       # no memory, same bytes as router + store
    ("param_4x", "param", (1, 2, 3), 1.0, 0, 60),       # no memory, 4x the bytes
]
TOTAL_CAP_S = 600   # hard stop: Obi's 10-minute ceiling


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


def pick_param_dim(target_bytes, n_ent, vocab):
    """Smallest width (multiple of 4) whose fp32 weights take at least target_bytes."""
    d = 4
    while True:
        with torch.device("meta"):
            m = ParamQA(vocab, n_ent, 3, d=d, max_len=MAX_LEN)
        if 4 * n_params(m) >= target_bytes:
            return d
        d += 4


def train_one(name, kind, model, batches, store_t, n_ent, seconds, dev, smoke):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if kind == "router" else 3e-3, weight_decay=0.0)
    ids, topic, gold = batches
    N = len(ids)
    bs = 512 if kind == "router" else 1024
    if smoke:
        bs = 16
    t0, step, seen, losses = time.time(), 0, 0, []
    gen = torch.Generator(device="cpu").manual_seed(0)
    while True:
        perm = torch.randperm(N, generator=gen).to(dev)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            b_ids, b_top, b_gold = ids[idx], topic[idx], gold[idx]
            target = multi_hot(b_gold, n_ent)
            if kind == "router":
                rel_p, hop_p = model(b_ids)
                score = soft_follow(b_top, rel_p, hop_p, *store_t, n_ent)
                score = score.scatter(1, b_top[:, None], 0.0)              # the topic is never an answer
                loss = F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6), target, reduction="sum") / len(idx)
            else:
                logits = model(b_ids, b_top).scatter(1, b_top[:, None], -1e9)
                lse_gold = torch.logsumexp(logits.masked_fill(target == 0, -1e9), -1)
                loss = (torch.logsumexp(logits, -1) - lse_gold).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen += len(idx)
            losses.append(float(loss))
            if smoke and step >= 3:
                break
            if time.time() - t0 > seconds:
                break
        if (smoke and step >= 3) or time.time() - t0 > seconds:
            break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    print(f"  {name:<13} steps={step:<6} examples={seen:<8} loss={np.mean(losses[-50:]):.4f} {secs:.0f}s", flush=True)
    return dict(steps=step, examples=seen, final_loss=float(np.mean(losses[-50:])), seconds=secs,
                epochs=seen / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 steps per model on a tiny slice (code check)")
    ap.add_argument("--only", default="", help="comma-separated run names")
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

    root = D.download() if not a.smoke else D.DATA
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    split = "dev" if a.smoke else "train"     # the smoke check also runs where only dev files exist
    lim = 64 if a.smoke else None
    qs = {h: D.load_questions(kb, h, split, root, limit=lim) for h in (1, 2, 3)}
    facts = fact_queries(store)
    if a.smoke:
        facts = facts[:64]
    vocab = D.Vocab.build([q for h in qs for q in qs[h]] + facts, extra=[D.ENTITY_SLOT])
    n_ent = kb.n_entities
    store_t = tuple(torch.tensor(x.astype(np.int64), device=dev) for x in (store.e_head, store.e_dir, store.e_tail))

    probe = Router(len(vocab), max_len=MAX_LEN)
    router_bytes = 4 * n_params(probe)
    pointer_bytes = router_bytes + store.nbytes
    dims = {"param_1x": pick_param_dim(pointer_bytes, n_ent, len(vocab)),
            "param_4x": pick_param_dim(4 * pointer_bytes, n_ent, len(vocab))}
    log = dict(device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", torch=torch.__version__,
               vocab=vocab.itos, n_entities=n_ent, store_bytes=store.nbytes, router_bytes_fp32=router_bytes,
               pointer_system_bytes_fp32=pointer_bytes, param_dims=dims, runs={}, smoke=a.smoke)

    # untrained router: a control (real store, random "how to connect" weights)
    torch.manual_seed(0)
    np.savez_compressed(out_dir / "router_untrained.npz", **to_numpy_dict(Router(len(vocab), max_len=MAX_LEN)))

    only = set(filter(None, a.only.split(",")))
    t_all = time.time()
    for name, kind, hops, frac, seed, secs in RUNS:
        if only and name not in only:
            continue
        if time.time() - t_all > TOTAL_CAP_S:
            print(f"  {name}: skipped, total time cap reached")
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        train = [q for h in hops for q in qs[h]]
        if frac < 1:
            train = [train[i] for i in rng.choice(len(train), max(8, int(frac * len(train))), replace=False)]
        if kind == "router":
            model = Router(len(vocab), max_len=MAX_LEN).to(dev)
        else:
            train = train + facts
            model = ParamQA(len(vocab), n_ent, vocab.stoi[D.ENTITY_SLOT], d=dims[name], max_len=MAX_LEN).to(dev)
        stats = train_one(name, kind, model, tensorize(train, vocab, dev), store_t, n_ent, secs, dev, a.smoke)
        stats.update(kind=kind, hops=list(hops), fraction=frac, seed=seed, n_train=len(train),
                     params=n_params(model), budget_s=secs)
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
