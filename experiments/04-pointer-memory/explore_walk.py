"""
Exploratory follow-up (added AFTER the pre-registered run, so it does not change the verdict).

The routers scored ~83% on 3-hop, under C2's 85%. Error analysis: almost every miss is a question like
"who directed movies that share actors with the movie <E>". The router picks the right 3 pointers,
but the walk can come back through the topic movie itself (movie -> its actors -> back to the same
movie -> its director), and those paths outvote the real answers. MetaQA's answers exclude them.

This script re-scores the SAME routers with a non-backtracking walk (paths may not revisit the topic).
No retraining. Writes results/nonbacktracking.json.

  python explore_walk.py
"""
from __future__ import annotations

import json
from pathlib import Path

import data as D
from router_np import NpRouter, load
from store import PointerStore

HERE = Path(__file__).parent


def main():
    vocab = D.Vocab(json.loads((HERE / "weights/train_log.json").read_text())["vocab"])
    root = D.download([f for f in D.FILES if "test" in f or "kb" in f])
    kb = D.KB.load(root / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    qs = {h: D.load_questions(kb, h, "test", root) for h in (1, 2, 3)}
    out = {}
    for s in ("router_s0", "router_s1", "router_s2", "router_1pct"):
        r = NpRouter(load(HERE / f"weights/{s}.npz"))
        out[s] = {}
        for h, q in qs.items():
            paths = []
            for i in range(0, len(q), 1024):
                paths += r.paths(vocab.batch(q[i:i + 1024], 32))
            ok = [store.answer(x["topic"], p, avoid_topic=True) in set(x["answers"]) for x, p in zip(q, paths)]
            out[s][h] = 100.0 * sum(ok) / len(ok)
        print(s, {h: round(a, 2) for h, a in out[s].items()})
    out["worst_of_3_seeds"] = {h: min(out[s][h] for s in ("router_s0", "router_s1", "router_s2")) for h in (1, 2, 3)}
    (HERE / "results/nonbacktracking.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
