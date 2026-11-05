"""
The pointer memory: a dictionary from (entity, relation) to a list of entity ids.

It is stored as one CSR table (compressed sparse rows), the same trick search engines use for
posting lists: `indptr[k]` and `indptr[k+1]` bracket the neighbours of key k = entity*18 + direction
inside one flat `indices` array. A lookup is two array reads and a slice, so it costs the same
whether the store holds 1k or 100M facts. This is the "constant time" pointer Obi asked about.

The store holds all content. The router (router_np.py) holds only "which pointer to follow next".
"""
from __future__ import annotations

import numpy as np

from data import DIR_NAMES, ENTITY_SLOT, KB, N_DIR, RELATIONS


class PointerStore:
    def __init__(self, heads, dirs, tails, n_entities):
        self.n = int(n_entities)
        key = heads.astype(np.int64) * N_DIR + dirs.astype(np.int64)
        order = np.lexsort((tails, key))
        key, tails = key[order], tails[order].astype(np.int32)
        counts = np.bincount(key, minlength=self.n * N_DIR)
        self.indptr = np.zeros(self.n * N_DIR + 1, dtype=np.int32)
        np.cumsum(counts, out=self.indptr[1:])
        self.indices = tails
        # edge list (head, dir, tail) for the GPU soft executor in model_torch.py
        self.e_head = (key // N_DIR).astype(np.int32)
        self.e_dir = (key % N_DIR).astype(np.int8)
        self.e_tail = tails

    # ------------------------------------------------------------- building ------------
    @classmethod
    def from_kb(cls, kb: KB, h=None, r=None, t=None):
        """Both directions of every fact: forward r, inverse r+9."""
        h = kb.h if h is None else h
        r = kb.r if r is None else r
        t = kb.t if t is None else t
        R = len(RELATIONS)
        heads = np.concatenate([h, t])
        dirs = np.concatenate([r.astype(np.int16), r.astype(np.int16) + R])
        tails = np.concatenate([t, h])
        # de-duplicate repeated facts
        k = np.unique(np.stack([heads.astype(np.int64), dirs.astype(np.int64), tails.astype(np.int64)], 1), axis=0)
        return cls(k[:, 0], k[:, 1], k[:, 2], kb.n_entities)

    def permuted(self, seed=0):
        """Negative control: same keys and same number of pointers per key, but every pointer aims
        at a random entity. Shape of the memory intact, content destroyed."""
        rng = np.random.default_rng(seed)
        s = PointerStore.__new__(PointerStore)
        s.n, s.indptr = self.n, self.indptr
        s.indices = rng.permutation(self.indices).astype(np.int32)
        s.e_head, s.e_dir, s.e_tail = self.e_head, self.e_dir, s.indices
        return s

    @classmethod
    def empty(cls, n_entities):
        z = np.zeros(0, dtype=np.int32)
        return cls(z, z, z, n_entities)

    # ------------------------------------------------------------- lookups -------------
    def neighbours(self, entity: int, direction: int) -> np.ndarray:
        k = entity * N_DIR + direction
        return self.indices[self.indptr[k]:self.indptr[k + 1]]

    def hop(self, entities: np.ndarray, weights: np.ndarray, direction: int):
        """Follow one pointer type from a weighted set of entities. Returns (entities, weights),
        where weight = number of paths that reach each entity."""
        if len(entities) == 0:
            return entities, weights
        k = entities.astype(np.int64) * N_DIR + direction
        start, stop = self.indptr[k], self.indptr[k + 1]
        lens = stop - start
        total = int(lens.sum())
        if total == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        # vectorised gather of all neighbour slices
        offs = np.repeat(start - np.cumsum(lens) + lens, lens) + np.arange(total)
        nb = self.indices[offs]
        w = np.repeat(weights, lens)
        uniq, inv = np.unique(nb, return_inverse=True)
        return uniq.astype(np.int32), np.bincount(inv, weights=w).astype(np.float32)

    def follow(self, topic: int, path, avoid_topic=False) -> tuple[np.ndarray, np.ndarray]:
        """avoid_topic=True is a non-backtracking walk: paths may not pass back through the topic
        (added after the pre-registered run; see results/nonbacktracking.json)."""
        ents, w = np.array([topic], np.int32), np.ones(1, np.float32)
        for d in path:
            ents, w = self.hop(ents, w, int(d))
            if avoid_topic:
                keep = ents != topic
                ents, w = ents[keep], w[keep]
        return ents, w

    def answer(self, topic: int, path, avoid_topic=False) -> int:
        """Top-1 answer: the most-reached entity other than the topic (ties -> lowest id)."""
        ents, w = self.follow(topic, path, avoid_topic)
        keep = ents != topic
        ents, w = ents[keep], w[keep]
        if len(ents) == 0:
            return -1
        return int(ents[np.argmax(w)])

    @property
    def nbytes(self):
        return int(self.indptr.nbytes + self.indices.nbytes)


def fact_queries(store: PointerStore):
    """For the no-memory baseline: every (entity, pointer type) key as a pseudo-question, so the
    baseline is shown every fact in the store at least once per epoch."""
    keys = np.nonzero(np.diff(store.indptr))[0]
    out = []
    for k in keys:
        e, d = divmod(int(k), N_DIR)
        nb = store.indices[store.indptr[k]:store.indptr[k + 1]]
        ans = sorted(set(nb.tolist()) - {e})
        if ans:
            out.append(dict(tokens=["<fact>", f"<r:{DIR_NAMES[d]}>", ENTITY_SLOT], topic=e, answers=ans, hop=0))
    return out
