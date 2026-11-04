"""
NumPy inference for the router and the no-memory baseline (weights saved by train.py as .npz).

Hard pointer mode: the router picks the most likely number of hops and, for each hop, the most
likely pointer type. The PointerStore then follows those pointers exactly. No softmax over the
whole knowledge base, no GPU, no torch.
"""
from __future__ import annotations

import numpy as np

N_HOPS = 3


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _lin(W, p, x):
    return x @ W[p + ".weight"].T + W[p + ".bias"]


class NpEncoder:
    def __init__(self, W, prefix="enc", heads=4):
        self.W = W
        self.p = prefix
        self.layers = sorted({int(k.split(".")[2]) for k in W if k.startswith(f"{prefix}.blocks.")})
        self.heads = heads

    def __call__(self, ids, x=None):
        W, p = self.W, self.p
        pad = ids == 0
        if x is None:
            x = W[f"{p}.tok.weight"][ids]
        x = x + W[f"{p}.pos.weight"][: ids.shape[1]]
        B, L, d = x.shape
        hd = d // self.heads
        for i in self.layers:
            b = f"{p}.blocks.{i}"
            y = _ln(x, W[f"{b}.ln1.weight"], W[f"{b}.ln1.bias"])
            qkv = _lin(W, f"{b}.qkv", y).reshape(B, L, 3, self.heads, hd).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
            att = np.where(pad[:, None, None, :], -1e9, att)
            o = (_softmax(att) @ v).transpose(0, 2, 1, 3).reshape(B, L, d)
            x = x + _lin(W, f"{b}.proj", o)
            y = _ln(x, W[f"{b}.ln2.weight"], W[f"{b}.ln2.bias"])
            x = x + _lin(W, f"{b}.ff2", np.maximum(_lin(W, f"{b}.ff1", y), 0))
        return _ln(x, W[f"{p}.ln.weight"], W[f"{p}.ln.bias"]), pad


def load(path) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


class NpRouter:
    def __init__(self, W):
        self.W = W
        self.enc = NpEncoder(W)
        self.d = W["hop_emb"].shape[1]

    def probs(self, ids):
        """ids [B, L] -> rel_p [B, 3, n_dir], hop_p [B, 3]"""
        W = self.W
        h, pad = self.enc(ids)
        cls = h[:, 0]
        prev = np.zeros_like(cls)
        rels = []
        for t in range(N_HOPS):
            q = np.tanh(_lin(W, "q", np.concatenate([cls, np.broadcast_to(W["hop_emb"][t], cls.shape), prev], -1)))
            a = np.einsum("bld,bd->bl", h, q) / np.sqrt(self.d)
            a = _softmax(np.where(pad, -1e9, a))
            c = np.einsum("bl,bld->bd", a, h)
            p = _softmax(_lin(W, "rel", np.concatenate([c, q], -1)))
            rels.append(p)
            prev = p @ W["rel_emb"]
        return np.stack(rels, 1), _softmax(_lin(W, "n_hops", cls))

    def paths(self, ids):
        """Hard decisions: list of pointer-type sequences, one per question."""
        rel_p, hop_p = self.probs(ids)
        n = hop_p.argmax(-1) + 1
        r = rel_p.argmax(-1)
        return [tuple(int(x) for x in r[i, : n[i]]) for i in range(len(ids))]


class NpParamQA:
    def __init__(self, W, slot_id):
        self.W, self.slot = W, slot_id
        self.enc = NpEncoder(W)

    def logits(self, ids, topic):
        W = self.W
        x = W["enc.tok.weight"][ids]
        x = np.where((ids == self.slot)[:, :, None], W["ent.weight"][topic][:, None, :], x)
        h, _ = self.enc(ids, x)
        return _lin(W, "out", h[:, 0]) @ W["ent.weight"].T + W["bias"]

    def answer(self, ids, topic):
        lg = self.logits(ids, topic)
        lg[np.arange(len(topic)), topic] = -np.inf          # never answer the topic itself
        return lg.argmax(-1)
