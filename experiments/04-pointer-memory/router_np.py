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
    xc = x - x.mean(-1, keepdims=True)
    return xc * (w / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)) + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _lin(W, p, x):
    wt = W.get(p + ".wT")
    if wt is None:                                  # cache a contiguous transpose: ~2x faster matmul
        wt = W[p + ".wT"] = np.ascontiguousarray(W[p + ".weight"].T)
    y = x.reshape(-1, x.shape[-1]) @ wt + W[p + ".bias"]      # 2-D matmul hits BLAS directly
    return y.reshape(*x.shape[:-1], wt.shape[1])


class NpEncoder:
    def __init__(self, W, prefix="enc", heads=4):
        self.W = W
        self.p = prefix
        self.layers = sorted({int(k.split(".")[2]) for k in W if k.startswith(f"{prefix}.blocks.")})
        self.heads = heads

    def __call__(self, ids, x=None):
        W, p = self.W, self.p
        n = int((ids != 0).sum(1).max())                  # drop all-padding columns: same output, less work
        ids = ids[:, :n]
        x = None if x is None else x[:, :n]
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
        d = self.d = W["hop_emb"].shape[1]
        # Split the hop-query and relation layers into blocks once, so each hop is a few small
        # matmuls with no concatenation (algebraically identical to model_torch.Router).
        Wq, Wr = W["q.weight"], W["rel.weight"]
        self.q_cls = np.ascontiguousarray(Wq[:, :d].T)
        self.q_hop = W["hop_emb"] @ Wq[:, d:2 * d].T + W["q.bias"]           # [3, d] constant per hop
        self.q_prev = W["rel_emb"] @ Wq[:, 2 * d:].T                         # [n_dir, d]: p @ this
        self.r_c = np.ascontiguousarray(Wr[:, :d].T)
        self.r_q = np.ascontiguousarray(Wr[:, d:].T)
        self.n_w = np.ascontiguousarray(W["n_hops.weight"].T)

    def probs(self, ids):
        """ids [B, L] -> rel_p [B, 3, n_dir], hop_p [B, 3]"""
        W = self.W
        h, pad = self.enc(ids)
        cls = h[:, 0]
        base = cls @ self.q_cls
        neg = np.where(pad, -1e9, 0.0).astype(np.float32)
        scale = 1.0 / np.sqrt(self.d)
        rels, prev = [], 0.0
        for t in range(N_HOPS):
            q = np.tanh(base + self.q_hop[t] + prev)
            a = _softmax((h @ q[:, :, None])[:, :, 0] * scale + neg)
            c = (a[:, None, :] @ h)[:, 0]
            p = _softmax(c @ self.r_c + q @ self.r_q + W["rel.bias"])
            rels.append(p)
            prev = p @ self.q_prev
        return np.stack(rels, 1), _softmax(cls @ self.n_w + W["n_hops.bias"])

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
