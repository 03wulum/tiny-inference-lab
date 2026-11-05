"""
NumPy inference mirrors of models_torch.py (weights saved by train.py as fp16 .npz).
Router flavors return hard pointer paths; the no-memory rivals return top-1 entity ids.
"""
from __future__ import annotations

import numpy as np

FIXED_HOPS = 3


def _ln(x, w, b, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    return xc * (w / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)) + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lin(W, p, x):
    wt = W.get(p + ".wT")
    if wt is None:                                  # cached contiguous transpose: faster matmul
        wt = W[p + ".wT"] = np.ascontiguousarray(W[p + ".weight"].T)
    y = x.reshape(-1, x.shape[-1]) @ wt + W[p + ".bias"]
    return y.reshape(*x.shape[:-1], wt.shape[1])


def load(path) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


def _trim(ids):
    n = int((ids != 0).sum(1).max())                # padding is at the end: drop all-pad columns
    return ids[:, :n]


def run_blocks(W, prefix, x, pad, heads=4):
    layers = sorted({int(k[len(prefix):].split(".")[0]) for k in W if k.startswith(prefix)})
    B, L, d = x.shape
    hd = d // heads
    for i in layers:
        b = f"{prefix}{i}"
        y = _ln(x, W[f"{b}.ln1.weight"], W[f"{b}.ln1.bias"])
        qkv = _lin(W, f"{b}.qkv", y).reshape(B, L, 3, heads, hd).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = np.where(pad[:, None, None, :], -1e9, q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd))
        x = x + _lin(W, f"{b}.proj", (_softmax(att) @ v).transpose(0, 2, 1, 3).reshape(B, L, d))
        y = _ln(x, W[f"{b}.ln2.weight"], W[f"{b}.ln2.bias"])
        x = x + _lin(W, f"{b}.ff2", np.maximum(_lin(W, f"{b}.ff1", y), 0))
    return x


def encode(W, ids, x=None, p="enc"):
    """models_torch.Encoder (absolute positions)."""
    pad = ids == 0
    x = (W[f"{p}.tok.weight"][ids] if x is None else x) + W[f"{p}.pos.weight"][: ids.shape[1]]
    x = run_blocks(W, f"{p}.blocks.", x, pad)
    return _ln(x, W[f"{p}.ln.weight"], W[f"{p}.ln.bias"]), pad


def _hard(rel_p, hop_p):
    n = hop_p.argmax(-1) + 1
    r = rel_p.argmax(-1)
    return [tuple(int(x) for x in r[i, : n[i]]) for i in range(len(r))]


class NpRouter:
    def __init__(self, W):
        self.W = W
        d = self.d = W["hop_emb"].shape[1]
        Wq, Wr = W["q.weight"], W["rel.weight"]
        self.q_cls = np.ascontiguousarray(Wq[:, :d].T)
        self.q_hop = W["hop_emb"] @ Wq[:, d:2 * d].T + W["q.bias"]
        self.q_prev = W["rel_emb"] @ Wq[:, 2 * d:].T
        self.r_c, self.r_q = np.ascontiguousarray(Wr[:, :d].T), np.ascontiguousarray(Wr[:, d:].T)
        self.n_w = np.ascontiguousarray(W["n_hops.weight"].T)

    def probs(self, ids, T=FIXED_HOPS):
        W = self.W
        h, pad = encode(W, _trim(ids))
        cls = h[:, 0]
        base = cls @ self.q_cls
        neg = np.where(pad, -1e9, 0.0).astype(np.float32)
        rels, prev = [], 0.0
        for t in range(FIXED_HOPS):
            q = np.tanh(base + self.q_hop[t] + prev)
            a = _softmax((h @ q[:, :, None])[:, :, 0] / np.sqrt(self.d) + neg)
            c = (a[:, None, :] @ h)[:, 0]
            p = _softmax(c @ self.r_c + q @ self.r_q + W["rel.bias"])
            rels.append(p)
            prev = p @ self.q_prev
        return np.stack(rels, 1), _softmax(cls @ self.n_w + W["n_hops.bias"])

    def paths(self, ids, T=FIXED_HOPS):
        return _hard(*self.probs(ids))


class NpBowRouter:
    def __init__(self, W):
        self.W = W
        self.n_dir = W["rel.weight"].shape[0] // FIXED_HOPS

    def probs(self, ids, T=FIXED_HOPS):
        W = self.W
        m = (ids != 0).astype(np.float32)[:, :, None]
        x = (W["tok.weight"][ids] * m).sum(1) / np.maximum(m.sum(1), 1)
        x = np.maximum(_lin(W, "mlp", x), 0)
        rel = _softmax(_lin(W, "rel", x).reshape(-1, FIXED_HOPS, self.n_dir))
        return rel, _softmax(_lin(W, "n_hops", x))

    def paths(self, ids, T=FIXED_HOPS):
        return _hard(*self.probs(ids))


class NpRecRouter:
    def __init__(self, W, slot_id):
        self.W, self.slot = W, slot_id
        self.d = W["tok.weight"].shape[1]
        self.gamma = float(np.log1p(np.exp(W["cov"])))
        self.cw = W["conv.weight"]                        # [out, in, 3]

    def encode(self, ids):
        W = self.W
        pad = ids == 0
        x = W["tok.weight"][ids] * (~pad)[:, :, None]
        xp = np.pad(x, ((0, 0), (1, 1), (0, 0)))
        conv = sum(xp[:, k:k + x.shape[1]] @ self.cw[:, :, k].T for k in range(3)) + W["conv.bias"]
        x = x + np.maximum(conv, 0)
        x = run_blocks(W, "blocks.", x, pad)
        return _ln(x, W["ln.weight"], W["ln.bias"]), pad

    def _gru(self, x, h):
        W, d = self.W, self.d
        gi = x @ W["cell.weight_ih"].T + W["cell.bias_ih"]
        gh = h @ W["cell.weight_hh"].T + W["cell.bias_hh"]
        r = _sigmoid(gi[:, :d] + gh[:, :d])
        z = _sigmoid(gi[:, d:2 * d] + gh[:, d:2 * d])
        n = np.tanh(gi[:, 2 * d:] + r * gh[:, 2 * d:])
        return (1 - z) * n + z * h

    def run(self, ids, T):
        W = self.W
        ids = _trim(ids)
        h, pad = self.encode(ids)
        B, L, d = h.shape
        epos = (ids == self.slot).argmax(1)
        dist = (epos[:, None] - np.arange(L)[None, :]).astype(np.float32) / 8.0
        relness = _sigmoid(_lin(W, "relness", h))[:, :, 0] * (~pad)
        keep = (~pad).astype(np.float32)[:, :, None]
        s = (h * keep).sum(1) / keep.sum(1)
        cov = np.zeros((B, L), np.float32)
        prev = np.zeros((B, d), np.float32)
        rels, halts = [], []
        for _ in range(T):
            q = np.tanh(_lin(W, "q", np.concatenate([s, prev], -1)))
            score = (h @ q[:, :, None])[:, :, 0] / np.sqrt(d) + _lin(W, "slope", q) * dist - self.gamma * cov
            a = _softmax(np.where(pad, -1e9, score))
            c = (a[:, None, :] @ h)[:, 0]
            cov = cov + a
            p = _softmax(_lin(W, "rel", np.concatenate([c, q], -1)))
            remaining = (relness * np.maximum(1 - cov, 0)).sum(1, keepdims=True)
            halts.append(_sigmoid(_lin(W, "halt", np.concatenate([c, q, remaining], -1)))[:, 0])
            rels.append(p)
            s = self._gru(c, s)
            prev = p @ W["rel_emb"]
        return np.stack(rels, 1), np.stack(halts, 1)

    def paths(self, ids, T=5):
        """Hard halting: stop after the first step whose halt probability > 0.5 (at most T steps)."""
        rel_p, halt = self.run(ids, T)
        r = rel_p.argmax(-1)
        stop = halt > 0.5
        stop[:, -1] = True
        n = stop.argmax(1) + 1
        return [tuple(int(x) for x in r[i, : n[i]]) for i in range(len(r))]


class NpParamQA:
    def __init__(self, W, slot_id):
        self.W, self.slot = W, slot_id

    def logits(self, ids, topic):
        W = self.W
        ids = _trim(ids)
        x = W["enc.tok.weight"][ids]
        x = np.where((ids == self.slot)[:, :, None], W["ent.weight"][topic][:, None, :], x)
        h, _ = encode(W, ids, x)
        return _lin(W, "out", h[:, 0]) @ W["ent.weight"].T + W["bias"]

    def answer(self, ids, topic):
        lg = self.logits(ids, topic)
        lg[np.arange(len(topic)), topic] = -np.inf
        return lg.argmax(-1)


class NpKGEQA:
    def __init__(self, W):
        self.W = W
        self.k = W["ent.weight"].shape[1] // 2

    def logits(self, ids, topic):
        W, k = self.W, self.k
        h, _ = encode(W, _trim(ids))
        r = _lin(W, "to_rel", h[:, 0])
        e = W["ent.weight"][topic]
        hr_re = e[:, :k] * r[:, :k] - e[:, k:] * r[:, k:]
        hr_im = e[:, :k] * r[:, k:] + e[:, k:] * r[:, :k]
        E = W["ent.weight"]
        return hr_re @ E[:, :k].T + hr_im @ E[:, k:].T + W["bias"]

    def answer(self, ids, topic):
        lg = self.logits(ids, topic)
        lg[np.arange(len(topic)), topic] = -np.inf
        return lg.argmax(-1)
