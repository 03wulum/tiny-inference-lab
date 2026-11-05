"""
PyTorch models for experiment 05, used ONLY for GPU training (train.py). Evaluation runs on the
NumPy mirrors in models_np.py.

Pointer flavors (facts live in the PointerStore; the router never sees the entity name):
  Router     fixed 3-hop router from experiment 04 (abs. positions, picks hop count 1-3)
  BowRouter  order-blind: bag of words -> pointer types per hop (what does word order buy?)
  RecRouter  recurrent: one shared hop step, repeated; attends to the not-yet-used part of the question
             (coverage), prefers words nearest the entity (a linear distance term that extrapolates),
             halts when no unused relation words remain. Built to go past the 3 hops it trains on.

No-memory rivals (facts must live in the weights):
  ParamQA    transformer "memoriser" with a tied entity table (same as experiment 04)
  KGEQA      knowledge-graph-embedding QA in the style of EmbedKGQA: ComplEx entity embeddings,
             the question becomes one complex "relation" vector, answers scored by ComplEx.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FIXED_HOPS = 3


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.heads = heads

    def forward(self, x, pad):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        att = att.masked_fill(pad[:, None, None, :], -1e9).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))


class Encoder(nn.Module):
    def __init__(self, vocab, d, layers=2, heads=4, max_len=48):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, 2 * d) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, ids, x=None):
        pad = ids == 0
        x = (self.tok(ids) if x is None else x) + self.pos.weight[: ids.shape[1]]
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x), pad


class Router(nn.Module):
    """Experiment 04's router: fixed 3 hop slots + a hop-count head."""

    def __init__(self, vocab, n_dir=18, d=128, layers=2, heads=4, max_len=48):
        super().__init__()
        self.enc = Encoder(vocab, d, layers, heads, max_len)
        self.hop_emb = nn.Parameter(torch.randn(FIXED_HOPS, d) * 0.02)
        self.rel_emb = nn.Parameter(torch.randn(n_dir, d) * 0.02)
        self.q = nn.Linear(3 * d, d)
        self.rel = nn.Linear(2 * d, n_dir)
        self.n_hops = nn.Linear(d, FIXED_HOPS)
        self.d = d

    def forward(self, ids, T=FIXED_HOPS):
        h, pad = self.enc(ids)
        cls = h[:, 0]
        prev = torch.zeros_like(cls)
        rels = []
        for t in range(FIXED_HOPS):
            q = torch.tanh(self.q(torch.cat([cls, self.hop_emb[t].expand_as(cls), prev], -1)))
            a = ((h @ q[:, :, None]).squeeze(-1) / math.sqrt(self.d)).masked_fill(pad, -1e9).softmax(-1)
            c = (a[:, :, None] * h).sum(1)
            p = self.rel(torch.cat([c, q], -1)).softmax(-1)
            rels.append(p)
            prev = p @ self.rel_emb
        return torch.stack(rels, 1), self.n_hops(cls).softmax(-1)


class BowRouter(nn.Module):
    """Order-blind control flavor: mean of word embeddings -> MLP -> pointer type per hop slot."""

    def __init__(self, vocab, n_dir=18, d=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.mlp = nn.Linear(d, d)
        self.rel = nn.Linear(d, FIXED_HOPS * n_dir)
        self.n_hops = nn.Linear(d, FIXED_HOPS)
        self.n_dir = n_dir

    def forward(self, ids, T=FIXED_HOPS):
        m = (ids != 0).float()[:, :, None]
        x = (self.tok(ids) * m).sum(1) / m.sum(1).clamp(min=1)
        x = F.relu(self.mlp(x))
        rel = self.rel(x).view(-1, FIXED_HOPS, self.n_dir).softmax(-1)
        return rel, self.n_hops(x).softmax(-1)


class RecRouter(nn.Module):
    def __init__(self, vocab, slot_id, n_dir=18, d=128, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.conv = nn.Conv1d(d, d, 3, padding=1)          # local word order, no absolute positions
        self.blocks = nn.ModuleList(Block(d, heads, 2 * d) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.rel_emb = nn.Parameter(torch.randn(n_dir, d) * 0.02)
        self.q = nn.Linear(2 * d, d)
        self.slope = nn.Linear(d, 1)
        self.cov = nn.Parameter(torch.tensor(2.0))
        self.relness = nn.Linear(d, 1)
        self.rel = nn.Linear(2 * d, n_dir)
        self.halt = nn.Linear(2 * d + 1, 1)
        self.cell = nn.GRUCell(d, d)
        self.slot, self.d = slot_id, d

    def encode(self, ids):
        pad = ids == 0
        x = self.tok(ids) * (~pad)[:, :, None]
        x = x + F.relu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x), pad

    def forward(self, ids, T=FIXED_HOPS):
        h, pad = self.encode(ids)
        B, L, d = h.shape
        epos = (ids == self.slot).float().argmax(1)
        dist = (epos[:, None] - torch.arange(L, device=ids.device)[None, :]).float() / 8.0
        relness = torch.sigmoid(self.relness(h)).squeeze(-1) * (~pad)
        keep = (~pad).float()[:, :, None]
        s = (h * keep).sum(1) / keep.sum(1)
        cov = torch.zeros(B, L, device=ids.device)
        prev = torch.zeros(B, d, device=ids.device)
        rels, halts = [], []
        gamma = F.softplus(self.cov)
        for _ in range(T):
            q = torch.tanh(self.q(torch.cat([s, prev], -1)))
            score = (h @ q[:, :, None]).squeeze(-1) / math.sqrt(d) + self.slope(q) * dist - gamma * cov
            a = score.masked_fill(pad, -1e9).softmax(-1)
            c = (a[:, :, None] * h).sum(1)
            cov = cov + a
            p = self.rel(torch.cat([c, q], -1)).softmax(-1)
            remaining = (relness * (1 - cov).clamp(min=0)).sum(1, keepdim=True)
            halts.append(torch.sigmoid(self.halt(torch.cat([c, q, remaining], -1))).squeeze(-1))
            rels.append(p)
            s = self.cell(c, s)
            prev = p @ self.rel_emb
        return torch.stack(rels, 1), halt_to_hop_probs(torch.stack(halts, 1))


def halt_to_hop_probs(halt):
    """halt [B, T] = P(stop after step t | reached t). Returns P(answer after exactly t steps) [B, T]."""
    B, T = halt.shape
    out, alive = [], torch.ones(B, device=halt.device)
    for t in range(T):
        p = alive * (halt[:, t] if t < T - 1 else torch.ones_like(alive))
        out.append(p)
        alive = alive * (1 - halt[:, t])
    return torch.stack(out, 1)


def soft_follow(topic, rel_p, hop_p, e_head, e_dir, e_tail, n_entities, avoid_topic=True):
    """Differentiable multi-hop walk over the whole store. With avoid_topic the walk never passes back
    through the topic entity (the loop fix from experiment 04, now used in training too)."""
    B, T = hop_p.shape
    ar = torch.arange(B, device=topic.device)
    e = torch.zeros(B, n_entities, device=topic.device)
    e[ar, topic] = 1.0
    score = torch.zeros_like(e)
    for t in range(T):
        msg = e[:, e_head] * rel_p[:, t, e_dir]
        e = torch.zeros_like(e).index_add_(1, e_tail, msg).clamp(max=1.0)
        if avoid_topic:
            e = e.index_put((ar, topic), torch.zeros(B, device=e.device))
        score = score + hop_p[:, t:t + 1] * e
    return score


class ParamQA(nn.Module):
    def __init__(self, vocab, n_entities, slot_id, d=64, layers=2, heads=4, max_len=48):
        super().__init__()
        self.enc = Encoder(vocab, d, layers, heads, max_len)
        self.ent = nn.Embedding(n_entities, d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(n_entities))
        self.slot = slot_id

    def forward(self, ids, topic):
        x = self.enc.tok(ids)
        x = torch.where((ids == self.slot)[:, :, None], self.ent(topic)[:, None, :].expand_as(x), x)
        h, _ = self.enc(ids, x)
        return self.out(h[:, 0]) @ self.ent.weight.T + self.bias


class KGEQA(nn.Module):
    """EmbedKGQA-style: ComplEx scores Re(<h, r_q, conj(t)>), r_q read from the question."""

    def __init__(self, vocab, n_entities, k=32, d=64, layers=2, heads=4, max_len=48):
        super().__init__()
        self.enc = Encoder(vocab, d, layers, heads, max_len)
        self.ent = nn.Embedding(n_entities, 2 * k)
        nn.init.normal_(self.ent.weight, std=0.1)
        self.to_rel = nn.Linear(d, 2 * k)
        self.bias = nn.Parameter(torch.zeros(n_entities))
        self.k = k

    def forward(self, ids, topic):
        h, _ = self.enc(ids)
        r = self.to_rel(h[:, 0])
        e = self.ent(topic)
        k = self.k
        hr_re = e[:, :k] * r[:, :k] - e[:, k:] * r[:, k:]
        hr_im = e[:, :k] * r[:, k:] + e[:, k:] * r[:, :k]
        W = self.ent.weight
        return hr_re @ W[:, :k].T + hr_im @ W[:, k:].T + self.bias


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def to_numpy_dict(m, dtype="float16"):
    return {k: v.detach().float().cpu().numpy().astype(dtype) for k, v in m.state_dict().items()}
