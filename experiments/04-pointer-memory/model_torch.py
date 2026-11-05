"""
PyTorch models used ONLY for training on the GPU (train.py). Inference and evaluation use the
NumPy mirror in router_np.py, so CI and laptops need no torch.

  Router     : question -> which pointer to follow at each hop (+ how many hops). ~0.3M params.
               It sees the question with the topic entity blanked out as <E>, so it cannot
               memorise facts. All facts live in the PointerStore.
  ParamQA    : the no-memory baseline. Same encoder, but it is given the topic entity as an
               embedding and must answer from its own weights (an entity table + output head).
  soft_follow: differentiable version of PointerStore.follow, so the router learns from
               (question, answer set) pairs only, never from labelled relation paths.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HOPS = 3


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.heads = heads

    def forward(self, x, pad):                    # x [B, L, d], pad [B, L] True = padding
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        att = att.masked_fill(pad[:, None, None, :], -1e9).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))


class Encoder(nn.Module):
    def __init__(self, vocab, d, layers=2, heads=4, ff=None, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff or 2 * d) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, ids, x=None):
        pad = ids == 0
        if x is None:
            x = self.tok(ids)
        x = x + self.pos.weight[: ids.shape[1]]
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x), pad


class Router(nn.Module):
    def __init__(self, vocab, n_dir=18, d=128, layers=2, heads=4, max_len=32):
        super().__init__()
        self.enc = Encoder(vocab, d, layers, heads, max_len=max_len)
        self.hop_emb = nn.Parameter(torch.randn(N_HOPS, d) * 0.02)
        self.rel_emb = nn.Parameter(torch.randn(n_dir, d) * 0.02)
        self.q = nn.Linear(3 * d, d)
        self.rel = nn.Linear(2 * d, n_dir)
        self.n_hops = nn.Linear(d, N_HOPS)
        self.d = d

    def forward(self, ids):
        """Returns relation probabilities per hop [B, 3, n_dir] and hop-count probabilities [B, 3]."""
        h, pad = self.enc(ids)
        cls = h[:, 0]
        prev = torch.zeros_like(cls)
        rels = []
        for t in range(N_HOPS):
            q = torch.tanh(self.q(torch.cat([cls, self.hop_emb[t].expand_as(cls), prev], -1)))
            a = ((h @ q[:, :, None]).squeeze(-1) / math.sqrt(self.d)).masked_fill(pad, -1e9).softmax(-1)
            c = (a[:, :, None] * h).sum(1)
            p = self.rel(torch.cat([c, q], -1)).softmax(-1)
            rels.append(p)
            prev = p @ self.rel_emb
        return torch.stack(rels, 1), self.n_hops(cls).softmax(-1)


def soft_follow(topic, rel_p, hop_p, e_head, e_dir, e_tail, n_entities):
    """Differentiable multi-hop pointer following over the whole store, on the GPU.
    topic [B], rel_p [B, 3, n_dir], hop_p [B, 3]. Returns entity scores [B, n_entities] in [0, 1]."""
    B = topic.shape[0]
    e = torch.zeros(B, n_entities, device=topic.device)
    e[torch.arange(B), topic] = 1.0
    score = torch.zeros_like(e)
    for t in range(N_HOPS):
        msg = e[:, e_head] * rel_p[:, t, e_dir]                    # [B, n_edges]
        e = torch.zeros_like(e).index_add_(1, e_tail, msg).clamp(max=1.0)
        score = score + hop_p[:, t:t + 1] * e
    return score


class ParamQA(nn.Module):
    """No-memory baseline: all knowledge must live in the weights."""

    def __init__(self, vocab, n_entities, entity_slot_id, d=64, layers=2, heads=4, max_len=32):
        super().__init__()
        self.enc = Encoder(vocab, d, layers, heads, max_len=max_len)
        self.ent = nn.Embedding(n_entities, d)          # input AND output entity table (tied)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(n_entities))
        self.slot = entity_slot_id

    def forward(self, ids, topic):
        x = self.enc.tok(ids)
        is_slot = (ids == self.slot)[:, :, None]
        x = torch.where(is_slot, self.ent(topic)[:, None, :].expand_as(x), x)
        h, _ = self.enc(ids, x)
        return self.out(h[:, 0]) @ self.ent.weight.T + self.bias      # logits [B, n_entities]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def to_numpy_dict(m: nn.Module, dtype="float16") -> dict:
    return {k: v.detach().float().cpu().numpy().astype(dtype) for k, v in m.state_dict().items()}
