"""
GPU latency microbenchmarks for Obi's "nanoseconds or microseconds?" question. Needs CUDA.

  python bench_gpu.py      # ~20 s, writes results/latency_gpu.json

Measures, on the GPU:
  launch_us            one tiny kernel + sync: the fixed cost of asking the GPU anything
  pcie_roundtrip_us    copy 1 int to the GPU and back
  router_batch1_us     the router for ONE question (eager PyTorch), incl. sync
  router_batch4096_ns  the router per question when 4,096 questions share one call
  deref_batched_ns     one pointer dereference per key, 16M keys in one call (memory-bound)
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import data as D
from model_torch import Router
from store import PointerStore

HERE = Path(__file__).parent


def timed(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = torch.device("cuda")
    log = json.loads((HERE / "weights/train_log.json").read_text())
    vocab = D.Vocab(log["vocab"])
    W = np.load(HERE / "weights/router_s0.npz")
    m = Router(len(vocab), max_len=32)
    m.load_state_dict({k: torch.tensor(W[k].astype(np.float32)) for k in W.files})
    m = m.to(dev).eval()

    kb = D.KB.load(D.download() / "kb/kb.txt")
    store = PointerStore.from_kb(kb)
    qs = D.load_questions(kb, 2, "test")
    out = {"gpu": torch.cuda.get_device_name()}
    x = torch.zeros(1, device=dev)
    out["launch_us"] = timed(lambda: x.add_(1), 2000) * 1e6
    host = torch.zeros(1, dtype=torch.int64).pin_memory()
    out["pcie_roundtrip_us"] = timed(lambda: host.to(dev, non_blocking=True).cpu(), 2000) * 1e6
    one = torch.tensor(vocab.batch(qs[:1], 32), device=dev)
    big = torch.tensor(vocab.batch((qs * 2)[:4096], 32), device=dev)
    with torch.inference_mode():
        out["router_batch1_us"] = timed(lambda: m(one), 500) * 1e6
        out["router_batch4096_ns_per_q"] = timed(lambda: m(big), 50) / 4096 * 1e9
        indptr = torch.tensor(store.indptr.astype(np.int64), device=dev)
        indices = torch.tensor(store.indices.astype(np.int64), device=dev)
        keys = torch.randint(0, len(indptr) - 1, (16_000_000,), device=dev)
        out["deref_batched_ns_per_pointer"] = timed(
            lambda: indices[indptr[keys].clamp(max=len(indices) - 1)], 20) / len(keys) * 1e9
    out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/latency_gpu.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
