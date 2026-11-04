# tiny-inference-lab

Small, runnable reproductions of recent research on **efficient, on-device AI**: KV-cache compression, quantization, low-memory inference, and local agents that never touch the cloud.

Each experiment is:

- **one paper, one or more falsifiable claims**, with a verdict (reproduced, partially, or not)
- **runnable in minutes on a laptop CPU** (NumPy first, heavier deps only when the claim needs a real model)
- **honest about what the abstract leaves out**: every README has a "what I found" section

| # | Experiment | Paper | Headline result | Runtime |
|---|---|---|---|---|
| 01 | [TurboQuant KV-cache compression](experiments/01-turboquant-kv-cache) | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026) | Distortion bounds reproduced exactly. At ~2 bits, KL to fp16 attention is 25× lower than per-token int quantization. The "unbiased" variant is *worse* for softmax attention | ~15 s |
| 02 | Adaptive per-token KV bit allocation | [arXiv:2604.04722](https://arxiv.org/abs/2604.04722) | _next_ | |

## Quick start

```bash
git clone https://github.com/03wulum/tiny-inference-lab && cd tiny-inference-lab
pip install -r requirements.txt
cd experiments/01-turboquant-kv-cache && python run.py
```

CI runs every experiment's tests and a `--quick` smoke run on each push.

## License

MIT
