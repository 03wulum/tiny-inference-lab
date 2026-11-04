# tiny-inference-lab

Small, runnable reproductions of recent research on **efficient, on-device AI**: KV-cache compression, quantization, and low-memory inference. Every experiment runs in minutes on a laptop CPU and ends in a pre-registered PASS / FAIL / INCONCLUSIVE verdict.

## Results

| # | Goal | Verdict | Details |
|---|---|---|---|
| 01 | Compress the KV cache to ~3 bits with no calibration ([TurboQuant](https://arxiv.org/abs/2504.19874), ICLR 2026) | **INCONCLUSIVE** (4/5) ✨ | [README](experiments/01-turboquant-kv-cache) |
| 02 | Give each token its own KV bit budget ([Don't Waste Bits](https://arxiv.org/abs/2604.04722), 2026) | *in progress* | |
| 04 | Answer multi-hop questions with a 0.3M-param router that only stores *how to connect* facts, plus a constant-time pointer store | *pending (GPU run queued)* | [README](experiments/04-pointer-memory) |

✨ **01:** every claim from the paper reproduced. TurboQuant beat int quantization on 5 of 6 KV caches, and a bad random rotation can cancel the win on the sixth.

## How each experiment is judged

- **Pass criteria are written down before the full run** in `criteria.py`, with thresholds from the paper or a baseline.
- **Each experiment has a negative control,** a broken variant that should FAIL. If it passes, the verdict is INCONCLUSIVE.
- **Perturbations:** a criterion must hold across seeds, sizes, and data draws, not just one lucky run.
- **Verdicts:** PASS = all criteria pass. FAIL = the core criterion fails. INCONCLUSIVE = mixed results. ✨ INTRIGUING marks a surprise worth a follow-up.

## Run

```bash
git clone https://github.com/03wulum/tiny-inference-lab && cd tiny-inference-lab
pip install -r requirements.txt
python run_all.py --quick   # smoke-run everything in seconds
python run_all.py           # full runs → verdict table
python run_all.py --clean   # remove caches and downloaded data
```

Each experiment folder has a short README (verdict + PASS/FAIL table), an `EXPLAINER.md` in plain language, and `docs/DETAILS.md` with method, controls, and limitations. CI runs the tests and a quick run of every experiment on each push.

MIT licensed.
