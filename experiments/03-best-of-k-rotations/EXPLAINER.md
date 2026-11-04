# Experiment 03, explained

**Background:** experiment 01's TurboQuant spins each vector with a random rotation before compressing it. On average this works great, but on one KV cache a single rotation happened to be a bad draw, and the compression did no better than the simple method it was supposed to beat.

**The idea:** a rotation is just a fixed 128×128 matrix. At model-load time, we can try 8 of them and keep the one that distorts attention least on a small sample of 64 queries. Then we test on *different* queries, so we're not just memorizing the sample.

**What happened:** it works. The worst case went from 1.14× the error of plain int quantization to 0.46×, and the average error fell by 39%. Choosing takes about a quarter of a second per cache. As a sanity check, deliberately picking the *worst* of the 8 makes things 3× worse than int quantization, which proves the selection signal is real and not luck.

**Why it matters:** this is calibration-light. It needs 64 queries, not a dataset or any training, and it removes the main risk in a method otherwise attractive for phones. On a real model you'd pick one rotation per layer (or per head) once and store it: 64 KB per 128×128 matrix in fp32.

**Words worth knowing:** *calibration set:* a few examples used to tune a setting before use. *Held-out:* data not used for tuning, used only to measure. *Negative control:* a deliberately broken variant that should fail, to prove the test can tell good from bad.

**Further reading:** [TurboQuant](https://arxiv.org/abs/2504.19874) · [QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406). SpinQuant *learns* the rotation instead of picking one at random, which is the heavier-weight version of this idea.
