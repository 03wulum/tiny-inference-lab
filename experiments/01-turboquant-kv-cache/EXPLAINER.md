# Experiment 01, explained

## The problem

When a language model writes text, it keeps a memory of every word so far: the **KV cache**. For each word, at each layer, it stores two lists of numbers (a "key" and a "value"). On a phone or laptop this cache, not the model itself, is often what runs out of memory on long conversations. A 1B-parameter model with an 8K-token context can need hundreds of MB of cache in fp16.

The obvious fix is to store each number with fewer bits, say 3 instead of 16. That's 5× less memory. The catch is that real key vectors have a few "outlier" channels that are huge. Standard quantization sets its range by the biggest value, so those outliers leave very few levels for everything else.

## The paper's trick

TurboQuant spins every vector by the same random rotation before quantizing it. After a random rotation, every coordinate looks the same statistically (roughly a bell curve with a known width), whatever the input looked like. Outliers get smeared across all coordinates. Because the distribution is known in advance, you can precompute the single best set of quantization levels for it once, and reuse it for every model, layer, and token. No calibration data, no training.

The paper proves that this lands within about 2.7× of the best error *any* method could achieve at that bit width.

## What I did

1. **Checked the math** on random vectors: is the error inside the proven bound, and does it match the paper's own numbers? Yes, almost exactly (0.361 vs their 0.36 at 1 bit, 0.116 vs 0.117 at 2 bits).
2. **Checked the two variants:** the paper has an "MSE" version (smallest error) and a "prod" version (adds a 1-bit sketch so dot products come out right on average). Both behave exactly as the theory says.
3. **Built a fake KV cache** with outlier channels like real models have, and compared attention (which words the model "looks at") under TurboQuant vs standard int quantization at 2, 3, and 4 bits.
4. **Negative control:** ran the same thing *without* the rotation, to show that the test can catch a broken method. It breaks badly (7× over the bound), which is what we wanted.

## What happened

- At ~2-3 bits, TurboQuant usually keeps attention far closer to the original: 2-6× less error than int quantization on 5 of the 6 caches I tried.
- On one cache it was no better. The cause was the rotation itself. Different random rotations gave very different results on that cache (the spread was larger than the average).
- Surprise: the "prod" version, which is mathematically unbiased, is *worse* for attention. Attention runs through softmax, which amplifies noise, so a little consistent bias does less harm than extra noise.

## Why it matters

Memory is the bottleneck for running LLMs on-device. A method that needs no calibration and runs at ~235k vectors/s per CPU core in plain NumPy could be dropped into a phone runtime. The open question this experiment raises is how to avoid drawing a bad rotation.

## Words worth knowing

- **Quantization:** storing numbers with fewer bits (e.g. 16 → 3).
- **MSE (mean squared error):** the average squared difference between original and compressed vectors.
- **KL divergence:** how different two probability distributions are. Here, the original vs compressed attention.
- **Lloyd-Max quantizer:** the optimal set of levels for a known distribution.
- **QJL:** a 1-bit sketch: keep only the sign of random projections.

## Further reading

- The paper: [TurboQuant, arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Why KV caches have outliers: [KIVI, arXiv:2402.02750](https://arxiv.org/abs/2402.02750) and [KVQuant, arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- Rotations before quantization in LLMs: [QuaRot, arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
