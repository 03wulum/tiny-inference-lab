# Experiment 04: all numbers

Questions per hop: {1: 9945, 2: 14872, 3: 14274}. Train device: NVIDIA GeForce RTX 4080 SUPER, total GPU training 291s. Eval took 148s.

## Hits@1 (%) on MetaQA test

| model | 1-hop | 2-hop | 3-hop |
|---|---|---|---|
| router_s0 | 95.5 | 99.2 | 82.8 |
| router_s1 | 95.4 | 97.6 | 83.8 |
| router_s2 | 97.1 | 99.9 | 83.8 |
| param_1x | 6.8 | 10.0 | 30.1 |
| param_4x | 16.2 | 10.8 | 34.3 |
| control: random_pointers | 0.2 | 0.4 | 0.5 |
| control: no_memory | 0.0 | 0.0 | 0.0 |
| control: untrained_router | 0.0 | 0.0 | 0.0 |
| exploratory: router_12hop | 97.1 | 99.9 | 0.0 |
| exploratory: router_1pct | 97.1 | 98.4 | 83.4 |

## Knowledge edit (no retraining)

1642 movies rewired; 568 test questions whose answer changed (of 2305 with a unique gold path).

- pointer model before edit: 99.6%, after edit (vs new answers): 99.6%
- no-memory model vs new answers: 6.2% (still gives the old answer: 7.7%)

## Did the no-memory models learn the facts they were shown?

- param_1x: Hits@1 on 3000 training facts (entity + pointer type -> any target): 7.4%
- param_4x: Hits@1 on 3000 training facts (entity + pointer type -> any target): 25.9%

## Latency (CPU, this machine)

- deref_python_ns: 219.6
- deref_vectorised_ns: 6.8
- e2e_cpu_batch1_us_median: 549.0
- e2e_cpu_batch1_us_p90: 775.5
- router_cpu_batched_us_per_q: 266.6
- param_1x_cpu_batch1_us_median: 3,923.3

## Latency (GPU, from bench_gpu.py)

- gpu: NVIDIA GeForce RTX 4080 SUPER
- launch_us: 17.4
- pcie_roundtrip_us: 25.3
- router_batch1_us: 3523.35
- router_batch4096_ns_per_q: 3669.019
- deref_batched_ns_per_pointer: 0.08

## Size

- kept weights + store: 21.2 MB (store 4.2 MB)
- parameters: {'router_s0': 344981, 'router_s1': 344981, 'router_s2': 344981, 'router_12hop': 344981, 'router_1pct': 344981, 'param_1x': 1450626, 'param_4x': 5690814}
- no-memory widths: {'param_1x': 32, 'param_4x': 124}
