# Experiment 04 in plain language: a network of pointers, not a pile of facts

## The idea (Obi's words, made concrete)

Most language models learn facts by soaking them into their weights. Ask "who directed Heat?" and the
answer is smeared across millions of numbers. That has two costs. The model has to get bigger to know
more, and changing one fact means retraining.

Obi's idea: keep the model tiny and make its nodes **pointers** to larger stores. The model doesn't
know *that* Michael Mann directed Heat. It knows *how to get there*: "for a 'who directed' question,
follow the `directed_by` pointer from the movie." The facts live in a dictionary next to it. The
model's job is to pick which pointer to follow, and how many times, as fast as possible, and hand
the result to a bigger model that talks or acts.

## What we built

1. **The pointer store** (`store.py`). The whole knowledge base (135k movie facts from WikiMovies)
   as one dictionary: key = (thing, kind of link), value = the list of things it points to. It is
   stored as two flat arrays (a "CSR" table, the same trick search engines use), so a lookup is
   two array reads and a slice. That costs the same with 135 thousand facts or 135 million.
2. **The router** (`model_torch.py`, `router_np.py`). A ~0.3M-parameter transformer that reads the
   question with the name blanked out ("which person directed the movies starred by `<E>`") and
   outputs a plan: how many hops (1-3), and which of 18 pointer types to follow at each hop. Since it
   never sees the name, it *cannot* memorise facts. It can only learn how ideas connect.
3. **Training without being told the plan.** We only give it questions and their answers. During
   training the router follows *all* pointers a little bit at once, weighted by how much it believes
   in each one (a "soft" walk over the whole store on the GPU). If the soft walk lands on the right
   answers, the plan gets reinforced. At test time it commits to one plan and follows real pointers.
4. **The competitor: a no-memory model** of the *same total size in bytes* as router + store. It gets
   the name as an input and has to answer from its own weights. It is shown every fact in the store
   during training, so it has every chance to memorise them.

## How we judge it (written down before training)

| | Test | Why |
|---|---|---|
| C1 ★ | Pointer model beats the same-size no-memory model by ≥ 20 points on 2- and 3-hop questions | The core claim: pointers beat memorising |
| C2 | Accuracy at the level of published systems (≥ 95 / 94 / 85 on 1/2/3 hops) | Not just "better than a weak baseline" |
| C3 | One question, end to end, in ≤ 500 µs on a CPU | Fast enough to feed a main model |
| C4 | Rewire 10% of the movies' facts: keep ≥ 90% of accuracy with zero retraining | Knowledge you can edit, not retrain |
| C5 | Weights + store ≤ 50 MB, GPU training ≤ 10 min | Project budget |
| N1 | **Negative control:** scramble where the pointers go. It must FAIL. | Proves the router uses the memory |

Also measured (no effect on the verdict): an untrained router with the real store, the router with
an empty store, a router that never saw a 3-hop question, and a router trained on 1% of the data.

## What happened

**Verdict: INCONCLUSIVE (3 of 5).** The big claim held. Two side targets missed, one of them by a hair.

- **Pointers crush memorising.** On 2-hop questions the router scored 97.6-99.9% and the same-size
  no-memory model scored 10%. Even with 4x the bytes, the no-memory model only remembered about a
  quarter of the facts it had been shown. Storing "how to connect" is far cheaper than storing "what is".
- **Edit the dictionary, the answers change.** We rewired 1,642 movies (new directors, years, actors).
  With no retraining, the pointer model got 99.6% of the changed questions right. The no-memory model
  got 6%; it kept giving the old answers.
- **3-hop missed (82.8 vs 85).** The router was choosing the right three pointers. The problem was the
  walk: for "who directed movies that share actors with [Heat]", it went Heat → its actors → back to
  Heat → Heat's director, and those loops outvoted the real answers. One rule, "don't walk back through
  the starting point", lifts the same routers to 98.9-100% with no retraining. We found this *after*
  the pre-registered run, so it doesn't change the verdict. It's a follow-up, honestly labelled.
- **Speed missed by 10% (549 µs vs 500).** The run shared Fortran with another job. The router itself is
  ~270 µs per question in a batch; Python overhead is most of the rest.
- **1% of the data was enough.** A router trained on ~3,300 questions matched one trained on 330,000.
  That's the "compact training on first principles" idea in miniature: there are only a few dozen
  question shapes, and once the router knows them, the dictionary does the rest.
- **It didn't learn to count past what it saw.** A router trained on 1- and 2-hop questions scored 0%
  on 3-hop. It learned "how many hops" as a fixed choice, not as a rule it can extend. That's the
  honest limit of generalisation here.

## Your question: nanoseconds or microseconds?

Short answer: **one pointer is nanoseconds; one question is microseconds; a GPU only gets to
nanoseconds per question when you batch thousands of them.**

| Operation | Typical time | What limits it |
|---|---|---|
| Read one value already in CPU cache | ~1 ns | The speed of the chip |
| Read one value from main memory (RAM) | ~80-100 ns | Memory latency: the data has to travel |
| Python dictionary lookup | ~30-60 ns | Python's own overhead |
| One pointer in our store, called from Python | ~0.3-1 µs | Python call overhead, not the memory |
| The same pointer, 1M at once in NumPy | a few ns each | Memory bandwidth |
| Start one GPU operation ("kernel launch") | ~3-10 µs | Driver + CPU→GPU command queue |
| Copy a number to the GPU and back over PCIe | ~10-30 µs | The PCIe bus round trip |
| A small transformer on the GPU, one question | ~0.3-1 ms | ~50 kernel launches in a row |
| The same transformer, 4,096 questions per call | ~100s of ns per question | Launch cost shared by 4,096 |

**Measured on Fortran** (GPU shared with another job, so GPU numbers are pessimistic): one GPU kernel
launch 17 µs, a PCIe round trip 25 µs, the router on the GPU for one question 3.5 ms (vs 0.55 ms on the
CPU), and **0.08 ns per pointer** when the GPU does 16 million lookups in one go. One pointer from
Python: 220 ns. The same lookup vectorised on the CPU: 6.8 ns.

So for **one** question at a time, the CPU wins: the router is a few small matrix multiplies and
the pointers are array reads, all in a few hundred microseconds or less, with no trip to the GPU.
The RTX 4080 SUPER wins when you send it thousands of questions at once, which is how you'd feed a
main model that is itself on the GPU. "Nanosecond" speed is real for the lookups; the thinking step
(the router) is what costs microseconds. `bench_gpu.py` measures every GPU row of this table on
Fortran, and `run.py` measures the CPU rows.

## Related work, and how this differs

- **kNN-LM** (Khandelwal et al., 2020) and **RETRO** (Borgeaud et al., 2022): a language model plus a
  huge lookup table of text. They retrieve by similarity search, then read text. We retrieve by
  following typed pointers, which is exact and constant time, but needs structured facts.
- **Memorizing Transformers** (Wu et al., 2022): attention over a kNN memory of past keys/values.
  Same spirit (memory outside the weights), but the memory is raw activations, not facts.
- **Product-key memory** (Lample et al., 2019): a huge learned key-value table with fast lookup inside
  a transformer layer. The memory is learned; ours is an editable dictionary.
- **Hash and Bloom embeddings** (Svenstrup et al., 2017; spaCy): map many items to few slots with
  hashing. Our "linking" from a name to a node is one hash lookup.
- **Knowledge-graph QA**: KV-MemNN (Miller et al., 2016), GraftNet (Sun et al., 2018), EmbedKGQA
  (Saxena et al., 2020), NSM (He et al., 2021) and **TransferNet** (Shi et al., 2021). TransferNet is
  the closest: it also learns which relations to follow, softly, from answers only, and reaches
  ~100% on MetaQA. Our twist is the budget framing: a router with no entity knowledge at all, hard
  constant-time pointers at inference, a same-size no-memory control, latency, and live editing.
- **ChatGPT on MetaQA** (StructGPT paper, 2023): with no knowledge base, 31% on 2-hop and 43% on
  3-hop. With a knowledge-base tool (StructGPT), 94% and 80%. That's a 100 GB-class model.

## Applications

- On-device assistants: a tiny router + the user's own data (contacts, files, calendar) as a pointer
  store. Change the data, the answers change, no retraining, no cloud.
- A "fact feeder" for a bigger model: hand it the right entities in microseconds, so it doesn't
  hallucinate them (that's what RAG does, but with exact pointers instead of fuzzy search).
- Mobile apps: a 1-2 MB router + a few MB of structured data fits easily in an app bundle.

## Limits worth knowing

- MetaQA marks the name in brackets, so linking is free. Real questions need a name-finder.
- MetaQA has a small, fixed set of question templates, which makes the router's job easier than
  open-ended questions.
- The store must be structured (things + typed links). Free text needs an extraction step first.

## Further reading

- Shi et al., *TransferNet: An Effective and Transparent Framework for Multi-hop QA over Relation Graph*, EMNLP 2021.
- Zhang et al., *Variational Reasoning for Question Answering with Knowledge Graph* (MetaQA), AAAI 2018.
- Khandelwal et al., *Generalization through Memorization: Nearest Neighbor Language Models*, ICLR 2020.
- Borgeaud et al., *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO), ICML 2022.
- Lample et al., *Large Memory Layers with Product Keys*, NeurIPS 2019.
- Wu et al., *Memorizing Transformers*, ICLR 2022.
- Allen-Zhu & Li, *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws* (≈2 bits of knowledge per parameter), 2024.
