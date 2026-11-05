# Experiment 05 in plain language

Experiment 04 showed that a tiny "librarian" model works. It stores no facts, only which links to follow through a dictionary of 135k movie facts, and it crushed a same-size "memoriser". But it missed two of its targets. On 3-hop questions the walk looped back to its starting point. And one question took 549 µs against a 500 µs bar. The full story is in the [04 explainer](../04-pointer-memory/EXPLAINER.md).

Experiment 05 asks Obi's follow-up questions:

1. **Is it conclusive once the loop fix is part of the method from day one?** We retrain from scratch with the rule "never walk back through the start". It applies in training as well as in answering. The bars are the same as before.
2. **What happens as questions get harder?** We build questions with 1 to 5 jumps from the same facts, like "what is the director of the movies starring the actors in [Heat]". Training only ever shows 1-3 jumps, so 4 and 5 jumps test whether a model can *extend* what it learned.
3. **Does it beat every kind of rival, at every level?** The rivals are a memoriser at the same size and at 4x the size, and a knowledge-graph embedding model. That last one is the EmbedKGQA style: every movie and person becomes a point in space, and a question becomes a direction to move.
4. **Which tweak beats which rival?** We try five router flavors, from simplest to richest:
   - *tiny*: 18k numbers
   - *order-blind*: reads the words as a bag, ignoring their order
   - *fixed-hop without the loop fix*
   - *fixed-hop with the loop fix*
   - *recurrent*: one "jump" step it can repeat, which stops when no unused link words remain

   For each level and rival, we report the simplest flavor that wins by 10+ points.

The recurrent flavor is the bet on generalising. It reads the question from the name outward, one link phrase per jump. It marks each phrase as used, and stops when nothing is left. That is a rule that should work for 5 jumps even though it only ever practised 3. If it scores 80% or more on 4 and 5 jumps, the idea "learn a few first principles, then compose them" has real support.

Glossary and background: [Pointer network, explained](https://claude.ai/artifact/48GEcpDYzSJYxm9Cv4VPng).
