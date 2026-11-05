"""
Compositional 1-5 hop questions over the same MetaQA movie knowledge base ("complexity levels").

MetaQA stops at 3 hops. To test 4 and 5 hops we compose questions from one phrase per pointer type:

  path [movies starring X -> director of] = "what is the director of the movies starring [X]"

Gold answers = the non-backtracking walk (never passing back through the topic entity), and we keep
only questions whose answer set has 1-50 entities, so they stay answerable and checkable.
Deterministic given the seed; no download beyond the KB.
"""
from __future__ import annotations

import numpy as np

import data as D
from store import PointerStore

# One phrase per pointer type. Forward = movie -> attribute, inverse = attribute -> movies.
PHRASE = {
    "directed_by": ("the director of", "the movies directed by"),
    "written_by": ("the writer of", "the movies written by"),
    "starred_actors": ("the actors in", "the movies starring"),
    "release_year": ("the release year of", "the movies released in"),
    "in_language": ("the language of", "the movies in the language"),
    "has_genre": ("the genre of", "the movies in the genre"),
}
DIRS = [D.RELATIONS.index(r) for r in PHRASE] + [D.RELATIONS.index(r) + len(D.RELATIONS) for r in PHRASE]
MAX_ANSWERS = 50


def phrase(d: int) -> str:
    R = len(D.RELATIONS)
    name = D.RELATIONS[d % R]
    return PHRASE[name][0 if d < R else 1]


def question_text(path, topic_name):
    return "what is " + " ".join(phrase(d) for d in reversed(path)) + f" [{topic_name}]"


def generate(kb: D.KB, store: PointerStore, hops: int, n: int, split: str):
    """n questions with exactly `hops` pointer steps. Random non-backtracking walks: at every step
    pick a pointer type that leads somewhere new; keep walks whose final answer set is 1-50 entities."""
    seed = {"train": 1, "test": 2}[split]
    rng = np.random.default_rng(seed * 100 + hops)
    out, seen, tries = [], set(), 0
    n_ent = kb.n_entities
    R = len(D.RELATIONS)
    is_movie = np.zeros(n_ent, bool)
    is_movie[kb.h] = True
    fwd, inv = [d for d in DIRS if d < R], [d for d in DIRS if d >= R]
    while len(out) < n and tries < n * 400:
        tries += 1
        topic = int(rng.integers(n_ent))
        if (topic % 10 == 0) != (split == "test"):     # test topics (id % 10 == 0) never appear in training
            continue
        ents, w, path = np.array([topic], np.int32), np.ones(1, np.float32), []
        ok, movie = True, bool(is_movie[topic])
        for _ in range(hops):
            # typed walk: movies take forward pointers, people/years/genres take inverse ones
            opts = list(rng.permutation(fwd if movie else inv))
            for d in opts:
                e2, w2 = store.hop(ents, w, int(d))
                keep = e2 != topic
                if keep.any():
                    ents, w = e2[keep], w2[keep]
                    path.append(int(d))
                    movie = not movie
                    break
            else:
                ok = False
                break
        if not ok or not (1 <= len(ents) <= MAX_ANSWERS):
            continue
        key = (topic, tuple(path))
        if key in seen:
            continue
        seen.add(key)
        q = question_text(path, kb.names[topic])
        out.append(dict(q=q, tokens=D.tokenize(q), topic=topic, answers=sorted(ents.tolist()), hop=hops,
                        path=tuple(path)))
    return out
