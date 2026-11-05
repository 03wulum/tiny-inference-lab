"""
MetaQA data: the WikiMovies knowledge base (135k facts) and 1/2/3-hop questions.

Source: MetaQA (Zhang et al., AAAI 2018), released under a Creative Commons license, built on
WikiMovies (CC-BY 3.0). We download the raw text files (~68 MB) from a pinned Hugging Face mirror
into data/ (git-ignored; `run.py --clean` deletes them).

Everything here is plain Python + NumPy so CI and laptops can run it without torch.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
MIRROR = "https://huggingface.co/datasets/camazlucas/MetaQA/resolve/{rev}/{path}"
REVISION = "f83854092893610686ab19f2038c039afa9bc643"   # pinned commit of the mirror
FILES = ["kb/kb.txt"] + [f"{h}-hop/vanilla/qa_{s}.txt" for h in (1, 2, 3) for s in ("train", "dev", "test")]

RELATIONS = ["directed_by", "written_by", "starred_actors", "release_year", "in_language",
             "has_tags", "has_genre", "has_imdb_votes", "has_imdb_rating"]
# Each relation is stored in both directions: id r = forward, r + 9 = inverse ("movies directed by X").
N_DIR = 2 * len(RELATIONS)
DIR_NAMES = RELATIONS + [f"inv_{r}" for r in RELATIONS]
ENTITY_SLOT = "<E>"


def download(files=FILES, root: Path = DATA):
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = root / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(MIRROR.format(rev=REVISION, path=f), tmp)
        tmp.rename(dst)
    return root


# ---------------------------------------------------------------- knowledge base -------
class KB:
    """Entity names <-> ids, plus the raw forward triples (head, relation, tail) as int arrays."""

    def __init__(self, triples: list[tuple[str, str, str]]):
        names = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
        self.names = names
        self.id = {n: i for i, n in enumerate(names)}
        self.lower = {}
        for n in names:                      # case-insensitive fallback for entity linking
            self.lower.setdefault(n.lower(), self.id[n])
        rid = {r: i for i, r in enumerate(RELATIONS)}
        self.h = np.array([self.id[h] for h, _, _ in triples], dtype=np.int32)
        self.r = np.array([rid[r] for _, r, _ in triples], dtype=np.int8)
        self.t = np.array([self.id[t] for _, _, t in triples], dtype=np.int32)

    @property
    def n_entities(self):
        return len(self.names)

    def link(self, mention: str) -> int:
        """Entity linking = one hash-table lookup on the bracketed name (MetaQA marks the topic entity)."""
        i = self.id.get(mention)
        return self.lower.get(mention.lower(), -1) if i is None else i

    @classmethod
    def load(cls, path: Path):
        triples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            p = line.split("|")
            if len(p) == 3:
                triples.append((p[0], p[1], p[2]))
        return cls(triples)

    @classmethod
    def from_triples(cls, triples):
        return cls(list(triples))


# ---------------------------------------------------------------- questions ------------
_BRACKET = re.compile(r"\[(.+?)\]")


def tokenize(question: str) -> list[str]:
    """Lowercased words with the topic entity replaced by a slot: the router never sees WHICH entity."""
    return _BRACKET.sub(f" {ENTITY_SLOT} ", question).lower().replace(ENTITY_SLOT.lower(), ENTITY_SLOT).split()


def parse_qa_line(line: str):
    q, a = line.rstrip("\n").split("\t")
    m = _BRACKET.search(q)
    return q, m.group(1), a.split("|")


def load_questions(kb: KB, hop: int, split: str, root: Path = DATA, limit: int | None = None):
    """Returns list of dicts: tokens, topic id, answer ids (topic removed), hop."""
    out = []
    lines = (root / f"{hop}-hop/vanilla/qa_{split}.txt").read_text(encoding="utf-8").splitlines()
    for line in lines[:limit]:
        q, mention, answers = parse_qa_line(line)
        topic = kb.link(mention)
        ans = sorted({kb.id[a] for a in answers if a in kb.id} - {topic})
        if topic < 0 or not ans:
            continue
        out.append(dict(q=q, tokens=tokenize(q), topic=topic, answers=ans, hop=hop))
    return out


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, words):
        self.itos = ["<pad>", "<unk>", "<cls>"] + sorted(set(words) - {"<pad>", "<unk>", "<cls>"})
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.CLS = self.stoi["<cls>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        ids = [self.CLS] + [self.stoi.get(w, self.UNK) for w in tokens][: max_len - 1]
        return ids + [self.PAD] * (max_len - len(ids))

    def batch(self, questions, max_len):
        return np.array([self.encode(q["tokens"], max_len) for q in questions], dtype=np.int64)

    @classmethod
    def build(cls, questions, extra=()):
        return cls([w for q in questions for w in q["tokens"]] + list(extra))
