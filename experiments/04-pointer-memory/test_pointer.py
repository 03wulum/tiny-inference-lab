"""Correctness checks on a tiny hand-made knowledge base. No downloads, no torch needed."""
import numpy as np
import pytest

import data as D
from criteria import evaluate
from store import PointerStore

TRIPLES = [
    ("Heat", "directed_by", "Michael Mann"), ("Heat", "starred_actors", "Al Pacino"),
    ("Heat", "starred_actors", "Robert De Niro"), ("Collateral", "directed_by", "Michael Mann"),
    ("Collateral", "starred_actors", "Tom Cruise"), ("Heat", "release_year", "1995"),
    ("Collateral", "release_year", "2004"), ("Casino", "starred_actors", "Robert De Niro"),
    ("Casino", "directed_by", "Martin Scorsese"),
]
R = {r: i for i, r in enumerate(D.RELATIONS)}
INV = len(D.RELATIONS)


@pytest.fixture
def kb():
    return D.KB.from_triples(TRIPLES)


@pytest.fixture
def store(kb):
    return PointerStore.from_kb(kb)


def names(kb, ids):
    return sorted(kb.names[i] for i in ids)


def test_one_hop_forward_and_inverse(kb, store):
    assert names(kb, store.neighbours(kb.id["Heat"], R["starred_actors"])) == ["Al Pacino", "Robert De Niro"]
    assert names(kb, store.neighbours(kb.id["Michael Mann"], R["directed_by"] + INV)) == ["Collateral", "Heat"]
    assert len(store.neighbours(kb.id["Al Pacino"], R["directed_by"])) == 0


def test_multi_hop_counts_paths(kb, store):
    # actors in movies directed by Michael Mann
    ents, w = store.follow(kb.id["Michael Mann"], [R["directed_by"] + INV, R["starred_actors"]])
    assert names(kb, ents) == ["Al Pacino", "Robert De Niro", "Tom Cruise"]
    # directors of De Niro's movies: Mann and Scorsese, one path each
    ents, w = store.follow(kb.id["Robert De Niro"], [R["starred_actors"] + INV, R["directed_by"]])
    assert names(kb, ents) == ["Martin Scorsese", "Michael Mann"] and list(w) == [1.0, 1.0]


def test_answer_excludes_topic(kb, store):
    # co-stars of De Niro: the path also reaches De Niro himself, who must not be the answer
    a = store.answer(kb.id["Robert De Niro"], [R["starred_actors"] + INV, R["starred_actors"]])
    assert kb.names[a] == "Al Pacino"


def test_permuted_store_keeps_shape(store):
    p = store.permuted(0)
    assert np.array_equal(p.indptr, store.indptr)
    assert sorted(p.indices.tolist()) == sorted(store.indices.tolist())


def test_empty_store(kb):
    s = PointerStore.empty(kb.n_entities)
    assert s.answer(kb.id["Heat"], [R["directed_by"]]) == -1


def test_tokenize_hides_entity():
    t = D.tokenize("which person directed the movies starred by [John Krasinski]")
    assert t[-1] == D.ENTITY_SLOT and "john" not in t
    assert D.parse_qa_line("what films did [X] write\tA|B")[1:] == ("X", ["A", "B"])


def test_linking_is_case_insensitive_fallback(kb):
    assert kb.link("heat") == kb.id["Heat"] and kb.link("Heat") == kb.id["Heat"] and kb.link("nope") == -1


def test_router_np_shapes():
    from router_np import NpRouter
    rng = np.random.default_rng(0)
    d, V, n_dir = 16, 10, D.N_DIR
    W = {"enc.tok.weight": rng.normal(size=(V, d)), "enc.pos.weight": rng.normal(size=(8, d)),
         "enc.ln.weight": np.ones(d), "enc.ln.bias": np.zeros(d),
         "hop_emb": rng.normal(size=(3, d)), "rel_emb": rng.normal(size=(n_dir, d)),
         "q.weight": rng.normal(size=(d, 3 * d)), "q.bias": np.zeros(d),
         "rel.weight": rng.normal(size=(n_dir, 2 * d)), "rel.bias": np.zeros(n_dir),
         "n_hops.weight": rng.normal(size=(3, d)), "n_hops.bias": np.zeros(3)}
    r = NpRouter(W)
    ids = np.array([[2, 5, 6, 0], [2, 7, 0, 0]])
    rel_p, hop_p = r.probs(ids)
    assert rel_p.shape == (2, 3, n_dir) and np.allclose(rel_p.sum(-1), 1) and np.allclose(hop_p.sum(-1), 1)
    assert all(1 <= len(p) <= 3 for p in r.paths(ids))


def test_torch_numpy_parity():
    torch = pytest.importorskip("torch")
    from model_torch import ParamQA, Router, soft_follow
    from router_np import NpParamQA, NpRouter
    torch.manual_seed(0)
    m = Router(12, d=32, max_len=8).eval()
    W = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    ids = np.array([[2, 5, 6, 3, 0, 0], [2, 7, 4, 0, 0, 0]])
    rp, hp = m(torch.tensor(ids))
    rn, hn = NpRouter(W).probs(ids)
    assert np.allclose(rp.detach().numpy(), rn, atol=1e-5) and np.allclose(hp.detach().numpy(), hn, atol=1e-5)
    pm = ParamQA(12, 20, 3, d=16, max_len=8).eval()
    Wp = {k: v.detach().numpy() for k, v in pm.state_dict().items()}
    topic = np.array([4, 9])
    lt = pm(torch.tensor(ids), torch.tensor(topic)).detach().numpy()
    assert np.allclose(lt, NpParamQA(Wp, 3).logits(ids, topic), atol=1e-4)


def test_soft_follow_matches_hard_store(kb, store):
    torch = pytest.importorskip("torch")
    from model_torch import soft_follow
    path = [R["directed_by"] + INV, R["starred_actors"]]
    rel_p = torch.zeros(1, 3, D.N_DIR)
    for t, d in enumerate(path):
        rel_p[0, t, d] = 1.0
    hop_p = torch.tensor([[0.0, 1.0, 0.0]])
    e = [torch.tensor(x.astype(np.int64)) for x in (store.e_head, store.e_dir, store.e_tail)]
    s = soft_follow(torch.tensor([kb.id["Michael Mann"]]), rel_p, hop_p, *e, kb.n_entities)[0]
    ents, _ = store.follow(kb.id["Michael Mann"], path)
    assert set(torch.nonzero(s).flatten().tolist()) == set(ents.tolist())


def test_verdict_logic():
    good = dict(c1_min_gap_pts=30, c2_min_margin_pts=1, c3_e2e_us=100, c4_edit_ratio=1.0, c5_budget_ok=1.0,
                n1_random_2hop=3.0)
    assert evaluate(good)["verdict"] == "PASS"
    assert evaluate({**good, "c1_min_gap_pts": 5})["verdict"] == "FAIL"
    assert evaluate({**good, "c3_e2e_us": 900})["verdict"] == "INCONCLUSIVE"
    assert evaluate({**good, "n1_random_2hop": 99})["verdict"] == "INCONCLUSIVE"


def test_non_backtracking_walk(kb, store):
    # De Niro -> his movies -> their actors -> those actors' movies: plain walk revisits De Niro
    path = [R["starred_actors"] + INV, R["starred_actors"], R["starred_actors"] + INV]
    plain, _ = store.follow(kb.id["Robert De Niro"], path)
    nb, _ = store.follow(kb.id["Robert De Niro"], path, avoid_topic=True)
    assert "Casino" in names(kb, plain)          # reached only by walking back through De Niro
    assert names(kb, nb) == ["Heat"]             # via Al Pacino only
