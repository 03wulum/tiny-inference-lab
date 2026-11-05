"""Correctness checks: store walks, the compositional generator, and torch <-> NumPy parity."""
import numpy as np
import pytest

import data as D
import synth
from criteria import evaluate
from store import PointerStore

TRIPLES = [
    ("Heat", "directed_by", "Michael Mann"), ("Heat", "starred_actors", "Al Pacino"),
    ("Heat", "starred_actors", "Robert De Niro"), ("Collateral", "directed_by", "Michael Mann"),
    ("Collateral", "starred_actors", "Tom Cruise"), ("Casino", "starred_actors", "Robert De Niro"),
    ("Casino", "directed_by", "Martin Scorsese"), ("Heat", "release_year", "1995"),
    ("Casino", "release_year", "1995"), ("Collateral", "has_genre", "Thriller"),
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


def test_non_backtracking(kb, store):
    path = [R["starred_actors"] + INV, R["starred_actors"], R["starred_actors"] + INV]
    plain, _ = store.follow(kb.id["Robert De Niro"], path)
    nb, _ = store.follow(kb.id["Robert De Niro"], path, avoid_topic=True)
    assert "Casino" in names(kb, plain) and names(kb, nb) == ["Heat"]


def test_question_text_reads_outside_in():
    path = [R["starred_actors"] + INV, R["directed_by"]]
    assert synth.question_text(path, "X") == "what is the director of the movies starring [X]"


def test_generator_is_typed_and_consistent(kb, store):
    qs = synth.generate(kb, store, 2, 20, "train") + synth.generate(kb, store, 2, 20, "test")
    assert qs
    for q in qs:
        ents, _ = store.follow(q["topic"], q["path"], avoid_topic=True)
        assert sorted(set(ents.tolist()) - {q["topic"]}) == q["answers"]
        assert 1 <= len(q["answers"]) <= synth.MAX_ANSWERS


def test_train_test_topics_disjoint(kb, store):
    tr = {q["topic"] for q in synth.generate(kb, store, 1, 50, "train")}
    te = {q["topic"] for q in synth.generate(kb, store, 1, 50, "test")}
    assert all(t % 10 != 0 for t in tr) and all(t % 10 == 0 for t in te)


def test_verdict_logic():
    good = dict(c1_min_margin_pts=1, c2_min_gap_pts=30, c3_rec_min_45=85, c4_e2e_us=100, c5_budget_ok=1.0,
                n1_random_2hop=1.0)
    assert evaluate(good)["verdict"] == "PASS"
    assert evaluate({**good, "c1_min_margin_pts": -1})["verdict"] == "FAIL"
    assert evaluate({**good, "c3_rec_min_45": 10})["verdict"] == "INCONCLUSIVE"
    assert evaluate({**good, "n1_random_2hop": 99})["verdict"] == "INCONCLUSIVE"


IDS = np.array([[2, 5, 6, 3, 7, 0, 0], [2, 7, 4, 3, 0, 0, 0]])   # 3 = entity slot


def _np(m):
    return {k: v.detach().numpy() for k, v in m.state_dict().items()}


def test_parity_routers():
    torch = pytest.importorskip("torch")
    import models_np as N
    import models_torch as M
    torch.manual_seed(0)
    t_ids = torch.tensor(IDS)
    for tm, nm, T in [(M.Router(12, d=32, max_len=8), N.NpRouter, 3), (M.BowRouter(12, d=16), N.NpBowRouter, 3)]:
        tm.eval()
        rp, hp = tm(t_ids, T)
        rn, hn = nm(_np(tm)).probs(IDS)
        assert np.allclose(rp.detach().numpy(), rn, atol=1e-5) and np.allclose(hp.detach().numpy(), hn, atol=1e-5)
    rec = M.RecRouter(12, 3, d=32).eval()
    rp, hp = rec(t_ids, 4)
    rn, halt = N.NpRecRouter(_np(rec), 3).run(IDS, 4)
    assert np.allclose(rp.detach().numpy(), rn, atol=1e-5)
    hp_np = M.halt_to_hop_probs(torch.tensor(halt)).numpy()
    assert np.allclose(hp.detach().numpy(), hp_np, atol=1e-5)


def test_parity_rivals():
    torch = pytest.importorskip("torch")
    import models_np as N
    import models_torch as M
    torch.manual_seed(0)
    topic = np.array([4, 9])
    pm = M.ParamQA(12, 20, 3, d=16, max_len=8).eval()
    assert np.allclose(pm(torch.tensor(IDS), torch.tensor(topic)).detach().numpy(),
                       N.NpParamQA(_np(pm), 3).logits(IDS, topic), atol=1e-4)
    km = M.KGEQA(12, 20, k=8, d=16, max_len=8).eval()
    assert np.allclose(km(torch.tensor(IDS), torch.tensor(topic)).detach().numpy(),
                       N.NpKGEQA(_np(km)).logits(IDS, topic), atol=1e-4)


def test_soft_walk_matches_hard_walk(kb, store):
    torch = pytest.importorskip("torch")
    from models_torch import soft_follow
    path = [R["starred_actors"] + INV, R["starred_actors"], R["starred_actors"] + INV]
    rel_p = torch.zeros(1, 3, D.N_DIR)
    for t, d in enumerate(path):
        rel_p[0, t, d] = 1.0
    e = [torch.tensor(x.astype(np.int64)) for x in (store.e_head, store.e_dir, store.e_tail)]
    s = soft_follow(torch.tensor([kb.id["Robert De Niro"]]), rel_p, torch.tensor([[0.0, 0.0, 1.0]]), *e,
                    kb.n_entities, avoid_topic=True)[0]
    ents, _ = store.follow(kb.id["Robert De Niro"], path, avoid_topic=True)
    assert set(torch.nonzero(s).flatten().tolist()) == set(ents.tolist())
