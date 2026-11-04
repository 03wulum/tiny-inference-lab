"""Fast checks (~3 s) for experiment 03."""
import numpy as np

import run


def test_cache_matches_experiment_01():
    # same generator + seed as exp 01, so the caches are the ones that failed there
    import importlib.util, pathlib
    p = pathlib.Path(run.__file__).parent.parent / "01-turboquant-kv-cache" / "run.py"
    src = p.read_text()
    assert "np.random.default_rng(3000 + ds)" in src
    Q, K, V = run.synthetic_kv(256, 128, 8, np.random.default_rng(3000))
    Q2, K2, V2 = run.synthetic_kv(256, 128, 8, np.random.default_rng(3000))
    assert np.array_equal(K, K2)


def test_calibration_queries_are_held_out():
    _, K, _ = run.synthetic_kv(512, 128, 8, np.random.default_rng(3000))
    Qc = run.calib_queries(K, 64, 0)
    Qe, _, _ = run.synthetic_kv(512, 128, 64, np.random.default_rng(3000))
    assert not np.allclose(Qc, Qe)


def test_selection_orders_candidates():
    rows = run.run_cache(512, 0, repeats=1, n_calib=32)
    r = rows[0]
    assert r["kl_oracle"] <= min(r["kl_best"], r["kl_random"], r["kl_worst"]) + 1e-12
    assert r["kl_int"] > 0


def test_attention_rows_sum_to_one():
    rng = np.random.default_rng(0)
    P = run.attn_probs(rng.standard_normal((4, 16)), rng.standard_normal((10, 16)))
    assert np.allclose(P.sum(1), 1)
