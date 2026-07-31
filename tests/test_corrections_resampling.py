"""Tests for the resampling-based corrections.

These have no closed-form worked example in the source papers, so the anchors
are the published *properties*:

* White (2000) and Hansen (2005): both control the size of the test of "no
  candidate beats the benchmark" under the complete null;
* Hansen (2005), Section 3: the three SPA p-values are ordered
  ``p_lower <= p_consistent <= p_upper``, and the upper one is the version
  that uses White's recentring, so the Reality Check p-value tracks it;
* Hansen (2005), Section 4: adding candidates that are clearly worse than the
  benchmark raises the Reality Check p-value but leaves SPA roughly alone.
  This is the paper's central criticism of the Reality Check and the clearest
  behavioural test of a correct implementation;
* Romano and Wolf (2005): the step-down procedure rejects at least as many
  hypotheses as the single-step maximum, and controls FWER.
"""

from __future__ import annotations

import numpy as np
import pytest

from bzoo.corrections import bootstrap_tests as bt
from bzoo.corrections import fwer


def _null_panel(rng, t=250, k=10, rho=0.5):
    """Zero-mean correlated differentials: the complete null."""
    common = rng.standard_normal((t, 1))
    idio = rng.standard_normal((t, k))
    return np.sqrt(rho) * common + np.sqrt(1 - rho) * idio


def test_reality_check_size_under_the_complete_null():
    rng = np.random.default_rng(0)
    n_rep = 300
    rejects = 0
    for i in range(n_rep):
        d = _null_panel(rng)
        r = bt.white_reality_check(d, n_boot=200, scheme="iid", seed=i)
        rejects += r.extra["p_value"] <= 0.05
    # Conservative by construction, so we only bound it from above.
    assert rejects / n_rep <= 0.10


def test_spa_size_under_the_complete_null():
    rng = np.random.default_rng(1)
    n_rep = 300
    rejects = 0
    for i in range(n_rep):
        d = _null_panel(rng)
        r = bt.hansen_spa(d, n_boot=200, scheme="iid", seed=i)
        rejects += r.extra["p_value"] <= 0.05
    assert rejects / n_rep <= 0.12


def test_spa_pvalue_ordering():
    rng = np.random.default_rng(2)
    for i in range(20):
        d = _null_panel(rng, k=15)
        d[:, 0] += 0.1  # one mildly good candidate
        r = bt.hansen_spa(d, n_boot=400, scheme="iid", seed=i)
        assert r.extra["p_lower"] <= r.extra["p_value"] + 1e-12
        assert r.extra["p_value"] <= r.extra["p_upper"] + 1e-12


def test_adding_bad_candidates_hurts_the_reality_check_more_than_spa():
    """Hansen's (2005) main point about the Reality Check, reproduced."""
    rng = np.random.default_rng(3)
    t = 400
    # A borderline good model: mean 0.055 with sd 0.5 over 400 periods is a
    # t-statistic near 2.2, so the p-values stay away from the resampling
    # floor and the comparison is informative.
    good = rng.standard_normal((t, 1)) * 0.5 + 0.055
    bad = rng.standard_normal((t, 40)) * 0.5 - 1.0  # forty clearly worse ones
    both = np.hstack([good, bad])

    rc_small = bt.white_reality_check(good, n_boot=2000, scheme="iid", seed=0)
    rc_large = bt.white_reality_check(both, n_boot=2000, scheme="iid", seed=0)
    spa_small = bt.hansen_spa(good, n_boot=2000, scheme="iid", seed=0)
    spa_large = bt.hansen_spa(both, n_boot=2000, scheme="iid", seed=0)

    rc_damage = rc_large.extra["p_value"] - rc_small.extra["p_value"]
    spa_damage = spa_large.extra["p_value"] - spa_small.extra["p_value"]
    assert rc_damage > spa_damage
    # The bad models are dropped from the SPA null distribution entirely, so
    # its p-value barely moves.
    assert spa_damage < 0.02


def test_reality_check_detects_a_large_real_effect():
    rng = np.random.default_rng(4)
    d = rng.standard_normal((400, 8)) * 0.5
    d[:, 3] += 0.4
    r = bt.white_reality_check(d, n_boot=1000, scheme="iid", seed=0)
    assert r.extra["p_value"] < 0.01
    assert r.reject[3] and r.n_reject == 1


def test_romano_wolf_is_at_least_as_powerful_as_the_single_step_maximum():
    rng = np.random.default_rng(5)
    d = rng.standard_normal((300, 12)) * 0.5
    d[:, [0, 1, 2]] += 0.35
    obs, cent, _ = bt.bootstrap_centred_matrix(
        d, n_boot=2000, scheme="iid", seed=0, studentised=True
    )
    rw = fwer.romano_wolf(obs, cent, alpha=0.05)
    # Single-step: only the largest statistic, compared with the same
    # critical value, which is exactly the first step of the step-down walk.
    single = int(obs.max() > rw.critical_value)
    assert rw.n_reject >= single
    assert rw.n_reject >= 1


def test_romano_wolf_controls_fwer_under_the_complete_null():
    rng = np.random.default_rng(6)
    n_rep = 300
    bad = 0
    for i in range(n_rep):
        d = _null_panel(rng, t=200, k=10)
        obs, cent, _ = bt.bootstrap_centred_matrix(
            d, n_boot=300, scheme="iid", seed=i, studentised=True
        )
        bad += fwer.romano_wolf(obs, cent, alpha=0.05).n_reject > 0
    assert bad / n_rep <= 0.10


def test_westfall_young_maxt_controls_fwer_under_the_complete_null():
    from bzoo.resample.permutation import sign_flip_replicates, studentise

    rng = np.random.default_rng(7)
    n_rep = 300
    bad = 0
    for _ in range(n_rep):
        d = _null_panel(rng, t=200, k=10)
        rep = sign_flip_replicates(d, 300, rng)
        rep_s, obs_s = studentise(rep, d.mean(axis=0))
        bad += fwer.westfall_young_maxt(obs_s, rep_s, alpha=0.05).n_reject > 0
    assert bad / n_rep <= 0.10


def test_westfall_young_minp_controls_fwer_under_the_complete_null():
    from scipy import stats

    rng = np.random.default_rng(8)
    n_rep = 200
    n_perm, t, k = 300, 200, 8
    bad = 0
    for _ in range(n_rep):
        d = _null_panel(rng, t=t, k=k)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, t))
        # One-sided p-values, recomputed on every permutation replicate.
        def pvals(x):
            m = x.mean(axis=0)
            se = x.std(axis=0, ddof=1) / np.sqrt(t)
            return stats.norm.sf(m / se)

        p_obs = pvals(d)
        p_perm = np.array([pvals(signs[b][:, None] * d) for b in range(n_perm)])
        bad += fwer.westfall_young_minp(p_obs, p_perm, alpha=0.05).n_reject > 0
    assert bad / n_rep <= 0.10


def test_westfall_young_finds_a_planted_effect():
    from bzoo.resample.permutation import sign_flip_replicates, studentise

    rng = np.random.default_rng(9)
    d = rng.standard_normal((400, 10)) * 0.5
    d[:, 7] += 0.35
    rep = sign_flip_replicates(d, 2000, rng)
    rep_s, obs_s = studentise(rep, d.mean(axis=0))
    r = fwer.westfall_young_maxt(obs_s, rep_s, alpha=0.05)
    assert r.reject[7]


def test_stepdown_adjusted_pvalues_are_monotone():
    rng = np.random.default_rng(10)
    d = rng.standard_normal((300, 15)) * 0.5
    d[:, :4] += 0.3
    obs, cent, _ = bt.bootstrap_centred_matrix(d, n_boot=1000, scheme="iid", seed=0)
    r = fwer.romano_wolf(obs, cent)
    order = np.argsort(-obs)
    assert np.all(np.diff(r.adjusted_p[order]) >= -1e-12)


def test_resample_matrix_validation():
    rng = np.random.default_rng(11)
    obs = rng.standard_normal(5)
    with pytest.raises(ValueError):
        fwer.romano_wolf(obs, rng.standard_normal((100, 4)))
    with pytest.raises(ValueError):
        fwer.romano_wolf(obs, rng.standard_normal(100))
    with pytest.raises(ValueError):
        fwer.romano_wolf(obs, rng.standard_normal((5, 5)))


def test_block_length_is_chosen_when_not_supplied():
    rng = np.random.default_rng(12)
    d = rng.standard_normal((300, 5))
    r = bt.white_reality_check(d, n_boot=200, scheme="stationary", seed=0)
    assert r.extra["block_length"] >= 1.0
