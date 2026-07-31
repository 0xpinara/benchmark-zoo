"""Tests for the p-value corrections, against published worked examples.

The main reference case is the worked example in Benjamini and Hochberg
(1995), Section 2: the fifteen p-values from the multiple-endpoint analysis
of Neuhaus et al.  At level 0.05 the single-step and step-down family-wise
procedures reject three of them and the Benjamini-Hochberg step-up procedure
rejects four.  Reproducing both counts from one p-value vector exercises the
ordering logic of every procedure in the module at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from bzoo.corrections import fdr, fwer

# Benjamini and Hochberg (1995), the 15 p-values of their worked example.
BH1995_P = np.array(
    [
        0.0001,
        0.0004,
        0.0019,
        0.0095,
        0.0201,
        0.0278,
        0.0298,
        0.0344,
        0.0459,
        0.3240,
        0.4262,
        0.5719,
        0.6528,
        0.7590,
        1.0000,
    ]
)


def test_bonferroni_on_bh1995_example():
    r = fwer.bonferroni(BH1995_P, alpha=0.05)
    # 0.05/15 = 0.00333, so 0.0001, 0.0004 and 0.0019 pass and 0.0095 does not.
    assert r.n_reject == 3
    assert r.critical_value == pytest.approx(0.05 / 15)


def test_holm_on_bh1995_example():
    r = fwer.holm(BH1995_P, alpha=0.05)
    # Step-down bounds are 0.0033, 0.0036, 0.0038, 0.0042; the fourth
    # p-value 0.0095 stops the walk.
    assert r.n_reject == 3


def test_sidak_on_bh1995_example():
    assert fwer.sidak(BH1995_P, alpha=0.05).n_reject == 3


def test_bh_on_bh1995_example():
    r = fdr.benjamini_hochberg(BH1995_P, alpha=0.05)
    assert r.n_reject == 4
    # The paper's own statement of the rule: reject up to the largest i with
    # p_(i) <= i * q / m.  For i = 4 that bound is 0.0133.
    assert r.critical_value == pytest.approx(0.0095)


def test_bh_is_at_least_as_powerful_as_holm():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.beta(0.3, 1.0, size=50)
        assert (
            fdr.benjamini_hochberg(p).n_reject >= fwer.holm(p).n_reject
        )


def test_holm_is_at_least_as_powerful_as_bonferroni():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.beta(0.3, 1.0, size=50)
        assert fwer.holm(p).n_reject >= fwer.bonferroni(p).n_reject


def test_by_is_bh_with_harmonic_penalty():
    p = BH1995_P
    by = fdr.benjamini_yekutieli(p, alpha=0.05)
    c15 = sum(1.0 / i for i in range(1, 16))
    bh_scaled = fdr.benjamini_hochberg(p, alpha=0.05 / c15)
    assert by.n_reject == bh_scaled.n_reject
    assert by.extra["harmonic_c"] == pytest.approx(c15)
    assert by.n_reject <= fdr.benjamini_hochberg(p, 0.05).n_reject


def test_adjusted_pvalues_are_monotone_in_the_pvalue_order():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=200)
    order = np.argsort(p)
    for res in (
        fwer.bonferroni(p),
        fwer.sidak(p),
        fwer.holm(p),
        fdr.benjamini_hochberg(p),
        fdr.benjamini_yekutieli(p),
        fdr.storey_qvalues(p),
    ):
        adj = res.adjusted_p[order]
        assert np.all(np.diff(adj) >= -1e-12), res.method


def test_adjusted_pvalue_threshold_matches_reject_flag():
    """reject == (adjusted_p <= alpha) must hold for every procedure that
    reports adjusted p-values, otherwise the two ways of reading a table
    disagree."""
    rng = np.random.default_rng(3)
    p = rng.beta(0.4, 1.0, size=300)
    alpha = 0.05
    for res in (
        fwer.bonferroni(p, alpha),
        fwer.sidak(p, alpha),
        fwer.holm(p, alpha),
        fdr.benjamini_hochberg(p, alpha),
        fdr.benjamini_yekutieli(p, alpha),
        fdr.storey_qvalues(p, alpha),
    ):
        assert np.array_equal(res.reject, res.adjusted_p <= alpha + 1e-15), res.method


def test_sidak_is_slightly_more_powerful_than_bonferroni():
    p = np.array([0.05 / 15 * 1.005] * 3 + [0.9] * 12)
    assert fwer.sidak(p).n_reject >= fwer.bonferroni(p).n_reject


def test_pi0_is_near_one_for_pure_noise():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=5000)
    pi0, info = fdr.estimate_pi0(p, seed=0)
    assert 0.9 <= pi0 <= 1.0
    assert 0.05 <= info["lambda"] <= 0.95


def test_pi0_drops_when_half_the_hypotheses_are_strong():
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(size=2500), rng.beta(0.1, 8.0, size=2500)])
    pi0, _ = fdr.estimate_pi0(p, seed=0)
    assert 0.4 <= pi0 <= 0.65


def test_storey_reduces_to_bh_when_pi0_is_one():
    p = BH1995_P
    st = fdr.storey_qvalues(p, alpha=0.05, pi0=1.0)
    bh = fdr.benjamini_hochberg(p, alpha=0.05)
    assert np.allclose(st.adjusted_p, bh.adjusted_p)
    assert st.n_reject == bh.n_reject


def test_fwer_control_under_a_complete_null():
    """Bonferroni and Holm must reject in at most alpha of the replications
    when every hypothesis is true."""
    rng = np.random.default_rng(6)
    n_rep, k, alpha = 2000, 20, 0.05
    bad_bonf = bad_holm = 0
    for _ in range(n_rep):
        p = rng.uniform(size=k)
        bad_bonf += fwer.bonferroni(p, alpha).n_reject > 0
        bad_holm += fwer.holm(p, alpha).n_reject > 0
    # Binomial standard error at alpha=0.05 over 2000 draws is 0.005.
    assert bad_bonf / n_rep < alpha + 0.015
    assert bad_holm / n_rep < alpha + 0.015


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        fwer.bonferroni(np.array([0.1, 1.2]))
    with pytest.raises(ValueError):
        fwer.bonferroni(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        fwer.bonferroni(np.array([]))
    with pytest.raises(ValueError):
        fwer.bonferroni(np.array([0.1]), alpha=0.0)
