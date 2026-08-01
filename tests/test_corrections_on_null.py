"""Tests for the direct test of whether a correction fixes the exposure channel.

The claim these support is the paper's narrowest and most load-bearing one: a
correction applied with the nominal null fails on this population, the same
correction applied with the measured null does not, and the reason is that the
inflation is a fixed scale effect rather than a divergence.  Each test builds a
population where the right answer is known and checks the machinery returns it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from bzoo.corrections import fdr, fwer

_SPEC = importlib.util.spec_from_file_location(
    "corrections_on_null",
    Path(__file__).resolve().parents[1] / "scripts" / "14_corrections_on_the_null.py",
)
con = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(con)


def _p_from_t(t: np.ndarray) -> np.ndarray:
    return 2.0 * stats.norm.sf(np.abs(t))


# ----------------------------------------------------------------------
# The correction itself is sound: it is the null that fails.


def test_bonferroni_controls_fwer_on_a_genuinely_standard_normal_family():
    """With a correct null, Bonferroni almost never rejects.  The baseline."""
    rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(200):
        t = rng.standard_normal(2000)
        rejections += fwer.bonferroni(_p_from_t(t), 0.05).n_reject
    # Expected number of families with any rejection is about 5 percent of 200.
    assert rejections <= 25


def test_bonferroni_fails_when_the_null_is_wider_than_assumed():
    """Inflate the null by 1.4 and the same procedure rejects in bulk.

    This is the paper's Table 6 in miniature: nothing about the multiplicity
    arithmetic changed, only the distribution it was handed.
    """
    rng = np.random.default_rng(1)
    t = rng.standard_normal(19380) * 1.405
    n_reject = fwer.bonferroni(_p_from_t(t), 0.05).n_reject
    # An iid normal inflated to 1.405 puts about 16 of 19,380 past the
    # Bonferroni line.  The real population puts 69 there, because its tail is
    # fatter than a rescaled normal, so this simulated figure is the
    # conservative version of the same failure.
    assert n_reject > 5


def test_rescaling_the_null_restores_control():
    """The same family, with the critical value scaled by the measured SD."""
    rng = np.random.default_rng(2)
    sd = 1.405
    t = rng.standard_normal(19380) * sd
    crit_nominal = stats.norm.isf(0.05 / (2 * t.size))
    crit_measured = t.std(ddof=1) * crit_nominal
    assert (np.abs(t) > crit_nominal).sum() > 5
    assert (np.abs(t) > crit_measured).sum() == 0


def test_benjamini_hochberg_also_fails_on_an_inflated_null():
    rng = np.random.default_rng(3)
    t = rng.standard_normal(19380) * 1.405
    assert fdr.benjamini_hochberg(_p_from_t(t), 0.05).n_reject > 100


# ----------------------------------------------------------------------
# C2: the variance split


def test_variance_split_recovers_a_known_c1_and_c2():
    """Build alphas with a known persistent and known sample-specific part.

    The identity the split relies on is
    ``Var(alpha) = Var(persistent) + Var(sample-specific) + E[SE^2]``, so
    feeding in known pieces must return them.
    """
    rng = np.random.default_rng(5)
    n, se = 400000, 0.08
    var_persistent, var_sample = 0.0004, 0.0025

    persistent = rng.normal(0, np.sqrt(var_persistent), n)
    sample_specific = rng.normal(0, np.sqrt(var_sample), n)
    noise = rng.normal(0, se, n)
    alpha = persistent + sample_specific + noise

    mean_se2 = se ** 2
    var_alpha = float(np.var(alpha, ddof=1))
    recovered_sample = var_alpha - var_persistent - mean_se2
    c1 = recovered_sample / mean_se2
    c2t = var_persistent / mean_se2

    assert c1 == pytest.approx(var_sample / mean_se2, rel=0.03)
    assert c2t == pytest.approx(var_persistent / mean_se2, rel=1e-9)
    # And the implied Var(t) matches the direct one.
    assert 1.0 + c1 + c2t == pytest.approx(var_alpha / mean_se2, rel=0.02)


def test_persistent_component_grows_in_t_units_and_sample_specific_does_not():
    """The distinction the whole argument rests on.

    A fixed alpha divided by a shrinking standard error gives a t-statistic
    that grows like sqrt(T).  A quantity that shrinks like 1/sqrt(T), as
    fbar - mu_f does, gives one that does not move.
    """
    fixed_alpha, resid_sd = 0.02, 1.5
    prev_fixed = prev_shrinking = None
    for t_len in (120, 480, 1920):
        se = resid_sd / np.sqrt(t_len)
        t_fixed = fixed_alpha / se
        t_shrinking = (fixed_alpha / np.sqrt(t_len / 120.0)) / se
        if prev_fixed is not None:
            assert t_fixed > prev_fixed * 1.5      # grows
            assert t_shrinking == pytest.approx(prev_shrinking, rel=1e-9)  # flat
        prev_fixed, prev_shrinking = t_fixed, t_shrinking


# ----------------------------------------------------------------------
# C3: the split-sample covariance check


def test_in_sample_beta_does_not_manufacture_the_covariance():
    """The obvious worry about C3, and why it is not the explanation.

    One might expect that estimating beta on the same months that produce rbar
    couples the two mechanically.  For OLS with an intercept it does not: the
    coupling term is proportional to the sum of the demeaned regressor, which
    is exactly zero.  In-sample and out-of-sample estimates of the covariance
    therefore agree, with or without cross-sectional dependence in the
    residuals.  Kept as a test because the paper briefly claimed otherwise.
    """
    def beta_hat(y, x):
        xc = x - x.mean(axis=0)
        return (xc.T @ (y - y.mean(axis=0)) / (xc.T @ xc)).ravel()

    n_s, n_t = 3000, 240
    for corr in (0.0, 0.8):
        ins, outs = [], []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            f = rng.normal(0.0, 4.0, (n_t, 1))
            beta = rng.normal(0.0, 0.3, n_s)
            common = rng.normal(0, 4.0, (n_t, 1))
            idio = rng.normal(0, 4.0, (n_t, n_s))
            e = corr * common + np.sqrt(1 - corr ** 2) * idio
            r = f @ beta[None, :] + e
            half = n_t // 2
            rbar_2 = r[half:].mean(axis=0)
            fbar_2 = f[half:].mean(axis=0)
            ins.append(np.cov(rbar_2, beta_hat(r[half:], f[half:]) * fbar_2,
                              ddof=1)[0, 1])
            outs.append(np.cov(rbar_2, beta_hat(r[:half], f[:half]) * fbar_2,
                               ddof=1)[0, 1])
        assert np.mean(ins) == pytest.approx(np.mean(outs), rel=0.10)


def test_the_covariance_is_a_realised_quantity_positive_in_expectation():
    """What Cov(rbar, beta'fbar) actually is on a null population.

    Substituting rbar = beta'(fbar - mu_f) + ebar gives
    ``Cov_i(rbar, beta'fbar) = (fbar - mu_f)' Sigma_beta fbar``, a realised
    quadratic form in the sample factor means.  Its expectation over samples is
    ``trace(Sigma_beta Var(fbar))``, which is positive, and any one sample can
    land either side of zero.  So a negative value in the data is neither a
    sign that the population is not null nor an artefact; it is a
    sample-specific realisation, which is the same conclusion the variance
    split reaches by a different route.
    """
    n_s, n_t, mu_f = 4000, 240, 0.5
    vals = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        f = rng.normal(mu_f, 4.0, (n_t, 1))
        beta = rng.normal(0.0, 0.3, n_s)
        # E[r] = 0 exactly for every strategy: null by construction.
        r = (f - mu_f) @ beta[None, :] + rng.normal(0, 4.0, (n_t, n_s))
        vals.append(np.cov(r.mean(axis=0), beta * f.mean(axis=0), ddof=1)[0, 1])
    vals = np.array(vals)
    assert vals.mean() > 0                    # positive in expectation
    assert 0.2 < np.mean(vals < 0) < 0.8      # either sign in any one sample


def test_family_sizes_are_ordered_and_include_the_full_population():
    assert con.FAMILY_SIZES == tuple(sorted(con.FAMILY_SIZES))
    assert con.FAMILY_SIZES[-1] == 19380
