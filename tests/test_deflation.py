"""Tests for the deflation module.

Anchors:

* the Gumbel approximation to ``E[max]`` must agree with simulation;
* the exact threshold and p-value must agree with each other and with the
  Sidak correction, which is the same statement on the p-value scale;
* the probabilistic Sharpe ratio must reduce to the Lo (2002) asymptotic
  standard error of the Sharpe ratio when returns are normal;
* the worked ordering in Bailey and Lopez de Prado (2014): a Sharpe ratio
  that looks strong on its own becomes insignificant once the number of
  trials behind it is accounted for.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bzoo.corrections import deflation as D


# Measured relative error of the Gumbel approximation against exact
# quadrature.  These are the numbers quoted in the docstring and in the paper,
# so the test pins them rather than allowing a loose band.
GUMBEL_ERROR = {5: 0.026, 10: 0.024, 100: 0.010, 1000: 0.005, 10000: 0.003}


@pytest.mark.parametrize("n", sorted(GUMBEL_ERROR))
def test_expected_max_matches_the_exact_value(n):
    sigma = 1.3
    approx = D.expected_max_normal(sigma, n)
    exact = D.expected_max_normal_exact(sigma, n)
    assert 0.0 < (approx - exact) / exact <= GUMBEL_ERROR[n]


def test_expected_max_matches_simulation():
    sigma, n = 1.3, 1000
    approx = D.expected_max_normal(sigma, n)
    sim, se = D.expected_max_monte_carlo(sigma, n, n_sim=40000, seed=1)
    assert abs(approx - sim) < 0.01 * sim


def test_gumbel_approximation_is_poor_for_two_trials():
    """Stated as a limitation rather than left to be discovered."""
    approx = D.expected_max_normal(1.0, 2)
    exact = D.expected_max_normal_exact(1.0, 2)
    assert (approx - exact) / exact < -0.07


def test_expected_max_scales_linearly_in_sigma():
    a = D.expected_max_normal(1.0, 500)
    b = D.expected_max_normal(2.5, 500)
    assert b == pytest.approx(2.5 * a)


def test_expected_max_increases_with_trials():
    vals = [D.expected_max_normal(1.0, n) for n in (10, 100, 1000, 10000)]
    assert all(np.diff(vals) > 0)


def test_threshold_and_pvalue_are_consistent():
    sigma, n, alpha = 0.004, 500, 0.05
    c = D.deflated_threshold(sigma, n, alpha)
    assert D.deflated_pvalue(c, sigma, n) == pytest.approx(alpha, rel=1e-8)


def test_deflated_pvalue_equals_the_sidak_adjustment():
    """The deflation p-value is the Sidak correction written on the statistic
    scale, so the two must agree exactly."""
    from bzoo.corrections.fwer import sidak

    sigma, n = 1.0, 200
    x = 3.1
    p_one_sided = float(stats.norm.sf(x / sigma))
    sidak_adj = sidak(np.array([p_one_sided] * n)).adjusted_p[0]
    assert D.deflated_pvalue(x, sigma, n) == pytest.approx(sidak_adj, rel=1e-9)


def test_deflated_pvalue_does_not_underflow():
    """1 - Phi(z)^N computed naively loses all precision for large z; the
    log-space version must not return exactly zero."""
    p = D.deflated_pvalue(8.0, 1.0, 1000)
    assert 0.0 < p < 1e-11


def test_implied_trials_inverts_the_pvalue():
    sigma, alpha = 1.0, 0.05
    for x in (2.5, 3.0, 4.0):
        n = D.implied_trials(x, sigma, alpha)
        assert D.deflated_pvalue(x, sigma, n) == pytest.approx(alpha, rel=1e-6)


def test_implied_trials_is_increasing_in_the_statistic():
    ns = [D.implied_trials(x, 1.0) for x in (2.0, 2.5, 3.0, 3.5)]
    assert all(np.diff(ns) > 0)


def test_psr_reduces_to_the_lo_standard_error_under_normality():
    """Lo (2002): se(SR) = sqrt((1 + SR^2/2)/(T-1)) for iid normal returns.
    The PSR z-score must be (SR - SR*)/se with that se."""
    sr, t = 0.25, 120
    psr = D.probabilistic_sharpe_ratio(sr, t, skew=0.0, kurtosis=3.0)
    se = np.sqrt((1.0 + 0.5 * sr ** 2) / (t - 1))
    assert psr == pytest.approx(float(stats.norm.cdf(sr / se)), rel=1e-10)


def test_psr_falls_with_negative_skew_and_fat_tails():
    base = D.probabilistic_sharpe_ratio(0.25, 120, skew=0.0, kurtosis=3.0)
    skewed = D.probabilistic_sharpe_ratio(0.25, 120, skew=-1.0, kurtosis=3.0)
    fat = D.probabilistic_sharpe_ratio(0.25, 120, skew=0.0, kurtosis=8.0)
    assert skewed < base
    assert fat < base


def test_psr_rises_with_sample_length():
    short = D.probabilistic_sharpe_ratio(0.2, 36)
    long = D.probabilistic_sharpe_ratio(0.2, 360)
    assert long > short


def test_deflated_sharpe_ratio_kills_a_mined_result():
    """A monthly Sharpe of 0.33 over 60 months is a t-statistic of about 2.6
    and looks significant on its own.  With 1,000 trials behind it and a
    cross-trial spread of 0.2 it should not survive."""
    out = D.deflated_sharpe_ratio(
        sr_hat=0.33, n_obs=60, sr_trials_sd=0.2, n_trials=1000
    )
    assert out["psr"] > 0.99
    assert out["dsr"] < 0.05
    assert out["sr_star"] > out["sr_hat"]


def test_deflated_sharpe_ratio_spares_a_strong_result():
    out = D.deflated_sharpe_ratio(
        sr_hat=0.45, n_obs=600, sr_trials_sd=0.05, n_trials=1000
    )
    assert out["dsr"] > 0.95


def test_deflated_improvement_reports_a_consistent_verdict():
    res = D.deflated_improvement(delta_obs=0.006, sigma_delta=0.002, n_trials=1000)
    assert res["survives"] == (res["delta_obs"] > res["threshold"])
    assert res["survives"] == (res["p_value"] < 0.05)
    assert res["expected_max"] < res["threshold"]  # E[max] < 95th percentile


def test_deflated_improvement_flips_with_the_trial_count():
    small = D.deflated_improvement(0.006, 0.002, 10)
    large = D.deflated_improvement(0.006, 0.002, 100000)
    assert small["survives"] and not large["survives"]
    assert large["threshold"] > small["threshold"]


def test_zero_sigma_edge_case():
    assert D.deflated_pvalue(0.1, 0.0, 100) == 0.0
    assert D.deflated_pvalue(-0.1, 0.0, 100) == 1.0
    assert D.implied_trials(0.1, 0.0) == float("inf")


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        D.expected_max_normal(1.0, 1)
    with pytest.raises(ValueError):
        D.expected_max_normal(-1.0, 10)
    with pytest.raises(ValueError):
        D.deflated_threshold(1.0, 10, alpha=1.0)
    with pytest.raises(ValueError):
        D.probabilistic_sharpe_ratio(0.2, 2)
