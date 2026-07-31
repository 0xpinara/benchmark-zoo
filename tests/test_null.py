"""Tests for the null-estimation modules.

The style here is recovery: build a population whose null properties are
known by construction, then check that the estimator returns them.  That is
the same logic as the finance testbed itself, at unit-test scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bzoo.null import dependence as dep
from bzoo.null import empirical as emp
from bzoo.null import tails


# ----------------------------------------------------------------------
# empirical.py


def test_summarise_null_recovers_a_standard_normal():
    rng = np.random.default_rng(0)
    t = rng.standard_normal(50000)
    s = emp.summarise_null(t, seed=0)
    assert abs(s.mean) < 0.02
    assert s.sd == pytest.approx(1.0, abs=0.02)
    assert s.mad_sd == pytest.approx(1.0, abs=0.02)
    assert s.kurtosis == pytest.approx(3.0, abs=0.1)
    assert s.frac_abs_gt_196 == pytest.approx(0.05, abs=0.005)
    assert s.ks_pvalue > 0.01


def test_summarise_null_detects_inflated_variance():
    rng = np.random.default_rng(1)
    t = rng.standard_normal(50000) * 1.4
    s = emp.summarise_null(t, seed=0)
    assert s.sd == pytest.approx(1.4, abs=0.03)
    assert s.frac_abs_gt_196 > 0.15
    assert s.ks_pvalue < 1e-10


def test_sd_and_mad_sd_separate_scale_from_tails():
    """A scale mixture has heavy tails but a near-normal centre, so the two
    scale estimates diverge.  This is the case where reporting only one of
    them is misleading."""
    rng = np.random.default_rng(2)
    t = rng.standard_normal(50000) * np.where(rng.random(50000) < 0.05, 5.0, 1.0)
    s = emp.summarise_null(t, seed=0)
    assert s.sd > 1.4
    assert s.mad_sd == pytest.approx(1.0, abs=0.05)


def test_variance_inflation_interval_covers_the_truth():
    rng = np.random.default_rng(3)
    t = rng.standard_normal(20000) * 1.25
    vi = emp.variance_inflation(t, n_boot=400, seed=0)
    assert vi["sd_lo"] <= 1.25 <= vi["sd_hi"]


def test_exceedance_table_matches_the_gaussian_when_the_null_is_gaussian():
    rng = np.random.default_rng(4)
    t = rng.standard_normal(200000)
    tab = emp.exceedance_table(t)
    assert np.allclose(tab["ratio"].to_numpy()[:3], 1.0, atol=0.15)
    for _, row in tab.iterrows():
        if row["n_exceed"] > 20:
            assert row["empirical_lo"] <= row["gaussian"] <= row["empirical_hi"]


def test_calibrated_threshold_recovers_the_scaled_quantile():
    rng = np.random.default_rng(5)
    sigma = 1.5
    t = rng.standard_normal(200000) * sigma
    thr = emp.calibrated_threshold(t, alpha=0.05, two_sided=True)
    assert thr == pytest.approx(1.959964 * sigma, rel=0.02)


def test_empirical_pvalues_are_uniform_under_the_null():
    rng = np.random.default_rng(6)
    null = rng.standard_normal(20000)
    obs = rng.standard_normal(5000)
    p = emp.empirical_pvalues(obs, null)
    assert stats.kstest(p, "uniform").pvalue > 0.01
    assert p.min() >= 1.0 / (null.size + 1)


def test_empirical_pvalues_are_monotone_in_the_statistic():
    rng = np.random.default_rng(7)
    null = rng.standard_normal(5000)
    obs = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    p = emp.empirical_pvalues(obs, null)
    assert np.all(np.diff(p) <= 0)


def test_conditional_summary_returns_one_row_per_group():
    rng = np.random.default_rng(8)
    groups = {
        "a": rng.standard_normal(5000),
        "b": rng.standard_normal(5000) * 2.0,
    }
    out = emp.conditional_summary(groups, seed=0)
    assert list(out["group"]) == ["a", "b"]
    assert out.loc[out["group"] == "b", "sd"].iloc[0] > 1.8


# ----------------------------------------------------------------------
# tails.py


@pytest.mark.parametrize("xi_true", [-0.2, 0.0, 0.25])
def test_gpd_recovers_the_shape_parameter(xi_true):
    rng = np.random.default_rng(int(100 * (xi_true + 1)))
    x = stats.genpareto.rvs(xi_true, loc=0.0, scale=1.0, size=40000, random_state=rng)
    fit = tails.fit_gpd(x, quantile=0.80)
    # Above a high threshold a GPD is still a GPD with the same shape.
    assert fit.xi == pytest.approx(xi_true, abs=max(0.05, 3 * fit.xi_se))


def test_gaussian_tail_has_shape_near_zero_or_below():
    """The normal is in the Gumbel domain of attraction, so a GPD fit to its
    tail should give a shape parameter that is not reliably positive."""
    rng = np.random.default_rng(9)
    x = np.abs(rng.standard_normal(200000))
    fit = tails.fit_gpd(x, quantile=0.95)
    assert fit.xi < 3 * fit.xi_se


def test_gpd_tail_probability_matches_the_empirical_one_in_sample():
    rng = np.random.default_rng(10)
    x = np.abs(rng.standard_normal(200000))
    fit = tails.fit_gpd(x, quantile=0.95)
    for c in (2.5, 3.0, 3.5):
        emp_p = float(np.mean(x > c))
        assert fit.tail_probability(c) == pytest.approx(emp_p, rel=0.25)


def test_gpd_quantile_inverts_tail_probability():
    rng = np.random.default_rng(11)
    x = stats.genpareto.rvs(0.15, scale=1.0, size=20000, random_state=rng)
    fit = tails.fit_gpd(x, quantile=0.90)
    for p in (1e-2, 1e-3, 1e-4):
        q = fit.quantile(p)
        assert fit.tail_probability(q) == pytest.approx(p, rel=1e-6)


def test_threshold_stability_is_flat_for_a_true_gpd():
    rng = np.random.default_rng(12)
    x = stats.genpareto.rvs(0.2, scale=1.0, size=60000, random_state=rng)
    tab = tails.threshold_stability(x)
    assert tab["xi"].std() < 0.06


def test_tail_comparison_flags_a_heavy_tail():
    rng = np.random.default_rng(13)
    x = rng.standard_t(df=4, size=100000)
    tab = tails.tail_comparison(x, quantile=0.95)
    high = tab.loc[tab["cutoff"] >= 4.0, "gpd_over_gaussian"]
    assert (high > 5).all()


def test_fit_gpd_refuses_too_few_exceedances():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError, match="exceedances"):
        tails.fit_gpd(rng.standard_normal(100), quantile=0.99)


# ----------------------------------------------------------------------
# dependence.py


def test_eigenvalues_sum_to_the_number_of_columns():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((200, 40))
    ev = dep.correlation_eigenvalues(x)
    assert ev.sum() == pytest.approx(40.0, rel=1e-8)
    assert np.all(ev >= -1e-10)


def test_effective_n_equals_k_for_independent_columns():
    rng = np.random.default_rng(16)
    x = rng.standard_normal((5000, 30))
    ev = dep.correlation_eigenvalues(x)
    out = dep.effective_n_eigen(ev)
    assert out["n_eff_cheverud_nyholt"] == pytest.approx(30, rel=0.15)
    assert out["n_eff_li_ji"] == pytest.approx(30, rel=0.20)


def test_effective_n_collapses_for_identical_columns():
    rng = np.random.default_rng(17)
    base = rng.standard_normal((500, 1))
    x = np.repeat(base, 30, axis=1) + rng.standard_normal((500, 30)) * 1e-6
    ev = dep.correlation_eigenvalues(x)
    out = dep.effective_n_eigen(ev)
    assert out["n_eff_cheverud_nyholt"] < 2.0
    assert out["n_eff_li_ji"] < 2.0
    assert out["top1_share"] > 0.95


def test_effective_n_is_between_one_and_k_for_partial_correlation():
    rng = np.random.default_rng(18)
    common = rng.standard_normal((1000, 1))
    x = 0.8 * common + 0.6 * rng.standard_normal((1000, 25))
    out = dep.effective_n_eigen(dep.correlation_eigenvalues(x))
    assert 1.0 < out["n_eff_cheverud_nyholt"] < 25.0
    assert 1.0 < out["n_eff_li_ji"] < 25.0


def test_sidak_effective_n_recovers_k_for_independent_statistics():
    rng = np.random.default_rng(19)
    k = 50
    reps = rng.standard_normal((40000, k))
    out = dep.effective_n_sidak(dep.max_stat_distribution(reps), k, alpha=0.05)
    assert out["n_eff_gaussian_marginal"] == pytest.approx(k, rel=0.2)


def test_sidak_effective_n_is_one_for_perfectly_dependent_statistics():
    rng = np.random.default_rng(20)
    col = rng.standard_normal((40000, 1))
    reps = np.repeat(col, 50, axis=1)
    out = dep.effective_n_sidak(dep.max_stat_distribution(reps), 50, alpha=0.05)
    assert out["n_eff_gaussian_marginal"] == pytest.approx(1.0, rel=0.25)


def test_bootstrap_effective_n_from_a_panel():
    """Independent zero-mean columns: the bootstrap route should recover a
    count near K, and a strongly co-moving panel should give far less."""
    rng = np.random.default_rng(21)
    indep = rng.standard_normal((300, 20))
    common = rng.standard_normal((300, 1))
    dependent = 0.97 * common + 0.24 * rng.standard_normal((300, 20))
    a = dep.effective_n_from_panel(indep, n_boot=1500, seed=0)
    b = dep.effective_n_from_panel(dependent, n_boot=1500, seed=0)
    assert a["n_eff_gaussian_marginal"] > 8
    assert b["n_eff_gaussian_marginal"] < a["n_eff_gaussian_marginal"] / 2


def test_dependence_report_has_all_three_methods():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((250, 30))
    rep = dep.dependence_report(x, n_boot=300, seed=0)
    assert len(rep) == 3
    assert set(rep.columns) == {"method", "n_eff", "n_tests", "ratio"}


def test_eigen_subsampling_gives_a_similar_answer():
    rng = np.random.default_rng(23)
    common = rng.standard_normal((400, 1))
    x = 0.7 * common + 0.7 * rng.standard_normal((400, 300))
    full = dep.effective_n_eigen(dep.correlation_eigenvalues(x))["ratio_cheverud"]
    sub = dep.effective_n_eigen(
        dep.correlation_eigenvalues(x, max_columns=100, seed=0)
    )["ratio_cheverud"]
    assert abs(full - sub) < 0.1
