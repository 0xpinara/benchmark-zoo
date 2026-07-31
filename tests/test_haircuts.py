"""Tests for the Harvey-Liu-Zhu haircuts.

There is no single worked example in the source papers that can be reproduced
without their unobserved-test model, so the anchors here are consistency
properties: with no unobserved tests the haircut procedures must coincide exactly
with the corresponding correction in :mod:`bzoo.corrections.fwer`, the haircut
must grow with the assumed number of tests, and calibrating the null must move it
in the direction the calibration implies.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bzoo.corrections import fwer, haircuts

T_STATS = np.array([5.2, 4.1, 3.6, 3.05, 2.6, 2.4, 2.1, 1.8, 1.2, 0.4])


def test_pvalue_and_t_are_inverses():
    t = np.array([0.5, 2.0, 3.0, 5.0])
    assert np.allclose(haircuts.pvalue_to_t(haircuts.t_to_pvalue(t)), t)


def test_with_no_unobserved_tests_it_matches_the_plain_correction():
    p = haircuts.t_to_pvalue(T_STATS)
    k = p.size
    for meth, plain in (
        ("bonferroni", fwer.bonferroni(p)),
        ("holm", fwer.holm(p)),
    ):
        hlz = haircuts.harvey_liu_zhu(p, n_total_tests=k, method=meth)
        assert np.allclose(hlz.adjusted_p, plain.adjusted_p), meth
        assert np.array_equal(hlz.reject, plain.reject), meth


def test_bhy_with_no_unobserved_tests_matches_benjamini_yekutieli():
    from bzoo.corrections import fdr

    p = haircuts.t_to_pvalue(T_STATS)
    hlz = haircuts.harvey_liu_zhu(p, n_total_tests=p.size, method="bhy")
    by = fdr.benjamini_yekutieli(p)
    assert np.allclose(hlz.adjusted_p, by.adjusted_p)


def test_haircut_grows_with_the_assumed_number_of_tests():
    """Measured on the strongest factor.  The haircut saturates at 1 for the weak
    ones -- once the adjusted p-value reaches 1 the adjusted t-statistic is 0 and
    the whole Sharpe ratio is gone -- so a median over all factors stops moving
    and says nothing."""
    strongest = []
    for m in (10, 100, 1_000, 10_000):
        hc = haircuts.haircuts(T_STATS, n_total_tests=m, method="bonferroni")
        strongest.append(float(hc["haircut"].iloc[0]))
    assert all(np.diff(strongest) > 0)
    assert 0.0 <= strongest[0] <= 1.0
    assert strongest[-1] <= 1.0


def test_adjusted_t_never_exceeds_the_observed_t():
    hc = haircuts.haircuts(T_STATS, n_total_tests=1_000, method="bhy")
    assert (hc["t_adj"] <= hc["t_obs"] + 1e-9).all()
    assert (hc["haircut"] >= -1e-9).all()


def test_a_wider_null_increases_the_haircut():
    """A wider null makes the same observed t-statistic less surprising, so more
    of the reported Sharpe ratio is attributed to the search and the cut is
    larger.  Getting this direction wrong would invert every survival count in
    the paper, so it is pinned here."""
    narrow = haircuts.haircuts(T_STATS, 1_000, method="bonferroni", marginal_sd=1.0)
    wide = haircuts.haircuts(T_STATS, 1_000, method="bonferroni", marginal_sd=1.4)
    assert wide["haircut"].iloc[0] > narrow["haircut"].iloc[0]
    assert wide["survives_05"].sum() <= narrow["survives_05"].sum()


def test_a_narrower_null_increases_survival():
    narrow = haircuts.haircuts(T_STATS, 1_000, method="bonferroni", marginal_sd=0.889)
    nominal = haircuts.haircuts(T_STATS, 1_000, method="bonferroni", marginal_sd=1.0)
    assert narrow["survives_05"].sum() >= nominal["survives_05"].sum()


def test_threshold_grid_is_increasing_in_m_and_matches_the_normal_quantile():
    grid = haircuts.threshold_grid([10, 100, 1_000, 10_000], alpha=0.05)
    for col in ("t_bonferroni", "t_holm", "t_bhy"):
        assert np.all(np.diff(grid[col].to_numpy()) > 0), col
    row = grid.loc[grid["n_total_tests"] == 1_000].iloc[0]
    assert row["t_bonferroni"] == pytest.approx(
        float(stats.norm.isf(0.05 / 1_000 / 2))
    )
    # A single new candidate gets the same treatment from Holm as from
    # Bonferroni, because the first step of Holm is the Bonferroni step.
    assert row["t_holm"] == pytest.approx(row["t_bonferroni"])
    # BHY is stricter than Bonferroni for a single candidate, by the harmonic
    # factor c(M); it becomes more permissive only at higher ranks.
    assert row["t_bhy"] > row["t_bonferroni"]


def test_calibrated_threshold_grid_scales_with_the_null():
    a = haircuts.threshold_grid([1_000], marginal_sd=1.0)
    b = haircuts.threshold_grid([1_000], marginal_sd=1.405)
    assert b["t_bonferroni"].iloc[0] == pytest.approx(
        1.405 * a["t_bonferroni"].iloc[0]
    )


def test_signs_are_ignored():
    a = haircuts.haircuts(np.array([3.0, -3.0]), 100)
    assert a["t_obs"].tolist() == [3.0, 3.0]
    assert a["haircut"].iloc[0] == pytest.approx(a["haircut"].iloc[1])


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="at least the number"):
        haircuts.haircuts(T_STATS, n_total_tests=3)
    with pytest.raises(ValueError, match="method must be"):
        haircuts.haircuts(T_STATS, 100, method="nonsense")
    with pytest.raises(ValueError, match="marginal_sd"):
        haircuts.haircuts(T_STATS, 100, marginal_sd=0.0)


def test_sigma_has_no_default_anywhere():
    """The paper and the README both claim the package refuses to guess a null
    spread.  That has to be true of the signatures, not just of the prose."""
    import inspect

    from bzoo.corrections import deflation

    for fn, arg in (
        (deflation.deflated_improvement, "sigma_delta"),
        (deflation.deflated_threshold, "sigma"),
        (deflation.deflated_pvalue, "sigma"),
        (deflation.expected_max_normal, "sigma"),
        (deflation.deflated_sharpe_ratio, "sr_trials_sd"),
    ):
        param = inspect.signature(fn).parameters[arg]
        assert param.default is inspect.Parameter.empty, f"{fn.__name__}.{arg}"
