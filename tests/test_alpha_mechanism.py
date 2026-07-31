"""Tests for the alpha-mechanism analysis.

The style is recovery, as elsewhere: build a population whose alpha
properties are known by construction and check that the analysis returns
them.  Three of these tests exist because the first version of the analysis
got the corresponding thing wrong, and the comment on each says which.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "alpha_mechanism",
    Path(__file__).resolve().parents[1] / "scripts" / "12_alpha_mechanism.py",
)
am = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(am)


FF5 = am.FF5


def _factors(n_months: int = 480, seed: int = 0) -> pd.DataFrame:
    """Factors with realistic, and importantly non-zero, mean returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-01", periods=n_months, freq="MS")
    means = {"mktrf": 0.55, "smb": 0.20, "hml": 0.30, "rmw": 0.28, "cma": 0.30}
    sds = {"mktrf": 4.4, "smb": 3.0, "hml": 2.8, "rmw": 2.2, "cma": 2.0}
    return pd.DataFrame(
        {c: rng.normal(means[c], sds[c], n_months) for c in FF5}, index=idx
    )


def _panel(factors: pd.DataFrame, betas: np.ndarray, resid_sd=1.5, seed=1):
    """Returns with exactly zero expected raw return and the given loadings.

    The point of the construction: each strategy's return is
    ``beta' (f - fbar) + noise``, which is mean zero *and* carries real
    exposure, so its true alpha is ``-beta' fbar`` and is not zero.
    """
    rng = np.random.default_rng(seed)
    f = factors[FF5].to_numpy()
    fc = f - f.mean(axis=0)
    n_t, n_s = f.shape[0], betas.shape[0]
    noise = rng.normal(0.0, resid_sd, (n_t, n_s))
    return pd.DataFrame(fc @ betas.T + noise, index=factors.index,
                        columns=range(n_s))


# ----------------------------------------------------------------------
# The identity


def test_identity_holds_to_machine_precision():
    """alpha = rbar - beta' fbar, on the regression's own sample.

    The first version computed rbar over each strategy's own months and the
    exposure over the regression months.  Those differ whenever the factor
    series starts later than the returns, and the identity then failed by a
    few percent of a cross-sectional SD.
    """
    f = _factors()
    betas = np.random.default_rng(2).normal(0, 0.3, (200, 5))
    res = am.fit_model(_panel(f, betas), f, FF5, min_months=60)
    err = np.abs(res["rbar"] - res["alpha"] - res["exposure"]).max()
    assert err < 1e-12


def test_identity_survives_a_late_starting_factor_series():
    f = _factors()
    betas = np.random.default_rng(3).normal(0, 0.3, (100, 5))
    panel = _panel(f, betas)
    # Returns start a year before the factors, as in the real data.
    extra = pd.date_range(panel.index[0] - pd.DateOffset(months=12),
                          periods=12, freq="MS")
    panel = pd.concat([pd.DataFrame(0.0, index=extra, columns=panel.columns), panel])
    res = am.fit_model(panel, f, FF5, min_months=60)
    assert np.abs(res["rbar"] - res["alpha"] - res["exposure"]).max() < 1e-12


# ----------------------------------------------------------------------
# A1


def test_zero_exposure_population_has_a_standard_normal_alpha_t():
    """No exposure, no alpha inflation.  This is the paper's falsifier."""
    f = _factors()
    betas = np.zeros((3000, 5))
    res = am.fit_model(_panel(f, betas, seed=7), f, FF5, min_months=60)
    assert res["alpha_t"].std(ddof=1) == pytest.approx(1.0, abs=0.05)


def test_exposure_widens_the_alpha_t_distribution():
    f = _factors()
    rng = np.random.default_rng(11)
    narrow = am.fit_model(_panel(f, rng.normal(0, 0.05, (3000, 5))), f, FF5, 60)
    wide = am.fit_model(_panel(f, rng.normal(0, 0.40, (3000, 5))), f, FF5, 60)
    assert wide["alpha_t"].std(ddof=1) > narrow["alpha_t"].std(ddof=1) + 0.3
    # And it is the numerator that moves, not the standard error.
    assert wide["alpha"].std(ddof=1) > narrow["alpha"].std(ddof=1)
    assert wide["alpha_se"].median() == pytest.approx(
        narrow["alpha_se"].median(), rel=0.05
    )


def test_se_inflation_factor_matches_the_regression_standard_error():
    """sqrt(1 + fbar' S^-1 fbar), checked against an actual OLS solve."""
    f = _factors(n_months=300, seed=5)
    months = f.index
    got = am.se_inflation_factor(f, FF5, months)
    x = np.column_stack([np.ones(len(f)), f[FF5].to_numpy()])
    xtx_inv = np.linalg.inv(x.T @ x)
    expected = np.sqrt(xtx_inv[0, 0] * len(f))
    assert got == pytest.approx(expected, rel=1e-10)
    # It is a Sharpe-ratio quantity, so it does not shrink with T.
    assert am.se_inflation_factor(_factors(1200, 5), FF5,
                                 _factors(1200, 5).index) > 1.0


# ----------------------------------------------------------------------
# A2


def test_slope_is_one_only_when_rbar_is_orthogonal_to_the_exposure():
    """The claimed 'population slope is exactly 1' needs an extra condition.

    By construction here the mean return is exactly zero for every strategy
    up to noise, so the covariance term vanishes and the slope is 1.  The
    general formula is checked in the next test.
    """
    f = _factors()
    betas = np.random.default_rng(13).normal(0, 0.3, (4000, 5))
    res = am.fit_model(_panel(f, betas), f, FF5, 60)
    y = res["alpha"].to_numpy()
    x = -res["exposure"].to_numpy()
    slope = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    assert slope == pytest.approx(1.0, abs=0.1)


def test_slope_formula_predicts_the_departure_from_one():
    """slope = 1 + Cov(rbar, -beta'fbar) / Var(beta'fbar), exactly.

    Built so the departure is large: the strategies with the most exposure
    are given a deliberately non-zero mean return, which is what happens in
    the value-weighted panel of the real data.
    """
    f = _factors()
    rng = np.random.default_rng(17)
    betas = rng.normal(0, 0.3, (4000, 5))
    panel = _panel(f, betas, seed=19)
    panel = panel + np.linspace(-0.4, 0.4, betas.shape[0])[None, :] * betas[:, 1]

    res = am.fit_model(panel, f, FF5, 60)
    y = res["alpha"].to_numpy()
    x = -res["exposure"].to_numpy()
    rbar = res["rbar"].to_numpy()
    slope = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    predicted = 1.0 + np.cov(rbar, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    assert slope == pytest.approx(predicted, rel=1e-10)


def test_variance_decomposition_is_exact():
    """Var(alpha) = Var(rbar) + Var(exposure) - 2 Cov(rbar, exposure)."""
    f = _factors()
    betas = np.random.default_rng(23).normal(0, 0.3, (1000, 5))
    res = am.fit_model(_panel(f, betas), f, FF5, 60)
    a, r, e = (res[c].to_numpy() for c in ("alpha", "rbar", "exposure"))
    lhs = np.var(a, ddof=1)
    rhs = np.var(r, ddof=1) + np.var(e, ddof=1) - 2 * np.cov(r, e, ddof=1)[0, 1]
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ----------------------------------------------------------------------
# A6


def test_placebo_recovers_a_unit_alpha_t_under_both_constructions():
    f = _factors()
    betas = np.random.default_rng(29).normal(0, 0.3, (1500, 5))
    panels = {"vw": _panel(f, betas, seed=31)}
    saved, am.WEIGHTINGS = am.WEIGHTINGS, ("vw",)
    try:
        out = am.a6_placebo(panels, f, n_rep=2)["vw"]
    finally:
        am.WEIGHTINGS = saved
    assert out["iid_normal"]["sd_alpha_t_mean"] == pytest.approx(1.0, abs=0.06)
    assert out["residual_bootstrap"]["sd_alpha_t_mean"] == pytest.approx(1.0, abs=0.08)


def test_residual_permutation_would_have_been_the_wrong_placebo():
    """Why A6 bootstraps rather than permutes.

    OLS residuals have exactly zero sample mean, so a permutation of them
    still has zero mean; alpha is then forced to -beta'fbar with beta near
    zero, and SD(t) collapses far below 1.  Kept as a test so nobody
    reintroduces it.
    """
    f = _factors()
    betas = np.random.default_rng(37).normal(0, 0.3, (800, 5))
    res = am.fit_model(_panel(f, betas, seed=41), f, FF5, 60)
    fitted = (f[FF5].to_numpy() @ res[[f"beta_{c}" for c in FF5]].to_numpy().T
              + res["alpha"].to_numpy()[None, :])
    resid = _panel(f, betas, seed=41).to_numpy() - fitted
    perm = np.random.default_rng(43).permutation(resid.shape[0])
    sim = pd.DataFrame(resid[perm, :], index=f.index, columns=res.index)
    assert am.fit_model(sim, f, FF5, 60)["alpha_t"].std(ddof=1) < 0.8


# ----------------------------------------------------------------------
# A7


def test_describe_ticker_reads_as_a_sentence():
    got = am._describe_ticker("L2_lng_1_8_sht_15_16")
    assert "second letter" in got
    assert "buy groups 1 and 8" in got
    assert "sell groups 15 and 16" in got
