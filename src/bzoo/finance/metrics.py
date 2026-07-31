"""Per-strategy performance statistics, computed on whole panels at once.

Everything here is vectorised across strategies.  With 19,380 strategies and
740 months, a per-strategy Python loop takes minutes and a matrix version
takes under a second, and the loop version is the one that ends up with a
subtle inconsistency between two code paths.

Definitions used throughout
---------------------------
Returns are monthly and in percent, as distributed by the source.

* ``mean``            arithmetic mean monthly return
* ``sd``              sample standard deviation, ``ddof=1``
* ``t_stat``          ``mean / (sd / sqrt(T))``, the ordinary t-statistic
                      against a zero mean
* ``t_stat_nw``       the same with a Newey-West standard error, ``L`` lags
* ``sharpe_monthly``  ``mean / sd``
* ``sharpe_annual``   ``sharpe_monthly * sqrt(12)``
* ``skew``            sample skewness (Fisher)
* ``kurtosis``        raw sample kurtosis, so 3 under normality

The annualisation is the usual square-root-of-12 convention, which assumes
serial independence.  We report ``t_stat_nw`` alongside ``t_stat`` so that
the reader can see how much that assumption is worth here; on these
long-short portfolios the difference is small.

The relation ``t_stat = sharpe_monthly * sqrt(T)`` holds exactly for the
non-Newey-West version, and :func:`summarise_panel` asserts it.  It is the
reason a threshold expressed in t units can be translated into a Sharpe
ratio haircut without any extra assumption.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _panel_to_array(panel: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
    x = panel.to_numpy(dtype=np.float64)
    mask = np.isfinite(x)
    return x, mask


def summarise_panel(
    panel: pd.DataFrame,
    min_months: int = 60,
    nw_lags: int = 6,
) -> pd.DataFrame:
    """Per-column performance statistics for a wide return panel.

    Parameters
    ----------
    panel:
        Rows are dates, columns are strategies, values are monthly returns
        in percent.  Missing values are allowed and handled per column.
    min_months:
        Columns with fewer than this many observations are dropped.  A
        t-statistic on 20 months is not comparable to one on 700, and
        keeping them widens the estimated null for a reason that has nothing
        to do with data mining.  The default of 60 months is the screen used
        by Chen and Zimmermann for the published predictors.
    nw_lags:
        Lag truncation for the Newey-West standard error.

    Returns
    -------
    Frame indexed by strategy with the statistics listed in the module
    docstring.
    """
    x, mask = _panel_to_array(panel)
    n_obs = mask.sum(axis=0)

    xz = np.where(mask, x, 0.0)
    total = xz.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / n_obs
        dev = np.where(mask, x - mean, 0.0)
        var = (dev ** 2).sum(axis=0) / (n_obs - 1)
        sd = np.sqrt(var)
        se = sd / np.sqrt(n_obs)
        t_stat = mean / se

        m2 = (dev ** 2).sum(axis=0) / n_obs
        m3 = (dev ** 3).sum(axis=0) / n_obs
        m4 = (dev ** 4).sum(axis=0) / n_obs
        skew = m3 / m2 ** 1.5
        kurt = m4 / m2 ** 2

    t_nw = _newey_west_t(dev, mask, mean, n_obs, nw_lags)

    out = pd.DataFrame(
        {
            "n_months": n_obs,
            "mean": mean,
            "sd": sd,
            "t_stat": t_stat,
            "t_stat_nw": t_nw,
            "sharpe_monthly": mean / sd,
            "sharpe_annual": (mean / sd) * np.sqrt(12.0),
            "skew": skew,
            "kurtosis": kurt,
        },
        index=panel.columns,
    )
    out = out.loc[out["n_months"] >= min_months].copy()

    # Identity check: t = sharpe_monthly * sqrt(T).  Cheap, and it catches
    # any future change that breaks the translation between the two scales.
    lhs = out["t_stat"].to_numpy()
    rhs = (out["sharpe_monthly"] * np.sqrt(out["n_months"])).to_numpy()
    good = np.isfinite(lhs) & np.isfinite(rhs)
    if good.any() and not np.allclose(lhs[good], rhs[good], rtol=1e-8, atol=1e-8):
        raise AssertionError("t-statistic and Sharpe ratio disagree")
    return out


def _newey_west_t(
    dev: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    n_obs: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Newey-West t-statistics for the column means, Bartlett kernel.

    ``dev`` holds demeaned returns with zeros where data are missing, which
    is the right fill here: a missing month contributes nothing to either
    the variance or the autocovariance terms.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma0 = (dev ** 2).sum(axis=0) / n_obs
        s = gamma0.copy()
        for l in range(1, lags + 1):
            prod = dev[l:] * dev[:-l]
            valid = mask[l:] & mask[:-l]
            gl = np.where(valid, prod, 0.0).sum(axis=0) / n_obs
            s = s + 2.0 * (1.0 - l / (lags + 1.0)) * gl
        s = np.where(s > 0, s, np.nan)
        se_nw = np.sqrt(s / n_obs)
        return mean / se_nw


def factor_alphas(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: "list[str]",
    min_months: int = 60,
    nw_lags: int = 6,
) -> pd.DataFrame:
    """Intercepts and their t-statistics from a common-factor regression.

    Runs ``r_it = a_i + b_i' f_t + e_it`` for every column of ``panel``
    against the same factor set.  Because the regressors are shared, this is
    one least-squares solve with many right-hand sides rather than one
    regression per strategy.

    Strategies with different missing-value patterns cannot share a single
    solve, so we group columns by their observation mask and solve once per
    distinct mask.  On these panels there are only a handful of distinct
    masks, so this stays fast while remaining exact.

    Returns
    -------
    Frame indexed by strategy with ``alpha``, ``alpha_se``, ``alpha_t``,
    ``alpha_t_nw``, ``r2`` and ``n_months``, plus one loading column per
    factor.
    """
    common = panel.index.intersection(factors.index)
    y_all = panel.loc[common]
    f_all = factors.loc[common, factor_cols]

    fmask = np.isfinite(f_all.to_numpy()).all(axis=1)
    y_all = y_all.loc[fmask]
    f_all = f_all.loc[fmask]

    y = y_all.to_numpy(dtype=np.float64)
    f = f_all.to_numpy(dtype=np.float64)
    n_f = f.shape[1]

    obs = np.isfinite(y)
    keep = obs.sum(axis=0) >= min_months
    y = y[:, keep]
    obs = obs[:, keep]
    names = y_all.columns[keep]

    alpha = np.full(y.shape[1], np.nan)
    alpha_se = np.full(y.shape[1], np.nan)
    alpha_t_nw = np.full(y.shape[1], np.nan)
    r2 = np.full(y.shape[1], np.nan)
    betas = np.full((y.shape[1], n_f), np.nan)
    n_used = obs.sum(axis=0)

    # Group columns by identical observation pattern.
    packed = np.packbits(obs, axis=0)
    _, group_ids = np.unique(packed.T, axis=0, return_inverse=True)

    for g in np.unique(group_ids):
        cols = np.flatnonzero(group_ids == g)
        rows = obs[:, cols[0]]
        xg = np.column_stack([np.ones(rows.sum()), f[rows]])
        yg = y[np.ix_(rows, cols)]
        coef, *_ = np.linalg.lstsq(xg, yg, rcond=None)
        resid = yg - xg @ coef
        dof = rows.sum() - (n_f + 1)
        if dof <= 0:
            continue
        sigma2 = (resid ** 2).sum(axis=0) / dof
        xtx_inv = np.linalg.pinv(xg.T @ xg)
        se = np.sqrt(np.outer(sigma2, np.diag(xtx_inv)))
        alpha[cols] = coef[0]
        alpha_se[cols] = se[:, 0]
        betas[cols] = coef[1:].T
        tss = ((yg - yg.mean(axis=0)) ** 2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[cols] = 1.0 - (resid ** 2).sum(axis=0) / tss
        alpha_t_nw[cols] = _nw_alpha_t(xg, resid, coef[0], xtx_inv, nw_lags)

    out = pd.DataFrame(
        {
            "n_months": n_used,
            "alpha": alpha,
            "alpha_se": alpha_se,
            "alpha_t": alpha / alpha_se,
            "alpha_t_nw": alpha_t_nw,
            "r2": r2,
        },
        index=names,
    )
    for j, col in enumerate(factor_cols):
        out[f"beta_{col}"] = betas[:, j]
    return out


def _nw_alpha_t(
    x: np.ndarray,
    resid: np.ndarray,
    alpha: np.ndarray,
    xtx_inv: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Newey-West t-statistic on the intercept for many regressions at once.

    Standard sandwich form: ``V = (X'X)^-1 S (X'X)^-1`` with ``S`` the
    Bartlett-weighted sum of ``x_t x_s' e_t e_s``.  We only need the (0,0)
    element of ``V``, so we contract with the first row of ``(X'X)^-1``
    instead of forming the whole matrix for every column.
    """
    n_t = x.shape[0]
    a = xtx_inv[0]  # first row
    # z_t = (a . x_t) * e_t ; then V_00 = sum_lags Bartlett * autocov(z)
    ax = x @ a  # (T,)
    z = ax[:, None] * resid  # (T, K)
    s = (z ** 2).sum(axis=0)
    for l in range(1, lags + 1):
        s = s + 2.0 * (1.0 - l / (lags + 1.0)) * (z[l:] * z[:-l]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.where(s > 0, s, np.nan))
        return alpha / se


def factor_residuals(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: "list[str]",
) -> "tuple[pd.DataFrame, float, float]":
    """Residuals from a common-factor regression on complete-history columns.

    Restricted to columns with no missing months, so that every regression
    shares one design matrix.  That is what makes the permutation null for the
    alpha cheap: with a shared ``X``, imposing ``alpha = 0`` and resampling is
    a single matrix product over the residuals.

    Returns ``(residuals, se_scale, dof_scale)`` where

    ``se_scale``
        ``sqrt(T * (X'X)^{-1}_{00})``, the factor by which the standard error
        of the intercept exceeds ``sigma / sqrt(T)``.  It is at least 1, with
        equality only when the factor means are zero, and it is the same for
        every column because ``X`` is shared.
    ``dof_scale``
        ``sqrt((T - 1) / (T - p))``, converting a ``ddof=1`` residual standard
        deviation into the regression's ``sigma``.

    A studentised statistic computed from the residuals as
    ``mean / (sd / sqrt(T))`` therefore has to be divided by
    ``se_scale * dof_scale`` to be on the same scale as ``alpha / se(alpha)``.
    """
    common = panel.index.intersection(factors.index)
    y = panel.loc[common]
    f = factors.loc[common, factor_cols]
    fmask = np.isfinite(f.to_numpy()).all(axis=1)
    y = y.loc[fmask]
    f = f.loc[fmask]

    keep = np.isfinite(y.to_numpy()).all(axis=0)
    y = y.loc[:, keep]
    if y.shape[1] == 0:
        raise ValueError("no complete-history columns")

    x = np.column_stack([np.ones(len(f)), f.to_numpy(dtype=np.float64)])
    yv = y.to_numpy(dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, yv, rcond=None)
    resid = yv - x @ coef

    t_obs, p = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    se_scale = float(np.sqrt(t_obs * xtx_inv[0, 0]))
    dof_scale = float(np.sqrt((t_obs - 1) / (t_obs - p)))
    return pd.DataFrame(resid, index=y.index, columns=y.columns), se_scale, dof_scale


def two_sided_pvalues(t: np.ndarray, dof: Optional[np.ndarray] = None) -> np.ndarray:
    """Two-sided p-values from t-statistics.

    Uses the normal approximation when ``dof`` is ``None``.  With 700 monthly
    observations the difference from the t distribution is in the fifth
    decimal place, but the published-predictor samples are shorter, so the
    re-evaluation script passes the degrees of freedom explicitly.
    """
    from scipy import stats

    t = np.asarray(t, dtype=float)
    if dof is None:
        return 2.0 * stats.norm.sf(np.abs(t))
    return 2.0 * stats.t.sf(np.abs(t), df=np.asarray(dof, dtype=float))
