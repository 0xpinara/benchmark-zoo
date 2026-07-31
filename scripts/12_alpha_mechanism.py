"""The alpha mechanism: why a null return is not a null alpha.

One claim, measured seven ways.  Least squares gives

    alpha_hat = rbar - beta_hat' fbar                                    (*)

exactly, on the regression's own sample.  So a strategy with no predictive
content has an expected raw return of zero and an expected alpha of
``-beta' fbar``, which is zero only if the strategy carries no exposure to a
priced factor.  A population that is null by construction is therefore null
with respect to the mean return and *not* with respect to the alpha, and the
alphas it produces are real rather than spurious -- which is why no
multiplicity correction addresses them.

The seven analyses, each writing one block of ``data/results/alpha_mechanism.json``:

A1 Dose-response across benchmark models.  K = 0 (raw mean), 1 (CAPM),
   3 (FF3), 4 (Carhart), 5 (FF5), 6 (FF5+MOM).  If (*) is the mechanism, the
   cross-sectional spread of the alpha t-statistic rises with the number of
   priced factors the sort accidentally loads on, and does not rise for a
   reason to do with the standard error.  Reported alongside: the exact
   finite-sample inflation of SE(alpha_hat) from having to estimate beta,
   which is ``sqrt(1 + fbar' S_f^-1 fbar)`` -- the squared tangency Sharpe
   ratio of the factor set, *not* a quantity that shrinks with T.

A2 The slope test and the exact variance decomposition.  Regressing
   alpha_hat on -beta_hat' fbar is often said to have a population slope of
   exactly 1, on the grounds that rbar is centred at zero.  That is wrong:
   the slope is ``1 + Cov(rbar, -beta' fbar)/Var(beta' fbar)``, so 1 requires
   the mean return to be orthogonal to the exposure, which holds
   equal-weighted and fails value-weighted.  What the identity does deliver
   without further assumptions is
   ``Var(alpha) = Var(rbar) + Var(beta' fbar) - 2 Cov(rbar, beta' fbar)``,
   exact in any population, and that is the primary exhibit.

A3 Within-population dose-response.  Deciles of |beta_hat' fbar|, and
   separately of regression R^2.  The exposure binning is the one that
   identifies the mechanism: R^2 binning confounds the numerator with the
   residual variance in the denominator of the t-statistic, so SD(alpha_hat)
   in basis points is reported next to SD(t) in every bin.

A4 Divergence in T.  A true alpha is a fixed non-zero population quantity,
   so its t-statistic grows with T while a genuine null's does not.  The
   ``SD(t_alpha)^2 = 1 + cT`` form of this is reported but does not fit well,
   for a reason that is not a defect of the mechanism: the windows are nested
   and end in a decade when the factor premia were near zero, so a short
   recent window has little exposure-driven alpha to find.  The robust
   version needs no assumption about stable premia -- fit
   ``Var(alpha_hat_T) = V_inf + k/T`` and ask whether the persistent
   component V_inf is zero, which is what pure estimation noise would give.
   This is the reason no correction is a remedy: a family-wise threshold
   grows like sqrt(2 log M), so offsetting a non-zero V_inf as T grows would
   need M exponential in T.

A5 Factor attribution.  Which factor's mean return does the work, and does
   the exposure trace back to the alphabet in an interpretable way.

A6 Zero-exposure placebo.  Simulate returns with zero mean and zero
   population exposure, matched to each strategy's residual volatility, and
   run the identical regressions, under two constructions -- iid normal, and
   a bootstrap of the fitted residuals that keeps their dependence and
   non-normality.  If the regression itself inflated the t-statistic, this
   would show it.  Expected SD ~ 1.00 under both.

A7 The maximum-|t| strategy, described in words, with its raw-return t next
   to its alpha t.

A8 The remedy.  Recompute the published predictors on both statistics and
   count how many clear the alpha bar but not the raw-return bar.  Counts and
   distributions only, never names.

Body results are value-weighted; the equal-weighted mirror is computed and
stored in the same file under ``ew`` keys.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.finance import loaders, metrics, partition
from bzoo.paths import RESULTS, ensure_dirs

MIN_MONTHS = 60
SEED = 20260801

# Nested factor models, in order of K.  Every one is a published benchmark a
# referee would recognise, so the curve is a curve over real screening choices
# rather than an arbitrary sequence of regressions.
MODELS = [
    ("K0_raw_mean", []),
    ("K1_capm", ["mktrf"]),
    ("K3_ff3", ["mktrf", "smb", "hml"]),
    ("K4_carhart", ["mktrf", "smb", "hml", "umd"]),
    ("K5_ff5", ["mktrf", "smb", "hml", "rmw", "cma"]),
    ("K6_ff5_mom", ["mktrf", "smb", "hml", "rmw", "cma", "umd"]),
]
FF5 = ["mktrf", "smb", "hml", "rmw", "cma"]
WEIGHTINGS = ("vw", "ew")


# ----------------------------------------------------------------------
# Shared machinery


def regression_sample(panel: pd.DataFrame, factors: pd.DataFrame, cols: list) -> pd.Index:
    """Months a regression of ``panel`` on ``cols`` can actually use.

    Kept as its own function because every quantity below has to be computed
    on *this* sample for the identity (*) to hold to machine precision.  The
    first version of this analysis took ``rbar`` over each strategy's own
    available months and ``beta' fbar`` over the regression months, which
    differ by the six months before the factor series starts, and the identity
    then failed by up to 3 percent of a cross-sectional standard deviation --
    small enough to look like a rounding artefact and large enough to blunt
    the slope-one test in A2.
    """
    common = panel.index.intersection(factors.index)
    if not cols:
        return common
    f = factors.loc[common, cols]
    return common[np.isfinite(f.to_numpy()).all(axis=1)]


def fit_model(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    cols: list,
    min_months: int = MIN_MONTHS,
) -> pd.DataFrame:
    """Alphas, loadings and the exposure term, all on the regression sample.

    Returns the frame :func:`bzoo.finance.metrics.factor_alphas` gives, plus
    ``rbar`` (mean return on the regression sample), ``exposure``
    (``beta' fbar``, recovered as ``rbar - alpha`` so that it is exact by
    construction) and ``mean_t`` (the raw-return t-statistic on the same
    months, so that the alpha and the mean return are never compared across
    different samples).
    """
    res = metrics.factor_alphas(panel, factors, cols, min_months=min_months)
    months = regression_sample(panel, factors, cols)
    y = panel.loc[months, res.index]

    res["rbar"] = y.mean(axis=0, skipna=True)
    res["exposure"] = res["rbar"] - res["alpha"]
    n = y.notna().sum(axis=0)
    res["mean_t"] = res["rbar"] / (y.std(axis=0, ddof=1, skipna=True) / np.sqrt(n))
    return res


def se_inflation_factor(factors: pd.DataFrame, cols: list, months: pd.Index) -> float:
    """``sqrt(1 + fbar' S_f^-1 fbar)``: the price of not knowing beta.

    The (1,1) element of ``(X'X)^-1`` for ``X = [1 F]`` is
    ``(1/T)(1 + fbar' S_f^-1 fbar)`` with ``S_f`` the sample factor covariance,
    so this is exactly how much wider SE(alpha_hat) is than ``sigma_e/sqrt(T)``.
    Worth reporting because the obvious objection to A1 is that adding
    regressors inflates the alpha mechanically -- and the arithmetic says the
    opposite.  The quantity is the squared maximum Sharpe ratio attainable
    from the factors, which does *not* shrink with T; what it does is make the
    t-statistic *smaller*, so the widening A1 measures is a lower bound.
    """
    if not cols:
        return 1.0
    f = factors.loc[months, cols].to_numpy(dtype=np.float64)
    fbar = f.mean(axis=0)
    s = np.cov(f, rowvar=False, ddof=0)
    s = np.atleast_2d(s)
    return float(np.sqrt(1.0 + fbar @ np.linalg.solve(s, fbar)))


def dist_summary(t: np.ndarray) -> dict:
    """The six numbers that describe a cross-sectional null, plus its cutoff."""
    t = np.asarray(t, dtype=np.float64)
    t = t[np.isfinite(t)]
    return {
        "n": int(t.size),
        "sd": float(t.std(ddof=1)),
        # MAD-SD is robust to the tail, so a gap between it and the SD is a
        # tail-shape effect and agreement means a pure scale effect.
        "mad_sd": float(stats.median_abs_deviation(t, scale="normal")),
        "kurtosis": float(stats.kurtosis(t, fisher=False)),
        "frac_abs_gt_196": float(np.mean(np.abs(t) > 1.96)),
        "frac_abs_gt_300": float(np.mean(np.abs(t) > 3.0)),
        "cutoff_5pct": float(np.quantile(np.abs(t), 0.95)),
    }


# ----------------------------------------------------------------------
# A1


def a1_across_models(panels: dict, factors: pd.DataFrame) -> dict:
    """SD(t_alpha) as a function of the benchmark model."""
    out = {}
    for w in WEIGHTINGS:
        panel = panels[w]
        rows = []
        for name, cols in MODELS:
            res = fit_model(panel, factors, cols)
            months = regression_sample(panel, factors, cols)
            alpha = res["alpha"].to_numpy(dtype=np.float64)
            exposure = res["exposure"].to_numpy(dtype=np.float64)
            row = {
                "model": name,
                "k": len(cols),
                "factors": cols,
                "n_strategies": int(len(res)),
                "n_months_max": int(len(months)),
                **dist_summary(res["alpha_t"].to_numpy()),
                "sd_alpha_bps": float(np.nanstd(alpha, ddof=1) * 100.0),
                "median_alpha_se_bps": float(np.nanmedian(res["alpha_se"]) * 100.0),
                "sd_exposure_bps": float(np.nanstd(exposure, ddof=1) * 100.0),
                "var_share_exposure": float(
                    np.nanvar(exposure, ddof=1) / np.nanvar(alpha, ddof=1)
                ),
                "median_r2": float(np.nanmedian(res["r2"])),
                "se_inflation_from_estimating_beta": se_inflation_factor(
                    factors, cols, months
                ),
                "identity_max_abs_error": float(
                    np.nanmax(np.abs(res["rbar"] - res["alpha"] - exposure))
                ),
                # Newey-West on the *alpha*, which the paper claimed without
                # ever computing it.  Reported for every model, both signs of
                # the answer.
                "sd_alpha_t_nw": float(res["alpha_t_nw"].std(ddof=1)),
            }
            rows.append(row)
        out[w] = rows
    return out


# ----------------------------------------------------------------------
# A2


def a2_slope_one(panels: dict, factors: pd.DataFrame) -> dict:
    """Cross-sectional regression of alpha_hat on -beta_hat' fbar.

    A slope of exactly 1 is often quoted as the sharp prediction of (*), on
    the grounds that rbar is centred at zero on a null population.  It is not:
    centring fixes the *mean* of rbar and the slope depends on its
    *covariance*.  Substituting (*) gives

        slope = 1 + Cov(rbar, -beta' fbar) / Var(beta' fbar)                 (**)

    so the prediction is 1 only under the extra condition that the mean return
    is cross-sectionally orthogonal to the exposure term, which is an
    empirical claim about the population and not an algebraic consequence of
    least squares.  We report the slope, the covariance that (**) says
    accounts for any departure, and the check that the two agree.

    The identity does support one exact, assumption-free statement, and we
    report that instead as the primary exhibit:

        Var(alpha_hat) = Var(rbar) + Var(beta' fbar) - 2 Cov(rbar, beta' fbar)

    every term of which is measurable.  This is an accounting identity, it
    holds in any population, and it says exactly how much dispersion risk
    adjustment adds to the dispersion the raw return already had.
    """
    out = {}
    for w in WEIGHTINGS:
        res = fit_model(panels[w], factors, FF5)
        y = res["alpha"].to_numpy(dtype=np.float64)
        x = -res["exposure"].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(x)
        y, x = y[ok], x[ok]
        n = y.size

        xm = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(xm, y, rcond=None)
        resid = y - xm @ coef
        xtx_inv = np.linalg.pinv(xm.T @ xm)
        # White standard errors.
        meat = (xm * resid[:, None]).T @ (xm * resid[:, None])
        vcov = xtx_inv @ meat @ xtx_inv
        se = np.sqrt(np.diag(vcov))
        ss_tot = float(((y - y.mean()) ** 2).sum())

        rbar = res["rbar"].to_numpy(dtype=np.float64)[ok]
        exposure = -x
        var_alpha = float(np.var(y, ddof=1))
        var_rbar = float(np.var(rbar, ddof=1))
        var_exp = float(np.var(exposure, ddof=1))
        cov_re = float(np.cov(rbar, exposure, ddof=1)[0, 1])

        out[w] = {
            "n": int(n),
            "intercept": float(coef[0]),
            "intercept_se": float(se[0]),
            "slope": float(coef[1]),
            "slope_se_white": float(se[1]),
            "t_slope_vs_zero": float(coef[1] / se[1]),
            "t_slope_vs_one": float((coef[1] - 1.0) / se[1]),
            "r2": float(1.0 - (resid ** 2).sum() / ss_tot),
            "corr_alpha_minus_exposure": float(np.corrcoef(y, x)[0, 1]),
            # Equation (**): what the slope should be, and whether it is.
            "slope_predicted_from_covariance": float(1.0 - cov_re / var_exp),
            "slope_prediction_error": float(coef[1] - (1.0 - cov_re / var_exp)),
            # The three quantities that reconcile the correlation with the
            # variance share: Var(bf)/Var(a), corr^2, and the cross term.
            "var_share_exposure": float(var_exp / var_alpha),
            "corr_squared": float(np.corrcoef(y, x)[0, 1] ** 2),
            "cov_rbar_exposure": cov_re,
            "sd_rbar": float(np.sqrt(var_rbar)),
            # The exact variance decomposition, in bps^2, and as shares of the
            # increase in dispersion that risk adjustment produces.
            "decomposition": {
                "var_alpha": var_alpha,
                "var_rbar": var_rbar,
                "var_exposure": var_exp,
                "minus_two_cov": -2.0 * cov_re,
                "residual_of_identity": float(
                    var_alpha - (var_rbar + var_exp - 2.0 * cov_re)
                ),
                "sd_ratio_alpha_over_rbar": float(np.sqrt(var_alpha / var_rbar)),
                "increase_share_from_exposure_var": float(
                    var_exp / (var_alpha - var_rbar)
                ),
                "increase_share_from_covariance": float(
                    -2.0 * cov_re / (var_alpha - var_rbar)
                ),
            },
        }
    return out


# ----------------------------------------------------------------------
# A3


def _bin_table(res: pd.DataFrame, key: np.ndarray, n_bins: int = 10) -> list:
    edges = np.nanquantile(key, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(key, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        t = res["alpha_t"].to_numpy()[m]
        rows.append(
            {
                "bin": b + 1,
                "n": int(m.sum()),
                "key_median": float(np.nanmedian(key[m])),
                "sd_alpha_t": float(np.nanstd(t, ddof=1)),
                "sd_alpha_bps": float(np.nanstd(res["alpha"].to_numpy()[m], ddof=1) * 100.0),
                "median_alpha_se_bps": float(np.nanmedian(res["alpha_se"].to_numpy()[m]) * 100.0),
                "frac_abs_t_gt_300": float(np.nanmean(np.abs(t) > 3.0)),
                "median_r2": float(np.nanmedian(res["r2"].to_numpy()[m])),
            }
        )
    return rows


def a3_within_population(panels: dict, factors: pd.DataFrame) -> dict:
    """Deciles on |exposure| (the mechanism) and on R^2 (the confound)."""
    out = {}
    for w in WEIGHTINGS:
        res = fit_model(panels[w], factors, FF5)
        out[w] = {
            "by_abs_exposure": _bin_table(res, np.abs(res["exposure"].to_numpy())),
            "by_r2": _bin_table(res, res["r2"].to_numpy()),
        }
    return out


# ----------------------------------------------------------------------
# A4


def a4_sqrt_t(panels: dict, factors: pd.DataFrame) -> dict:
    """Does SD(t_alpha) grow like sqrt(T) while SD(t_mean) stays flat?

    Restricted to the complete-history strategies, because the fourth ticker
    letter position only exists for the most recent 600 months and mixing
    sample lengths into a test *about* sample length would be circular.
    """
    out = {}
    lengths = [120, 180, 240, 360, 480, 600, 720]
    for w in WEIGHTINGS:
        panel = panels[w]
        months = regression_sample(panel, factors, FF5)
        sub = panel.loc[months]
        complete = sub.columns[sub.notna().all(axis=0)]
        sub = sub[complete]

        rows = []
        for t_len in lengths:
            if t_len > len(sub):
                continue
            win = sub.iloc[-t_len:]
            res = fit_model(win, factors, FF5, min_months=min(MIN_MONTHS, t_len))
            rows.append(
                {
                    "T": int(t_len),
                    "n_strategies": int(len(res)),
                    "start": str(win.index[0].date()),
                    "sd_alpha_t": float(res["alpha_t"].std(ddof=1)),
                    "sd_mean_t": float(res["mean_t"].std(ddof=1)),
                    "sd_alpha_bps": float(res["alpha"].std(ddof=1) * 100.0),
                    "sd_exposure_bps": float(res["exposure"].std(ddof=1) * 100.0),
                    "frac_abs_alpha_t_gt_300": float(np.mean(np.abs(res["alpha_t"]) > 3.0)),
                }
            )

        # One free parameter: SD(t_alpha)^2 = 1 + cT, fitted through the
        # constrained intercept of 1 rather than by free OLS, because the
        # intercept is the theory's prediction and not something to estimate.
        tt = np.array([r["T"] for r in rows], dtype=float)
        vv = np.array([r["sd_alpha_t"] ** 2 - 1.0 for r in rows])
        c = float((tt @ vv) / (tt @ tt))
        pred = np.sqrt(1.0 + c * tt)
        obs = np.array([r["sd_alpha_t"] for r in rows])
        # Same fit for the mean-return t, where the theory says c = 0.
        vv_m = np.array([r["sd_mean_t"] ** 2 - 1.0 for r in rows])
        c_mean = float((tt @ vv_m) / (tt @ tt))

        # The t-statistic form of the test is fragile, because it presumes the
        # factor risk premia are the same in every window and they are not:
        # the windows are nested and end in a decade where SMB and HML earned
        # nothing, so a short recent window has little exposure-driven alpha
        # to find.  The dispersion of alpha_hat itself is the robust version
        # of the same test and needs no such assumption.  If the alphas were
        # pure estimation noise, Var(alpha_hat) would fall exactly like 1/T
        # and the persistent component V_inf would be zero.  Fit
        # Var(alpha_hat_T) = V_inf + k/T and report both.
        va = np.array([(r["sd_alpha_bps"] / 100.0) ** 2 for r in rows])
        vm = np.array([(r["sd_exposure_bps"] / 100.0) ** 2 for r in rows])
        des = np.column_stack([np.ones(len(tt)), 1.0 / tt])
        (v_inf, k_noise), *_ = np.linalg.lstsq(des, va, rcond=None)
        fit_va = des @ np.array([v_inf, k_noise])
        # What pure noise alone would predict, anchored at the shortest window.
        pure_noise = va[0] * tt[0] / tt

        out[w] = {
            "windows": rows,
            "c_hat_alpha": c,
            "c_hat_mean": c_mean,
            "predicted_sd_alpha_t": [float(p) for p in pred],
            "rmse_alpha": float(np.sqrt(np.mean((obs - pred) ** 2))),
            "max_abs_error_alpha": float(np.max(np.abs(obs - pred))),
            "persistence_fit": {
                "note": (
                    "Var(alpha_hat_T) = V_inf + k/T. V_inf is the dispersion "
                    "of the true alphas, which pure estimation noise would "
                    "leave at zero."
                ),
                "v_inf_bps_squared": float(v_inf * 1e4),
                "sd_persistent_alpha_bps": float(np.sqrt(max(v_inf, 0.0)) * 100.0),
                "k_over_T_at_720_bps_squared": float(k_noise / 720.0 * 1e4),
                "share_persistent_at_720": float(v_inf / (v_inf + k_noise / 720.0)),
                "r2": float(1.0 - ((va - fit_va) ** 2).sum() / ((va - va.mean()) ** 2).sum()),
                "observed_sd_alpha_bps": [float(np.sqrt(v) * 100.0) for v in va],
                "pure_noise_sd_alpha_bps": [float(np.sqrt(v) * 100.0) for v in pure_noise],
                "sd_exposure_bps": [float(np.sqrt(v) * 100.0) for v in vm],
            },
        }
    return out


# ----------------------------------------------------------------------
# A5


def a5_attribution(panels: dict, factors: pd.DataFrame, names: pd.DataFrame) -> dict:
    """Which factor does the work, and does the exposure trace to the alphabet?"""
    out = {}
    months_ref = regression_sample(panels["vw"], factors, FF5)
    fbar = factors.loc[months_ref, FF5].mean()

    for w in WEIGHTINGS:
        res = fit_model(panels[w], factors, FF5)
        exposure = res["exposure"].to_numpy(dtype=np.float64)
        alpha = res["alpha"].to_numpy(dtype=np.float64)
        var_exp = float(np.var(exposure, ddof=1))
        var_alpha = float(np.var(alpha, ddof=1))

        terms = {}
        for f in FF5:
            term = res[f"beta_{f}"].to_numpy(dtype=np.float64) * float(fbar[f])
            terms[f] = {
                "fbar_pct_per_month": float(fbar[f]),
                "mean_beta": float(np.nanmean(res[f"beta_{f}"])),
                "sd_beta": float(np.nanstd(res[f"beta_{f}"], ddof=1)),
                "sd_term_bps": float(np.nanstd(term, ddof=1) * 100.0),
                # Additive: these sum to 1 by construction.
                "share_of_var_exposure": float(np.cov(term, exposure, ddof=1)[0, 1] / var_exp),
                "share_of_var_alpha": float(-np.cov(term, alpha, ddof=1)[0, 1] / var_alpha),
            }

        # Tails: a positive alpha on a null strategy should come from being
        # short the priced factors.
        q = np.nanquantile(alpha, [0.01, 0.99])
        lo, hi = alpha <= q[0], alpha >= q[1]
        tails = {
            side: {
                "n": int(m.sum()),
                "mean_alpha_bps": float(np.nanmean(alpha[m]) * 100.0),
                "mean_exposure_bps": float(np.nanmean(exposure[m]) * 100.0),
                **{f"mean_beta_{f}": float(np.nanmean(res[f"beta_{f}"].to_numpy()[m])) for f in FF5},
            }
            for side, m in (("bottom_1pct", lo), ("top_1pct", hi))
        }

        # Where does the exposure come from?  The only design variables are
        # the letter position and which alphabet groups sit on each leg.
        nm = names.set_index("signalid").loc[res.index]
        long_mid = np.array([np.mean(g) for g in nm["long_groups"]])
        short_mid = np.array([np.mean(g) for g in nm["short_groups"]])
        pos = nm["letter_position"].to_numpy()
        design = np.column_stack(
            [np.ones(len(nm)), long_mid, short_mid]
            + [(pos == p).astype(float) for p in (2, 3, 4)]
        )
        design_cols = ["const", "long_group_mean", "short_group_mean", "L2", "L3", "L4"]
        alphabet = {}
        for target in ["beta_smb", "beta_hml", "exposure"]:
            yv = (
                res[target].to_numpy(dtype=np.float64)
                if target in res
                else exposure
            )
            ok = np.isfinite(yv)
            coef, *_ = np.linalg.lstsq(design[ok], yv[ok], rcond=None)
            fitted = design[ok] @ coef
            ss = float(((yv[ok] - yv[ok].mean()) ** 2).sum())
            alphabet[target] = {
                "coef": dict(zip(design_cols, [float(c) for c in coef])),
                "r2": float(1.0 - ((yv[ok] - fitted) ** 2).sum() / ss),
            }

        out[w] = {
            "factor_terms": terms,
            "sum_share_of_var_exposure": float(
                sum(v["share_of_var_exposure"] for v in terms.values())
            ),
            "alpha_tails": tails,
            "alphabet_regressions": alphabet,
        }
    return out


# ----------------------------------------------------------------------
# A6


def a6_placebo(panels: dict, factors: pd.DataFrame, n_rep: int = 5) -> dict:
    """Zero mean, zero population exposure, same residual volatility.

    Two constructions.  ``iid_normal`` draws independent normals matched to
    each strategy's fitted residual volatility: this is the textbook null and
    should give SD(t) of exactly 1 up to the degrees-of-freedom correction.
    ``residual_bootstrap`` resamples the fitted residuals themselves with
    replacement, drawing the *same* rows for every strategy, so it keeps both
    the cross-sectional dependence and the non-normality of the real residuals
    and destroys only the alignment with the factors.  If either came out
    wide, the widening in A1 would be an artefact of the regression rather
    than of the exposures, and the paper would be wrong.

    A time *permutation* is the natural-looking third option and it is wrong
    here: OLS residuals have exactly zero sample mean, so any permutation of
    them also has exactly zero mean, which forces alpha_hat = -beta' fbar with
    beta near zero and collapses SD(t) to about 0.35.  That measures the
    demeaning, not the null.  Resampling with replacement restores the
    sampling variation in the mean that the test is about.
    """
    rng = np.random.default_rng(SEED)
    out = {}
    for w in WEIGHTINGS:
        panel = panels[w]
        months = regression_sample(panel, factors, FF5)
        res = fit_model(panel, factors, FF5)
        y = panel.loc[months, res.index]
        f = factors.loc[months, FF5]

        # Fitted residuals, per strategy, on the regression sample.
        beta = res[[f"beta_{c}" for c in FF5]].to_numpy(dtype=np.float64)
        fitted = f.to_numpy(dtype=np.float64) @ beta.T + res["alpha"].to_numpy()[None, :]
        resid = y.to_numpy(dtype=np.float64) - fitted
        resid_sd = np.nanstd(resid, axis=0, ddof=len(FF5) + 1)

        block = {}
        for label in ("iid_normal", "residual_bootstrap"):
            sds, fracs = [], []
            for _ in range(n_rep):
                if label == "iid_normal":
                    sim = rng.standard_normal(resid.shape) * resid_sd[None, :]
                else:
                    draw = rng.integers(0, resid.shape[0], resid.shape[0])
                    sim = resid[draw, :]
                sim_panel = pd.DataFrame(sim, index=months, columns=res.index)
                sim_res = fit_model(sim_panel, factors, FF5)
                sds.append(float(sim_res["alpha_t"].std(ddof=1)))
                fracs.append(float(np.mean(np.abs(sim_res["alpha_t"]) > 3.0)))
            block[label] = {
                "n_replicates": n_rep,
                "sd_alpha_t_mean": float(np.mean(sds)),
                "sd_alpha_t_all": sds,
                "frac_abs_t_gt_300_mean": float(np.mean(fracs)),
            }
        block["observed_sd_alpha_t"] = float(res["alpha_t"].std(ddof=1))
        block["median_resid_sd_pct_per_month"] = float(np.nanmedian(resid_sd))
        out[w] = block
    return out


# ----------------------------------------------------------------------
# A7


def _describe_ticker(name: str) -> str:
    """Turn ``L2_lng_3_4_sht_11_12`` into a sentence."""
    p = partition.parse_ticker_name(name)
    if p is None:
        return name
    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth"}[p.letter_position]
    return (
        f"sort stocks by the {ordinal} letter of the ticker symbol into twenty "
        f"groups, buy groups {p.long_groups[0]} and {p.long_groups[1]}, "
        f"sell groups {p.short_groups[0]} and {p.short_groups[1]}"
    )


def a7_the_winner(panels: dict, factors: pd.DataFrame, names: pd.DataFrame) -> dict:
    """The single most significant no-content strategy, in words."""
    lookup = names.set_index("signalid")["signalname"]
    out = {}
    for w in WEIGHTINGS:
        res = fit_model(panels[w], factors, FF5)
        i = int(res["alpha_t"].abs().idxmax())
        r = res.loc[i]
        out[w] = {
            "signalid": i,
            "signalname": str(lookup[i]),
            "description": _describe_ticker(str(lookup[i])),
            "n_months": int(r["n_months"]),
            "alpha_bps_per_month": float(r["alpha"] * 100.0),
            "alpha_t": float(r["alpha_t"]),
            "alpha_t_nw": float(r["alpha_t_nw"]),
            "mean_return_bps_per_month": float(r["rbar"] * 100.0),
            "mean_return_t": float(r["mean_t"]),
            "exposure_bps_per_month": float(r["exposure"] * 100.0),
            "r2": float(r["r2"]),
            **{f"beta_{c}": float(r[f"beta_{c}"]) for c in FF5},
        }
    return out


# ----------------------------------------------------------------------
# A8


def a8_remedy(factors: pd.DataFrame, cutoffs: dict) -> dict:
    """The published predictors on both statistics.

    Counts and distributions only: which predictors these are is not the
    point and naming them would turn a measurement into an accusation.
    """
    panel = loaders.osap_longshort_panel(sample="original", category="Predictor")
    res = fit_model(panel, factors, FF5)
    mean_t = res["mean_t"].to_numpy(dtype=np.float64)
    alpha_t = res["alpha_t"].to_numpy(dtype=np.float64)
    ok = np.isfinite(mean_t) & np.isfinite(alpha_t)
    mean_t, alpha_t = mean_t[ok], alpha_t[ok]

    def counts(bar_alpha: float, bar_mean: float) -> dict:
        a = np.abs(alpha_t) > bar_alpha
        m = np.abs(mean_t) > bar_mean
        return {
            "bar_alpha": bar_alpha,
            "bar_mean": bar_mean,
            "n_clear_alpha": int(a.sum()),
            "n_clear_mean": int(m.sum()),
            "n_clear_both": int((a & m).sum()),
            "n_alpha_only": int((a & ~m).sum()),
            "n_mean_only": int((~a & m).sum()),
            "n_neither": int((~a & ~m).sum()),
        }

    return {
        "n_predictors": int(mean_t.size),
        "median_abs_mean_t": float(np.median(np.abs(mean_t))),
        "median_abs_alpha_t": float(np.median(np.abs(alpha_t))),
        "corr_mean_t_alpha_t": float(np.corrcoef(mean_t, alpha_t)[0, 1]),
        "at_196": counts(1.96, 1.96),
        "at_300": counts(3.0, 3.0),
        # The measured single-test cutoffs from the known-null population:
        # what a 5 percent per-test false positive rate actually costs on each
        # statistic.  This is the usable recommendation.
        "at_measured_cutoffs": counts(cutoffs["alpha_vw"], cutoffs["mean_vw"]),
        "measured_cutoffs": cutoffs,
    }


# ----------------------------------------------------------------------
# The second no-content population, recomputed


def alt_population(factors: pd.DataFrame) -> dict:
    """The higher-moment past-return strategies: the falsification test.

    The ``std``, ``skew`` and ``kurt`` strategies built only from quarters 1
    to 8 -- returns three to five years old -- are no-content by absence of a
    documented effect rather than by construction, and their sorts induce
    much less factor exposure than a ticker letter does.  If the mechanism is
    right, their alpha t-statistics should sit at 1.0 while the ticker
    population's sit at 1.4.  ``robustness.json`` currently carries NaN here
    from a stale run; this recomputes it.
    """
    names = partition.partition(loaders.load_mined_names("pastret"), "pastret")
    old = names["quarters"].apply(lambda q: max(q) <= 8)
    hm = names["root"].isin(["std", "skew", "kurt"])
    keep = set(names.loc[old & hm, "signalid"])
    ret_longrun = set(names.loc[old & (names["root"] == "ret"), "signalid"])

    out = {
        "n_higher_moment": len(keep),
        "n_ret_longrun_for_comparison": len(ret_longrun),
        "definition": (
            "roots std/skew/kurt, all quarters in 1..8 (returns three to five "
            "years old); disjoint from the 'longrun' label in Table 1, which "
            "is the ret-root version of the same horizon"
        ),
    }
    for w in WEIGHTINGS:
        panel = loaders.mined_return_panel("pastret", w)
        sub = panel.loc[:, [c for c in panel.columns if c in keep]]
        res = fit_model(sub, factors, FF5)
        sd_alpha_t = float(res["alpha_t"].std(ddof=1))
        sd_mean_t = float(res["mean_t"].std(ddof=1))
        sd_alpha = float(res["alpha"].std(ddof=1))
        sd_rbar = float(res["rbar"].std(ddof=1))
        out[w] = {
            "n": int(len(res)),
            "n_months": int(res["n_months"].median()),
            "sd_mean_t": sd_mean_t,
            "sd_alpha_t": sd_alpha_t,
            "sd_alpha_bps": sd_alpha * 100.0,
            "sd_rbar_bps": sd_rbar * 100.0,
            "sd_exposure_bps": float(res["exposure"].std(ddof=1) * 100.0),
            "median_alpha_se_bps": float(res["alpha_se"].median() * 100.0),
            "var_share_exposure": float(
                np.var(res["exposure"], ddof=1) / np.var(res["alpha"], ddof=1)
            ),
            "frac_abs_alpha_t_gt_300": float(np.mean(np.abs(res["alpha_t"]) > 3.0)),
            # The comparison that matters.  The absolute width of this
            # population's alpha null is not the right thing to set against
            # the ticker population's, because its raw-return null is narrower
            # to begin with; what is comparable across two populations is how
            # much *wider* risk adjustment makes each one.
            "widening_t_ratio": sd_alpha_t / sd_mean_t,
            "widening_numerator_ratio": sd_alpha / sd_rbar,
        }
    return out


# ----------------------------------------------------------------------


def main() -> int:
    ensure_dirs()
    factors = loaders.download_factors()
    names = partition.partition(loaders.load_mined_names("ticker"), "ticker")
    panels = {w: loaders.mined_return_panel("ticker", w) for w in WEIGHTINGS}

    results: dict = {"config": {"min_months": MIN_MONTHS, "seed": SEED}}

    steps = [
        ("a1_across_models", lambda: a1_across_models(panels, factors)),
        ("a2_slope_one", lambda: a2_slope_one(panels, factors)),
        ("a3_within_population", lambda: a3_within_population(panels, factors)),
        ("a4_sqrt_t", lambda: a4_sqrt_t(panels, factors)),
        ("a5_attribution", lambda: a5_attribution(panels, factors, names)),
        ("a6_placebo", lambda: a6_placebo(panels, factors)),
        ("a7_the_winner", lambda: a7_the_winner(panels, factors, names)),
        ("alt_population", lambda: alt_population(factors)),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===", flush=True)
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a failed check is reported, not hidden
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {results[name]['error']}", flush=True)
            continue
        print(json.dumps(results[name], indent=2)[:1800], flush=True)

    # A8 needs the measured cutoffs from A1, so it runs last.
    print("\n=== a8_remedy ===", flush=True)
    try:
        by_model = {r["model"]: r for r in results["a1_across_models"]["vw"]}
        cutoffs = {
            "alpha_vw": by_model["K5_ff5"]["cutoff_5pct"],
            "mean_vw": by_model["K0_raw_mean"]["cutoff_5pct"],
        }
        results["a8_remedy"] = a8_remedy(factors, cutoffs)
        print(json.dumps(results["a8_remedy"], indent=2)[:1800], flush=True)
    except Exception as exc:  # noqa: BLE001
        results["a8_remedy"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  FAILED: {results['a8_remedy']['error']}", flush=True)

    out = RESULTS / "alpha_mechanism.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
