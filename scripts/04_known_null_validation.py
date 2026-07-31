"""Does the known-null assumption actually hold?

The finance testbed is only useful if the ticker strategies really are null.
This script attacks that assumption from four directions and reports what it
finds, including the place where the assumption fails.

Part A: alphabeticity
    There is published evidence that a firm's position in the alphabet
    affects turnover and valuation (Itzkowitz, Itzkowitz and Rothbort 2016,
    Review of Finance 20, 663-692), so a signal built from ticker letters is
    not automatically null.  We split the population by letter position, by
    the alphabetical region of the long leg, and by ticker length, and test
    every subgroup for a systematic non-zero mean.

Part B: raw returns versus factor alphas
    A strategy with zero expected return but non-zero factor loadings has a
    non-zero factor-model alpha, by the arithmetic of the regression:
    ``alpha_hat = rbar - beta_hat' fbar``.  So "no economic content" implies a
    null *raw return*, and does not imply a null *alpha*.  This part measures
    how much of the alpha dispersion the loadings account for, which decides
    whether the wide five-factor null of Section 4 is a miscalibrated standard
    error or a real non-zero alpha.

Part C: the null maximum for the alpha
    A permutation null that imposes ``alpha = 0`` directly, by sign-flipping
    the factor-model residuals with a shared sign vector.  This keeps the
    loadings and the cross-strategy dependence fixed, and gives the
    distribution of the largest alpha t-statistic under the null.

Part D: is the calibration reproducible on disjoint subpopulations?
    The 114 placebo characteristics documented in ``SignalDoc.csv`` would have
    made a second, independently constructed null, but the public
    ``PredictorPortsFull.csv`` release contains portfolios only for the 212
    predictors, so their returns are not available.  Instead we split the
    ticker population into the four disjoint sets defined by which letter of
    the ticker does the sorting, and calibrate on each separately.  The four
    sets share months, so they are not independent samples, but they are
    disjoint sets of strategies built from different information, and a
    calibration that moved a lot across them would not be usable.  The 212
    published predictors are reported in the same table as the contrast.

Writes ``data/results/null_validation.json``.
"""

from __future__ import annotations

import json
import string
import sys
import time

import numpy as np
import pandas as pd

from bzoo.finance import loaders, metrics, partition
from bzoo.null import dependence, empirical
from bzoo.paths import INTERIM, RESULTS, ensure_dirs
from bzoo.resample.stationary import optimal_block_length, stationary_bootstrap_indices

MIN_MONTHS = 60
SEED = 20260801
N_PERM = 20000
FF5 = ["mktrf", "smb", "hml", "rmw", "cma"]


def _block_bootstrap_mean_ci(series: np.ndarray, n_boot: int = 2000) -> dict:
    """Bootstrap interval for the mean of a monthly series."""
    s = series[np.isfinite(series)]
    bl = optimal_block_length(s)
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_indices(s.size, n_boot, bl, rng)
    means = s[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "n_months": int(s.size),
        "mean": float(s.mean()),
        "block_length": float(bl),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "zero_inside": bool(lo <= 0.0 <= hi),
        "boot_t": float(s.mean() / means.std(ddof=1)) if means.std(ddof=1) > 0 else 0.0,
    }


def part_a_alphabeticity() -> dict:
    """Subgroup tests for a real alphabet effect.

    Every subgroup is aggregated into one equal-weighted monthly series across
    its strategies, and the mean of that series is tested with a block
    bootstrap over months.  Averaging first is what makes the test valid: the
    strategies inside a subgroup share the same months and are strongly
    dependent, so a test that treated them as independent observations would
    reject constantly.
    """
    names = partition.partition(loaders.load_mined_names("ticker"), "ticker")
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        cols = panel.columns
        info = names.set_index("signalid").reindex(cols)

        groups = {}
        # (1) by letter position of the ticker used for sorting
        for pos in sorted(info["letter_position"].dropna().unique()):
            groups[f"letter_position_L{int(pos)}"] = info["letter_position"] == pos
        # (2) by which end of the alphabet the long leg sits in.  Groups are
        #     alphabetical vigintiles, so group 1 is A-ish and 20 is Z-ish.
        long_mean = info["long_groups"].apply(
            lambda g: np.mean(g) if isinstance(g, tuple) else np.nan
        )
        groups["long_leg_early_alphabet"] = long_mean <= 7.0
        groups["long_leg_late_alphabet"] = long_mean >= 14.0
        groups["long_leg_middle_alphabet"] = (long_mean > 7.0) & (long_mean < 14.0)
        # (3) extreme versus interior sorts, which is where a real alphabet
        #     effect would concentrate if it exists at all
        groups["extreme_sort"] = info["long_groups"].apply(
            lambda g: isinstance(g, tuple) and (1 in g or 20 in g)
        ) | info["short_groups"].apply(
            lambda g: isinstance(g, tuple) and (1 in g or 20 in g)
        )
        groups["interior_sort"] = ~groups["extreme_sort"]

        res = {}
        for name, mask in groups.items():
            sel = panel.loc[:, mask.fillna(False).to_numpy()]
            if sel.shape[1] < 20:
                continue
            agg = sel.mean(axis=1).to_numpy()
            r = _block_bootstrap_mean_ci(agg)
            r["n_strategies"] = int(sel.shape[1])
            res[name] = r
        out[weighting] = res

    # Bonferroni over all subgroup tests, since we are running many of them.
    n_tests = sum(len(v) for v in out.values())
    flagged = []
    for weighting, res in out.items():
        for name, r in res.items():
            # Two-sided normal p-value from the bootstrap t-ratio.
            from scipy import stats as _st

            p = float(2.0 * _st.norm.sf(abs(r["boot_t"])))
            r["p_value"] = p
            r["p_bonferroni"] = min(1.0, p * n_tests)
            if r["p_bonferroni"] <= 0.05:
                flagged.append(f"{weighting}:{name}")
    out["n_subgroup_tests"] = n_tests
    out["flagged_subgroups"] = flagged
    out["passes"] = len(flagged) == 0
    return out


def part_b_factor_exposure() -> dict:
    """Is the wide five-factor null a real alpha or a bad standard error?

    OLS gives ``alpha_hat = rbar - beta_hat' fbar`` exactly.  If the raw means
    are null and the loadings are heterogeneous, the alphas inherit the
    dispersion of ``beta_hat' fbar``.  We report that dispersion next to the
    dispersion of the alphas themselves, and the loading t-statistics, so the
    reader can see which it is.
    """
    factors = loaders.download_factors()
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        ff5 = metrics.factor_alphas(panel, factors, FF5, min_months=MIN_MONTHS)
        summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)

        common = panel.index.intersection(factors.index)
        fbar = factors.loc[common, FF5].mean()
        beta_cols = [f"beta_{c}" for c in FF5]
        exposure = ff5[beta_cols].to_numpy() @ fbar.to_numpy()

        rbar = summ.reindex(ff5.index)["mean"].to_numpy()
        alpha = ff5["alpha"].to_numpy()
        recon = rbar - exposure
        ok = np.isfinite(alpha) & np.isfinite(recon)

        out[weighting] = {
            "n": int(ok.sum()),
            "sd_raw_mean": float(np.nanstd(rbar, ddof=1)),
            "sd_alpha": float(np.nanstd(alpha, ddof=1)),
            "sd_beta_times_fbar": float(np.nanstd(exposure, ddof=1)),
            "identity_max_abs_error": float(np.nanmax(np.abs(alpha[ok] - recon[ok]))),
            "corr_alpha_with_minus_exposure": float(
                np.corrcoef(alpha[ok], -exposure[ok])[0, 1]
            ),
            "share_alpha_variance_from_exposure": float(
                np.nanvar(exposure, ddof=1) / np.nanvar(alpha, ddof=1)
            ),
            "median_beta": {c: float(np.nanmedian(ff5[f"beta_{c}"])) for c in FF5},
            "sd_beta": {c: float(np.nanstd(ff5[f"beta_{c}"], ddof=1)) for c in FF5},
            "factor_means_pct_per_month": {c: float(fbar[c]) for c in FF5},
        }
    return out


def part_c_alpha_null_maximum() -> dict:
    """Permutation null for the largest five-factor alpha t-statistic.

    Sign-flipping the residuals imposes ``alpha = 0`` exactly while leaving
    the loadings and the cross-strategy residual dependence untouched.  The
    studentised statistic that :func:`bzoo.null.dependence.max_t_permutation`
    computes from residuals is larger than ``alpha / se(alpha)`` by the fixed
    factor returned by :func:`bzoo.finance.metrics.factor_residuals`, so we
    divide it back out.
    """
    factors = loaders.download_factors()
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        resid, se_scale, dof_scale = metrics.factor_residuals(panel, factors, FF5)
        arr = resid.to_numpy(dtype=np.float64)
        bl = int(round(optimal_block_length(arr.mean(axis=1))))
        scale = se_scale * dof_scale

        joint, marg = dependence.max_t_permutation(
            arr, n_perm=N_PERM, block_length=bl, seed=SEED, joint=True,
            return_marginals=True,
        )
        indep, marg_i = dependence.max_t_permutation(
            arr, n_perm=4000, block_length=bl, seed=SEED + 1, joint=False,
            return_marginals=True,
        )
        joint = joint / scale
        indep = indep / scale
        marg = marg / scale
        marg_i = marg_i / scale

        sid = dependence.effective_n_sidak(
            joint, arr.shape[1], alpha=0.05, marginal_sample=marg_i
        )
        sid_ctrl = dependence.effective_n_sidak(
            indep, arr.shape[1], alpha=0.05, marginal_sample=marg_i
        )
        out[weighting] = {
            "n_strategies": int(arr.shape[1]),
            "n_months": int(arr.shape[0]),
            "block_length": bl,
            "se_scale": se_scale,
            "dof_scale": dof_scale,
            # sd of the *absolute* statistic; for a standard normal this is
            # sqrt(1 - 2/pi) = 0.603, which is the number to compare against.
            "permutation_marginal_abs_sd": float(marg.std(ddof=1)),
            "max_alpha_t_joint": {
                f"q{int(100 * q)}": float(np.quantile(joint, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "max_alpha_t_independent": {
                f"q{int(100 * q)}": float(np.quantile(indep, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "n_eff_joint": sid["n_eff_empirical_marginal"],
            "n_eff_independent_control": sid_ctrl["n_eff_empirical_marginal"],
            "dependence_ratio": sid["n_eff_empirical_marginal"]
            / sid_ctrl["n_eff_empirical_marginal"],
        }
        np.save(INTERIM / f"maxt_ff5_ticker_{weighting}.npy", joint)
    return out


def _null_row(t: np.ndarray) -> dict:
    s = empirical.summarise_null(t, seed=SEED)
    return {
        "n": int(t.size),
        "mean": s.mean,
        "median": s.median,
        "sd": s.sd,
        "mad_sd": s.mad_sd,
        "kurtosis": s.kurtosis,
        "frac_abs_gt_196": s.frac_abs_gt_196,
        "frac_abs_gt_300": s.frac_abs_gt_300,
    }


def part_d_subpopulations() -> dict:
    """Recalibrate on the four disjoint letter-position subsets."""
    names = partition.partition(loaders.load_mined_names("ticker"), "ticker")
    factors = loaders.download_factors()
    out = {"placebo_portfolios_available": False}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        info = names.set_index("signalid").reindex(panel.columns)
        summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)
        ff5 = metrics.factor_alphas(panel, factors, FF5, min_months=MIN_MONTHS)
        pos = info["letter_position"].reindex(summ.index)

        rows = {}
        for lp in (1, 2, 3, 4):
            sel = summ.loc[pos == lp, "t_stat"].dropna().to_numpy()
            sel_a = ff5.reindex(summ.index[pos == lp])["alpha_t"].dropna().to_numpy()
            rows[f"L{lp}"] = {
                "t_stat": _null_row(sel),
                "ff5_alpha_t": _null_row(sel_a),
            }
        rows["all"] = {
            "t_stat": _null_row(summ["t_stat"].dropna().to_numpy()),
            "ff5_alpha_t": _null_row(ff5["alpha_t"].dropna().to_numpy()),
        }
        sds = [rows[f"L{lp}"]["t_stat"]["sd"] for lp in (1, 2, 3, 4)]
        sds_a = [rows[f"L{lp}"]["ff5_alpha_t"]["sd"] for lp in (1, 2, 3, 4)]
        out[weighting] = {
            "by_letter_position": rows,
            "t_stat_sd_range": [float(min(sds)), float(max(sds))],
            "ff5_alpha_t_sd_range": [float(min(sds_a)), float(max(sds_a))],
        }

    # The published predictors, for contrast: not a null population.
    doc = loaders.load_osap_signal_doc()
    pub = loaders.osap_longshort_panel(sample="original", doc=doc)
    pub_summ = metrics.summarise_panel(pub, min_months=MIN_MONTHS)
    out["published_predictors_original_sample"] = _null_row(
        pub_summ["t_stat"].dropna().to_numpy()
    )
    return out


def main() -> int:
    ensure_dirs()
    t0 = time.time()
    results = {"config": {"min_months": MIN_MONTHS, "seed": SEED, "n_perm": N_PERM}}

    print("=== Part A: alphabeticity subgroups ===", flush=True)
    results["part_a_alphabeticity"] = part_a_alphabeticity()
    a = results["part_a_alphabeticity"]
    print(
        f"  {a['n_subgroup_tests']} subgroup tests, "
        f"{len(a['flagged_subgroups'])} flagged after Bonferroni: "
        f"{a['flagged_subgroups']}",
        flush=True,
    )
    for w in ("ew", "vw"):
        for name, r in a[w].items():
            print(
                f"  {w} {name:32s} mean={r['mean']:+.4f}%/mo "
                f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                f"p_bonf={r['p_bonferroni']:.3f}",
                flush=True,
            )

    print("\n=== Part B: raw returns versus factor alphas ===", flush=True)
    results["part_b_factor_exposure"] = part_b_factor_exposure()
    for w, r in results["part_b_factor_exposure"].items():
        print(
            f"  {w}: sd(rbar)={r['sd_raw_mean']:.4f}  sd(alpha)={r['sd_alpha']:.4f}  "
            f"sd(beta'fbar)={r['sd_beta_times_fbar']:.4f}  "
            f"corr(alpha, -beta'fbar)={r['corr_alpha_with_minus_exposure']:.3f}  "
            f"var share={r['share_alpha_variance_from_exposure']:.3f}  "
            f"identity error={r['identity_max_abs_error']:.2e}",
            flush=True,
        )

    print("\n=== Part C: null maximum for the five-factor alpha ===", flush=True)
    results["part_c_alpha_null_maximum"] = part_c_alpha_null_maximum()
    for w, r in results["part_c_alpha_null_maximum"].items():
        print(
            f"  {w}: marginal sd(|t|)={r['permutation_marginal_abs_sd']:.3f}  "
            f"max q95 joint={r['max_alpha_t_joint']['q95']:.3f} "
            f"independent={r['max_alpha_t_independent']['q95']:.3f}  "
            f"n_eff={r['n_eff_joint']:.0f} of {r['n_strategies']} "
            f"(control {r['n_eff_independent_control']:.0f}, "
            f"ratio {r['dependence_ratio']:.3f})",
            flush=True,
        )

    print("\n=== Part D: recalibration on disjoint subpopulations ===", flush=True)
    results["part_d_subpopulations"] = part_d_subpopulations()
    d = results["part_d_subpopulations"]
    for w in ("ew", "vw"):
        print(
            f"  {w}: sd(t) across letter positions "
            f"{d[w]['t_stat_sd_range'][0]:.3f}-{d[w]['t_stat_sd_range'][1]:.3f}; "
            f"sd(ff5 alpha t) "
            f"{d[w]['ff5_alpha_t_sd_range'][0]:.3f}-"
            f"{d[w]['ff5_alpha_t_sd_range'][1]:.3f}",
            flush=True,
        )
    pr = d["published_predictors_original_sample"]
    print(
        f"  published predictors (original samples): n={pr['n']} "
        f"sd={pr['sd']:.3f} median={pr['median']:+.3f} "
        f"P(|t|>1.96)={pr['frac_abs_gt_196']:.3f}",
        flush=True,
    )

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out = RESULTS / "null_validation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
