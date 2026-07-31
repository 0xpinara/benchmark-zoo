"""Estimate the empirical null from the known-null population.

This is the calibration step, and the one place in either literature where
the answer can be checked against ground truth: the ticker-symbol strategies
have no economic content by construction, so their cross-sectional
distribution of t-statistics is the null distribution, not an estimate of it.

Four things get measured, and they are kept apart because they have
different consequences for a correction:

1. **Scale.**  The standard deviation of the null t-statistics against the
   nominal 1.0.
2. **Shape.**  A generalised Pareto fit to the upper tail, and exceedance
   probabilities at fixed cutoffs against the Gaussian.
3. **Dependence.**  The eigenvalue spectrum, and the effective number of
   independent tests implied by the distribution of the maximum.
4. **Stability.**  All of the above by decade and by market volatility
   regime.  If the scale moves with volatility then a single unconditional
   threshold is the wrong object.

Writes ``data/results/empirical_null.json`` plus the tables and the null
statistics themselves, which later scripts reuse.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from bzoo.finance import loaders, metrics, partition
from bzoo.null import dependence, empirical, tails
from bzoo.paths import INTERIM, RESULTS, ensure_dirs

MIN_MONTHS = 60
SEED = 20260801
N_PERM = 20000
N_PERM_INDEP = 4000  # independent-sign control; costlier per replicate
N_BOOT_MAXT = 400  # bootstrap cross-check on the permutation route
CUTOFFS = (2.0, 2.5, 3.0, 3.5, 4.0)
ALPHAS = (0.10, 0.05, 0.01)


def strategy_statistics(population: str, weighting: str) -> pd.DataFrame:
    """Per-strategy statistics, including factor alphas."""
    panel = loaders.mined_return_panel(population, weighting)
    summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)
    factors = loaders.download_factors()

    capm = metrics.factor_alphas(panel, factors, ["mktrf"], min_months=MIN_MONTHS)
    ff5 = metrics.factor_alphas(
        panel, factors, ["mktrf", "smb", "hml", "rmw", "cma"], min_months=MIN_MONTHS
    )
    out = summ.join(
        capm[["alpha", "alpha_t", "alpha_t_nw"]].add_prefix("capm_"), how="left"
    ).join(ff5[["alpha", "alpha_t", "alpha_t_nw"]].add_prefix("ff5_"), how="left")
    out["population"] = population
    out["weighting"] = weighting
    return out


def volatility_regimes(factors: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Split months into high and low market volatility.

    The regime indicator uses a trailing 24-month standard deviation of the
    market excess return, split at its own median.  Trailing rather than
    contemporaneous, so the label is knowable in real time, which matters if
    a conditional threshold is ever to be used rather than just measured.
    """
    vol = factors["mktrf"].rolling(24, min_periods=18).std()
    vol = vol.reindex(index)
    cut = vol.median()
    lab = pd.Series(np.where(vol > cut, "high_vol", "low_vol"), index=index)
    lab[vol.isna()] = "unknown"
    return lab


def summarise_statistic(t: np.ndarray, name: str) -> dict:
    s = empirical.summarise_null(t, seed=SEED)
    vi = empirical.variance_inflation(t, n_boot=1000, seed=SEED)
    exc = empirical.exceedance_table(t, cutoffs=CUTOFFS)
    stab = tails.threshold_stability(np.abs(t))
    tail_cmp = tails.tail_comparison(t, cutoffs=CUTOFFS + (4.5, 5.0), quantile=0.95)
    thresholds = {
        f"calibrated_t_{int(100 * a)}": empirical.calibrated_threshold(t, alpha=a)
        for a in ALPHAS
    }
    return {
        "statistic": name,
        "summary": s.to_dict(),
        "variance_inflation": vi,
        "exceedance": exc.to_dict(orient="records"),
        "tail_stability": stab.to_dict(orient="records"),
        "tail_comparison": tail_cmp.to_dict(orient="records"),
        "thresholds": thresholds,
    }


def main() -> int:
    ensure_dirs()
    t0 = time.time()
    results: dict = {
        "config": {
            "min_months": MIN_MONTHS,
            "seed": SEED,
            "n_perm": N_PERM,
            "n_boot_maxt": N_BOOT_MAXT,
            "cutoffs": list(CUTOFFS),
        }
    }

    factors = loaders.download_factors()
    stats_frames = []

    for weighting in ("ew", "vw"):
        print(f"\n=== ticker, {weighting} ===", flush=True)
        stats = strategy_statistics("ticker", weighting)
        stats_frames.append(stats)
        print(f"  {len(stats)} strategies with >= {MIN_MONTHS} months", flush=True)

        block = {}
        for stat_name, col in (
            ("t_stat", "t_stat"),
            ("t_stat_nw", "t_stat_nw"),
            ("capm_alpha_t", "capm_alpha_t"),
            ("ff5_alpha_t", "ff5_alpha_t"),
        ):
            vals = stats[col].dropna().to_numpy()
            block[stat_name] = summarise_statistic(vals, stat_name)
            s = block[stat_name]["summary"]
            print(
                f"  {stat_name:14s} n={s['n']:6d} sd={s['sd']:.3f} "
                f"mad_sd={s['mad_sd']:.3f} kurt={s['kurtosis']:.2f} "
                f"P(|t|>1.96)={s['frac_abs_gt_196']:.4f} "
                f"P(|t|>3)={s['frac_abs_gt_300']:.5f}",
                flush=True,
            )

        # --- dependence
        panel = loaders.mined_return_panel("ticker", weighting)
        panel = panel.loc[:, panel.notna().sum() >= MIN_MONTHS]
        arr = panel.to_numpy(dtype=np.float64)
        complete = np.isfinite(arr).all(axis=0)
        arr_c = arr[:, complete]
        print(
            f"  dependence on {arr_c.shape[1]} complete-history strategies, "
            f"{arr_c.shape[0]} months",
            flush=True,
        )

        ev = dependence.correlation_eigenvalues(arr_c)
        eig = dependence.effective_n_eigen(ev)

        from bzoo.resample.stationary import optimal_block_length

        bl = int(round(optimal_block_length(arr_c.mean(axis=1))))

        # Joint sign flips: the correct scheme.  Independent sign flips: the
        # same marginals with the dependence removed.  The gap between the two
        # maxima is dependence and nothing else.
        perm_max, perm_marg = dependence.max_t_permutation(
            arr_c, n_perm=N_PERM, block_length=bl, seed=SEED, joint=True,
            return_marginals=True,
        )
        indep_max, indep_marg = dependence.max_t_permutation(
            arr_c, n_perm=N_PERM_INDEP, block_length=bl, seed=SEED + 1, joint=False,
            return_marginals=True,
        )
        marg_sd = float(block["t_stat"]["summary"]["sd"])
        sidak = dependence.effective_n_sidak(
            perm_max, arr_c.shape[1], alpha=0.05, marginal_sd=marg_sd,
            marginal_sample=indep_marg,
        )
        sidak_check = dependence.effective_n_sidak(
            indep_max, arr_c.shape[1], alpha=0.05, marginal_sd=marg_sd,
            marginal_sample=indep_marg,
        )
        boot = dependence.effective_n_from_panel(
            arr_c, n_boot=N_BOOT_MAXT, block_length=bl, seed=SEED
        )
        block["dependence"] = {
            "n_complete": int(arr_c.shape[1]),
            "n_months": int(arr_c.shape[0]),
            "block_length": bl,
            "eigen": eig,
            "eigenvalues_top20": [float(x) for x in ev[:20]],
            "sidak_permutation_joint": sidak,
            "sidak_permutation_independent_control": sidak_check,
            "sidak_bootstrap": boot,
            "permutation_marginal_sd_joint": float(perm_marg.std(ddof=1)),
            "permutation_marginal_sd_independent": float(indep_marg.std(ddof=1)),
            "max_t_quantiles_joint": {
                f"q{int(100 * q)}": float(np.quantile(perm_max, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "max_t_quantiles_independent": {
                f"q{int(100 * q)}": float(np.quantile(indep_max, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
        }
        print(
            f"  block length {bl}; n_eff out of {arr_c.shape[1]}: "
            f"Cheverud-Nyholt {eig['n_eff_cheverud_nyholt']:.0f}, "
            f"Li-Ji {eig['n_eff_li_ji']:.0f}, "
            f"Sidak/empirical-marginal {sidak['n_eff_empirical_marginal']:.0f} "
            f"(control on independent panel: "
            f"{sidak_check['n_eff_empirical_marginal']:.0f}), "
            f"Sidak/Gaussian-marginal {sidak['n_eff_gaussian_marginal']:.0f}",
            flush=True,
        )
        print(
            f"  max|t| q95: joint {np.quantile(perm_max, 0.95):.3f}, "
            f"independent {np.quantile(indep_max, 0.95):.3f}; "
            f"marginal sd joint {perm_marg.std(ddof=1):.3f}, "
            f"independent {indep_marg.std(ddof=1):.3f}",
            flush=True,
        )
        np.save(INTERIM / f"maxt_perm_ticker_{weighting}.npy", perm_max)
        np.save(INTERIM / f"maxt_indep_ticker_{weighting}.npy", indep_max)
        np.save(INTERIM / f"marg_perm_ticker_{weighting}.npy", indep_marg)

        # --- conditional: by decade and by volatility regime
        cond = {}
        dec = (panel.index.year // 10) * 10
        by_decade = {}
        for d in sorted(set(dec)):
            sub = panel.loc[dec == d]
            if sub.shape[0] < MIN_MONTHS:
                continue
            s = metrics.summarise_panel(sub, min_months=MIN_MONTHS)
            by_decade[f"{d}s"] = s["t_stat"].dropna().to_numpy()
        cond["decade"] = empirical.conditional_summary(by_decade, seed=SEED).to_dict(
            orient="records"
        )

        regime = volatility_regimes(factors, panel.index)
        by_regime = {}
        for r in ("low_vol", "high_vol"):
            sub = panel.loc[regime == r]
            if sub.shape[0] < MIN_MONTHS:
                continue
            s = metrics.summarise_panel(sub, min_months=MIN_MONTHS)
            by_regime[r] = s["t_stat"].dropna().to_numpy()
        cond["volatility"] = empirical.conditional_summary(
            by_regime, seed=SEED
        ).to_dict(orient="records")
        block["conditional"] = cond
        for row in cond["decade"]:
            print(
                f"  decade {row['group']}: n={row['n']} sd={row['sd']:.3f} "
                f"kurt={row['kurtosis']:.2f}",
                flush=True,
            )
        for row in cond["volatility"]:
            print(
                f"  regime {row['group']}: n={row['n']} sd={row['sd']:.3f} "
                f"kurt={row['kurtosis']:.2f}",
                flush=True,
            )

        results[f"ticker_{weighting}"] = block

    all_stats = pd.concat(stats_frames)
    all_stats.to_parquet(INTERIM / "ticker_strategy_stats.parquet")
    print(f"\nwrote {INTERIM / 'ticker_strategy_stats.parquet'}")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out = RESULTS / "empirical_null.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
