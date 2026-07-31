"""Partition the mined populations and run the three sanity checks.

Nothing downstream should run until all three pass.

Check 1  The published t-statistics of the 212 predictors, recomputed from
         the distributed long-short returns over each predictor's own sample
         window, must line up with the t-statistics reported in
         ``SignalDoc.csv``.
Check 2  The ticker population's aggregate mean return must be
         indistinguishable from zero.  Tested with a block bootstrap over
         months, not over strategies, because the strategies are dependent.
Check 3  Momentum and short-term reversal must show up as significant in the
         past-return population.  If they do not, the power set is broken and
         no statement about power means anything.

Writes ``data/results/sanity.json`` and the partition table.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.finance import loaders, metrics, partition
from bzoo.paths import RESULTS, ensure_dirs
from bzoo.report import tables
from bzoo.resample.stationary import optimal_block_length, stationary_bootstrap_indices

MIN_MONTHS = 60
SEED = 20260801


def partition_report() -> "dict[str, pd.DataFrame]":
    parts = {}
    for pop in ("ticker", "pastret"):
        names = loaders.load_mined_names(pop)
        parts[pop] = partition.partition(names, pop)
    return parts


def check_published_tstats() -> dict:
    """Check 1: reproduce the reported t-statistics."""
    doc = loaders.load_osap_signal_doc()
    panel = loaders.osap_longshort_panel(sample="original", doc=doc)
    summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)

    reported = doc.set_index("Acronym")["T-Stat"]
    reported = pd.to_numeric(reported, errors="coerce").dropna()
    both = summ.join(reported.rename("t_reported"), how="inner").dropna(
        subset=["t_stat", "t_reported"]
    )
    # Reported t-statistics are signed by the original paper's convention; the
    # OSAP portfolios are already sign-aligned so that the long-short return is
    # expected positive.  Compare magnitudes.
    x = both["t_reported"].abs().to_numpy()
    y = both["t_stat"].abs().to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "n_compared": int(len(both)),
        "n_predictors_in_panel": int(panel.shape[1]),
        "correlation_with_reported": corr,
        "median_recomputed_t": float(np.median(y)),
        "median_reported_t": float(np.median(x)),
        "share_recomputed_t_above_2": float(np.mean(y > 2.0)),
        "share_reported_t_above_2": float(np.mean(x > 2.0)),
        "passes": bool(corr > 0.5 and np.median(y) > 2.0),
    }


def check_ticker_mean_is_zero(n_boot: int = 2000) -> dict:
    """Check 2: the known-null population has no aggregate edge.

    The right unit of resampling here is the month.  Averaging 19,380
    strategies within a month leaves a single time series, and its mean is
    what a reader means by "the population has no edge on average".  A
    bootstrap over strategies would give an interval that is far too narrow,
    because the strategies share the same months.
    """
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        agg = panel.mean(axis=1).to_numpy()  # equal-weighted across strategies
        agg = agg[np.isfinite(agg)]
        t_obs = float(agg.mean() / (agg.std(ddof=1) / np.sqrt(agg.size)))

        bl = optimal_block_length(agg)
        rng = np.random.default_rng(SEED)
        idx = stationary_bootstrap_indices(agg.size, n_boot, bl, rng)
        boot_means = agg[idx].mean(axis=1)
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        out[weighting] = {
            "n_months": int(agg.size),
            "n_strategies": int(panel.shape[1]),
            "mean_pct_per_month": float(agg.mean()),
            "t_stat": t_obs,
            "block_length": float(bl),
            "boot_ci_low": float(lo),
            "boot_ci_high": float(hi),
            "zero_inside_ci": bool(lo <= 0.0 <= hi),
        }
    out["passes"] = bool(all(out[w]["zero_inside_ci"] for w in ("ew", "vw")))
    return out


def check_known_signals_appear() -> dict:
    """Check 3: the three known-signal families appear with the right sign.

    The test is on the signed t-statistic against the sign the literature
    predicts (rule R9).  A family that came out significant with the wrong
    sign would mean the quarter-index convention had been misread, and that
    error would silently invert every economic statement in the paper, so it
    is worth a check that can catch it.
    """
    names = partition.partition(loaders.load_mined_names("pastret"), "pastret")
    labels = names.set_index("signalid")["label"]
    families = ("momentum", "recent", "longrun")
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("pastret", weighting)
        summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)
        lab = labels.reindex(summ.index)
        res = {}
        for group in families + ("other",):
            sel = summ.loc[lab == group, "t_stat"].dropna()
            if not len(sel):
                continue
            entry = {
                "n": int(len(sel)),
                "median_t": float(np.median(sel)),
                "min_t": float(sel.min()),
                "max_t": float(sel.max()),
            }
            if group in families:
                sign = partition.expected_sign(group)
                entry["expected_sign"] = sign
                entry["median_t_in_expected_direction"] = float(
                    sign * np.median(sel)
                )
                entry["share_significant_correct_sign"] = float(
                    np.mean(sign * sel > 2.0)
                )
            res[group] = entry
        out[weighting] = res

    # Pass criterion, in two parts:
    #  (a) all three families point the predicted way in equal-weighted
    #      returns, with a median signed t-statistic above 1.5;
    #  (b) momentum, the effect that survives value weighting in the
    #      literature, clears 2.0 in both weightings.
    # Short-term and long-run reversal are small-stock effects, so they are
    # expected to weaken under value weighting and are reported rather than
    # required.  See DECISIONS.md, 2026-08-04.
    def signed(weighting: str, group: str) -> float:
        v = out[weighting].get(group, {}).get("median_t_in_expected_direction")
        return float("nan") if v is None else float(v)

    ok_ew = all(signed("ew", g) > 1.5 for g in families)
    ok_mom = all(signed(w, "momentum") > 2.0 for w in ("ew", "vw"))
    out["criterion"] = {
        "all_families_correct_sign_ew": bool(ok_ew),
        "momentum_strong_both_weightings": bool(ok_mom),
    }
    out["passes"] = bool(ok_ew and ok_mom)
    return out


def main() -> int:
    ensure_dirs()
    parts = partition_report()
    summary = partition.partition_summary(parts)
    print(summary.to_string(index=False), flush=True)

    results = {
        "min_months": MIN_MONTHS,
        "partition": {
            pop: {
                "n_signals": int(len(df)),
                "role": str(df["role"].iloc[0]),
            }
            for pop, df in parts.items()
        },
        "rules": partition.RULES,
    }

    print("\n[check 1] reproducing published t-statistics", flush=True)
    results["check1_published_tstats"] = check_published_tstats()
    print(json.dumps(results["check1_published_tstats"], indent=2), flush=True)

    print("\n[check 2] ticker population aggregate mean", flush=True)
    results["check2_ticker_mean_zero"] = check_ticker_mean_is_zero()
    print(json.dumps(results["check2_ticker_mean_zero"], indent=2), flush=True)

    print("\n[check 3] known signals in the past-return population", flush=True)
    results["check3_known_signals"] = check_known_signals_appear()
    print(json.dumps(results["check3_known_signals"], indent=2), flush=True)

    all_pass = all(
        results[k]["passes"]
        for k in (
            "check1_published_tstats",
            "check2_ticker_mean_zero",
            "check3_known_signals",
        )
    )
    results["all_checks_pass"] = bool(all_pass)

    out = RESULTS / "sanity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    tables.write_dataframe(
        "partition",
        summary.rename(
            columns={
                "population": "Population",
                "n_signals": "Signals",
                "role": "Role",
                "detail": "Composition",
            }
        ),
        caption=(
            "Mined strategy populations from Chen and Dim, after the "
            "classification rules of Section~\\ref{sec:data}. Counts are "
            "signals; each appears in an equal-weighted and a value-weighted "
            "version, so the strategy counts are twice these. Ticker "
            "strategies are null by construction and are used to calibrate; "
            "past-return strategies contain known effects and are used to "
            "check power. The \\emph{longrun} label counts only the "
            "\\texttt{ret}-root signals at that horizon, since those are the "
            "ones with a sign the literature predicts. The second no-content "
            "population of Section~\\ref{sec:mechanism} is the disjoint set of "
            "\\texttt{std}, \\texttt{skew} and \\texttt{kurt} signals over the "
            "same quarters, of which there are 210."
        ),
        label="tab:partition",
        column_format="lrlp{0.36\\linewidth}",
    )

    print("ALL CHECKS PASS" if all_pass else "SOME CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
