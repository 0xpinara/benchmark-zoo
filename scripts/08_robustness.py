"""The robustness suite: eight checks, all reported whatever they show.

Each item corresponds to one of the eight in the research plan.

R1 Expanding-window calibration, to guard against calibrating and evaluating
   on the same sample.
R2 Alternative mined population: the past-return strategies at horizons where
   the literature reports no effect, used as a second null.
R3 Bootstrap block length and replicate count sensitivity.
R4 Joint versus independent resampling, on both testbeds.
R5 Closed-form deflation versus the permutation alternative.
R6 Exclusion of ticker subgroups that could carry an alphabet effect.
R7 Subperiod and volatility-regime splits.
R8 Metric choice on the machine learning side: accuracy, macro-F1 and
   balanced accuracy from the same stored predictions.

Writes ``data/results/robustness.json``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.corrections import deflation
from bzoo.finance import loaders, metrics, partition
from bzoo.ml import nulls
from bzoo.null import dependence, empirical
from bzoo.paths import INTERIM, RESULTS, ensure_dirs
from bzoo.resample import instance as inst
from bzoo.resample.permutation import sign_flip_replicates
from bzoo.resample.stationary import optimal_block_length

MIN_MONTHS = 60
SEED = 20260801
FF5 = ["mktrf", "smb", "hml", "rmw", "cma"]
DATASETS = ("cora", "citeseer", "pubmed", "ogbn-arxiv")


def r1_expanding_window() -> dict:
    """Calibrate on an early window, evaluate on a later one.

    If the null estimated before 1995 predicts the exceedance rates observed
    after 1995, the calibration is not an artifact of fitting and testing on the
    same months.  Reported as the ratio of observed to predicted exceedances,
    where 1.0 is perfect.
    """
    out = {}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        cut = pd.Timestamp("1995-01-01")
        early = panel.loc[panel.index < cut]
        late = panel.loc[panel.index >= cut]
        s_early = metrics.summarise_panel(early, min_months=MIN_MONTHS)
        s_late = metrics.summarise_panel(late, min_months=MIN_MONTHS)
        t_early = s_early["t_stat"].dropna().to_numpy()
        t_late = s_late["t_stat"].dropna().to_numpy()
        sd_early = float(t_early.std(ddof=1))

        rows = []
        for c in (1.96, 2.5, 3.0):
            observed = float(np.mean(np.abs(t_late) > c))
            predicted_cal = float(2.0 * stats.norm.sf(c / sd_early))
            predicted_nom = float(2.0 * stats.norm.sf(c))
            rows.append(
                {
                    "cutoff": c,
                    "observed_late": observed,
                    "predicted_from_early_calibration": predicted_cal,
                    "predicted_from_nominal_null": predicted_nom,
                    "ratio_calibrated": observed / predicted_cal
                    if predicted_cal > 0
                    else float("nan"),
                    "ratio_nominal": observed / predicted_nom
                    if predicted_nom > 0
                    else float("nan"),
                }
            )
        # Which null is closer, summarised so the direction is not left to the
        # reader to work out: mean absolute log ratio, where 0 is perfect.
        def _err(key: str) -> float:
            vals = [r[key] for r in rows if np.isfinite(r[key]) and r[key] > 0]
            return float(np.mean(np.abs(np.log(vals)))) if vals else float("nan")

        err_cal, err_nom = _err("ratio_calibrated"), _err("ratio_nominal")
        out[weighting] = {
            "cut": str(cut.date()),
            "n_months_early": int(len(early)),
            "n_months_late": int(len(late)),
            "sd_early": sd_early,
            "sd_late": float(t_late.std(ddof=1)),
            "exceedances": rows,
            "mean_abs_log_error_calibrated": err_cal,
            "mean_abs_log_error_nominal": err_nom,
            "calibrated_closer": bool(err_cal < err_nom),
        }
    out["calibrated_closer_in_both_weightings"] = bool(
        all(out[w]["calibrated_closer"] for w in ("ew", "vw"))
    )
    out["interpretation"] = (
        "The scale of the null is not stable across halves of the sample: it "
        "falls from 0.96 to 0.86 equal-weighted and rises from 0.91 to 1.02 "
        "value-weighted. A calibration fitted on the first half therefore "
        "transfers imperfectly to the second, and the summary above says which "
        "null is closer in each weighting rather than asserting that the "
        "calibrated one always is."
    )
    return out


def r2_alternative_population() -> dict:
    """A second null from the past-return family.

    The ``std``, ``skew`` and ``kurt`` strategies built only from the oldest
    quarters (three to five years back) have no documented effect and are a
    different construction from ticker letters.  They are a weaker null than the
    ticker set, because absence of a documented effect is not the same as
    absence by construction, and the comparison is reported as corroboration.
    """
    names = partition.partition(loaders.load_mined_names("pastret"), "pastret")
    old = names["quarters"].apply(lambda q: max(q) <= 8)
    higher_moment = names["root"].isin(["std", "skew", "kurt"])
    keep_ids = set(names.loc[old & higher_moment, "signalid"])

    factors = loaders.download_factors()
    out = {"n_signals": int(len(keep_ids))}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("pastret", weighting)
        sub = panel.loc[:, [c for c in panel.columns if c in keep_ids]]
        s = metrics.summarise_panel(sub, min_months=MIN_MONTHS)
        ff5 = metrics.factor_alphas(sub, factors, FF5, min_months=MIN_MONTHS)
        out[weighting] = {
            "n": int(len(s)),
            "t_stat_sd": float(s["t_stat"].dropna().std(ddof=1)),
            "t_stat_mean": float(s["t_stat"].dropna().mean()),
            "ff5_alpha_t_sd": float(ff5["alpha_t"].dropna().std(ddof=1)),
            "frac_abs_t_gt_196": float(np.mean(np.abs(s["t_stat"].dropna()) > 1.96)),
        }
    return out


def r3_block_and_replicates() -> dict:
    """Does the calibrated maximum move with the block length or replicate count?"""
    panel = loaders.mined_return_panel("ticker", "ew")
    arr = panel.to_numpy(dtype=np.float64)
    arr = arr[:, np.isfinite(arr).all(axis=0)]
    auto = float(optimal_block_length(arr.mean(axis=1)))

    rows = []
    for bl in (1, 2, 3, 6, 12):
        m = dependence.max_t_permutation(arr, n_perm=4000, block_length=bl, seed=SEED)
        rows.append(
            {
                "block_length": bl,
                "max_t_q95": float(np.quantile(m, 0.95)),
                "max_t_median": float(np.median(m)),
            }
        )
    reps = []
    for n in (500, 2000, 8000, 20000):
        m = dependence.max_t_permutation(arr, n_perm=n, block_length=3, seed=SEED)
        reps.append({"n_permutations": n, "max_t_q95": float(np.quantile(m, 0.95))})
    q95 = [r["max_t_q95"] for r in rows]
    return {
        "automatic_block_length": auto,
        "block_length_sensitivity": rows,
        "replicate_sensitivity": reps,
        "q95_range_over_block_lengths": [float(min(q95)), float(max(q95))],
        "q95_spread_pct": float(100 * (max(q95) - min(q95)) / np.mean(q95)),
    }


def r4_joint_vs_independent() -> dict:
    """The joint-versus-independent comparison on both testbeds."""
    out = {}

    panel = loaders.mined_return_panel("ticker", "ew")
    arr = panel.to_numpy(dtype=np.float64)
    arr = arr[:, np.isfinite(arr).all(axis=0)]
    joint = dependence.max_t_permutation(arr, n_perm=4000, block_length=3, seed=SEED)
    indep = dependence.max_t_permutation(
        arr, n_perm=4000, block_length=3, seed=SEED, joint=False
    )
    out["finance"] = {
        "n_strategies": int(arr.shape[1]),
        "max_t_q95_joint": float(np.quantile(joint, 0.95)),
        "max_t_q95_independent": float(np.quantile(indep, 0.95)),
        "threshold_ratio": float(np.quantile(joint, 0.95) / np.quantile(indep, 0.95)),
    }

    ml = {}
    for ds in DATASETS:
        path = INTERIM / f"ml_runs_{ds}.parquet"
        if not path.exists():
            continue
        runs = pd.read_parquet(path)
        correct = np.load(INTERIM / f"ml_correct_{ds}.npy")
        pops = nulls.build_null_populations(runs)
        pop = pops["pooled"]
        n_boot = 4000 if correct.shape[1] <= 5_000 else 1000
        j = nulls.bootstrap_sigma_delta(
            correct, pop.baseline_run, pop.run_index, n_boot=n_boot, seed=SEED,
            joint=True,
        )
        i = nulls.bootstrap_sigma_delta(
            correct, pop.baseline_run, pop.run_index, n_boot=n_boot, seed=SEED,
            joint=False,
        )
        ml_extra = {"n_bootstrap": n_boot}
        ml[ds] = {
            **ml_extra,
            "sigma_delta_joint": j["median_sd_delta"],
            "sigma_delta_independent": i["median_sd_delta"],
            "ratio": j["median_sd_delta"] / i["median_sd_delta"],
            "pairwise_correlation_joint": j["mean_pairwise_correlation"],
            "pairwise_correlation_independent": i["mean_pairwise_correlation"],
        }
    out["ml"] = ml
    return out


def r5_closed_form_vs_permutation() -> dict:
    """Deflation threshold from the closed form against the permutation null."""
    out = {}
    for ds in DATASETS:
        path = INTERIM / f"ml_runs_{ds}.parquet"
        if not path.exists():
            continue
        runs = pd.read_parquet(path)
        correct = np.load(INTERIM / f"ml_correct_{ds}.npy")
        pops = nulls.build_null_populations(runs)
        pop = pops["pooled"]
        sigma = pop.summary()["sd_delta"]
        scores = correct[[pop.baseline_run] + list(pop.run_index)].T.astype(float)
        diffs = inst.paired_differences(scores, baseline_col=0)
        rng = np.random.default_rng(SEED)
        perm_max = sign_flip_replicates(diffs, 8000, rng).max(axis=1)
        n = max(2, diffs.shape[1])
        out[ds] = {
            "n_candidates": int(n),
            "sigma_delta_across_run": sigma,
            "permutation_threshold_q95": float(np.quantile(perm_max, 0.95)),
            "closed_form_threshold": deflation.deflated_threshold(sigma, n, 0.05),
            "permutation_expected_max": float(perm_max.mean()),
            "closed_form_expected_max": deflation.expected_max_normal(sigma, n),
            "note": (
                "The permutation null holds every model fixed and resamples the "
                "sign of the per-instance difference, so it contains test-set "
                "noise only.  The closed form is fed the across-run spread, "
                "which also contains real differences in model quality.  A gap "
                "between the two is therefore expected and is itself the "
                "measurement: it is how much of the spread is real."
            ),
        }
    return out


def r6_exclude_alphabet_subgroups() -> dict:
    """Recalibrate after dropping every strategy whose sort touches group 1 or 20.

    Those are the extreme ends of the alphabet, which is where a real
    alphabeticity effect would concentrate.  If dropping them leaves the
    calibration unchanged, the calibration does not depend on them.
    """
    names = partition.partition(loaders.load_mined_names("ticker"), "ticker")
    touches_extreme = names.apply(
        lambda r: (1 in r["long_groups"] or 20 in r["long_groups"]
                   or 1 in r["short_groups"] or 20 in r["short_groups"]),
        axis=1,
    )
    interior_ids = set(names.loc[~touches_extreme, "signalid"])
    factors = loaders.download_factors()
    out = {"n_interior": int(len(interior_ids)), "n_total": int(len(names))}
    for weighting in ("ew", "vw"):
        panel = loaders.mined_return_panel("ticker", weighting)
        full = metrics.summarise_panel(panel, min_months=MIN_MONTHS)
        sub = panel.loc[:, [c for c in panel.columns if c in interior_ids]]
        interior = metrics.summarise_panel(sub, min_months=MIN_MONTHS)
        ff5_full = metrics.factor_alphas(panel, factors, FF5, min_months=MIN_MONTHS)
        ff5_int = metrics.factor_alphas(sub, factors, FF5, min_months=MIN_MONTHS)
        out[weighting] = {
            "t_stat_sd_all": float(full["t_stat"].dropna().std(ddof=1)),
            "t_stat_sd_interior": float(interior["t_stat"].dropna().std(ddof=1)),
            "ff5_alpha_t_sd_all": float(ff5_full["alpha_t"].dropna().std(ddof=1)),
            "ff5_alpha_t_sd_interior": float(ff5_int["alpha_t"].dropna().std(ddof=1)),
        }
    return out


def r7_subperiods() -> dict:
    """Decade and volatility-regime splits, restated compactly from script 03."""
    en = json.loads((RESULTS / "empirical_null.json").read_text())
    out = {}
    for weighting in ("ew", "vw"):
        cond = en[f"ticker_{weighting}"]["conditional"]
        dec = {r["group"]: r["sd"] for r in cond["decade"]}
        vol = {r["group"]: r["sd"] for r in cond["volatility"]}
        out[weighting] = {
            "decade_sd": dec,
            "decade_sd_range": [min(dec.values()), max(dec.values())],
            "volatility_sd": vol,
            "volatility_sd_ratio": (
                vol.get("high_vol", float("nan")) / vol.get("low_vol", float("nan"))
            ),
        }
    return out


def _macro_f1_and_balanced(correct_row: np.ndarray, labels: np.ndarray) -> "tuple[float, float]":
    """Balanced accuracy and a class-averaged accuracy from a correctness vector.

    The stored record is per-node correctness, not the predicted label, so a
    true macro-F1 is not recoverable.  What is recoverable is per-class recall,
    hence balanced accuracy, which is the metric that changes most under class
    imbalance.  We report it as the metric-choice robustness check and say
    plainly that macro-F1 would need the predictions themselves.
    """
    recalls = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 0:
            continue
        recalls.append(float(correct_row[m].mean()))
    bal = float(np.mean(recalls))
    return bal, bal


def r8_metric_choice() -> dict:
    """Does the conclusion hold under balanced accuracy as well as accuracy?"""
    from bzoo.ml import loaders as mll

    out = {}
    for ds in DATASETS:
        path = INTERIM / f"ml_runs_{ds}.parquet"
        if not path.exists():
            continue
        runs = pd.read_parquet(path)
        correct = np.load(INTERIM / f"ml_correct_{ds}.npy")
        data = mll.load_dataset(ds)
        test_labels = data.labels[data.test_idx]
        if correct.shape[1] != test_labels.size:
            out[ds] = {"available": False, "reason": "test index length mismatch"}
            continue

        bal = np.array([_macro_f1_and_balanced(correct[i], test_labels)[0]
                        for i in range(correct.shape[0])])
        runs = runs.assign(balanced_accuracy=bal)
        pops_acc = nulls.build_null_populations(runs)
        base = pops_acc["pooled"].baseline_run
        pooled_rows = pops_acc["pooled"].run_index
        d_acc = pops_acc["pooled"].delta
        d_bal = bal[pooled_rows] - bal[base]
        out[ds] = {
            "available": True,
            "sigma_delta_accuracy": float(np.std(d_acc, ddof=1)),
            "sigma_delta_balanced_accuracy": float(np.std(d_bal, ddof=1)),
            "ratio": float(np.std(d_bal, ddof=1) / np.std(d_acc, ddof=1)),
            "spearman_between_deltas": float(stats.spearmanr(d_acc, d_bal).statistic),
            "note": (
                "Macro-F1 needs the predicted labels, which the stored record "
                "does not keep; balanced accuracy is recoverable from "
                "per-instance correctness and per-instance true labels."
            ),
        }
    return out


def main() -> int:
    ensure_dirs()
    results = {"config": {"seed": SEED, "min_months": MIN_MONTHS}}
    steps = [
        ("r1_expanding_window", r1_expanding_window),
        ("r2_alternative_population", r2_alternative_population),
        ("r3_block_and_replicates", r3_block_and_replicates),
        ("r4_joint_vs_independent", r4_joint_vs_independent),
        ("r5_closed_form_vs_permutation", r5_closed_form_vs_permutation),
        ("r6_exclude_alphabet_subgroups", r6_exclude_alphabet_subgroups),
        ("r7_subperiods", r7_subperiods),
        ("r8_metric_choice", r8_metric_choice),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===", flush=True)
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a failed check is reported, not hidden
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {results[name]['error']}", flush=True)
            continue
        print(json.dumps(results[name], indent=2)[:2600], flush=True)

    out = RESULTS / "robustness.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
