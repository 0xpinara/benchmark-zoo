"""Thresholds under each null, and what happens to the 212 published predictors.

Two outputs.

First, a threshold table.  For each statistic and each weighting it lines up
five thresholds on one scale:

``nominal``            1.96, the textbook two-sided 5 percent cutoff
``calibrated``         the 95th percentile of the measured null, no
                       multiplicity correction
``bonferroni``         the nominal cutoff with the raw trial count
``max_t_independent``  the 95th percentile of the null maximum with the
                       measured marginals but dependence removed
``max_t_joint``        the same with dependence intact: the correct answer

Comparing the last two isolates dependence.  Comparing ``bonferroni`` with
``max_t_joint`` says whether the conventional correction is right, and the
answer turns out to depend on which statistic is being corrected.

Second, the 212 published predictors of Chen and Zimmermann re-evaluated
under every correction, with the null taken first as theoretical and then as
calibrated, and with the trial count varied over five orders of magnitude
rather than fixed.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.corrections import bootstrap_tests, deflation, fdr, fwer, haircuts
from bzoo.finance import loaders, metrics
from bzoo.paths import INTERIM, RESULTS, ensure_dirs
from bzoo.report import tables

MIN_MONTHS = 60
SEED = 20260801
ALPHA = 0.05
N_GRID = (10, 100, 1_000, 10_000, 100_000)
N_BOOT = 2000


def load_null_calibration() -> dict:
    en = json.loads((RESULTS / "empirical_null.json").read_text())
    nv = json.loads((RESULTS / "null_validation.json").read_text())
    return {"empirical_null": en, "null_validation": nv}


def threshold_table(cal: dict) -> pd.DataFrame:
    en, nv = cal["empirical_null"], cal["null_validation"]
    rows = []
    for weighting in ("ew", "vw"):
        blk = en[f"ticker_{weighting}"]
        dep = blk["dependence"]
        k = int(dep["n_complete"])
        bonf = float(stats.norm.isf(ALPHA / (2 * k)))

        rows.append(
            {
                "statistic": "mean return t",
                "weighting": weighting.upper(),
                "null_sd": blk["t_stat"]["summary"]["sd"],
                "nominal": 1.959964,
                "calibrated": blk["t_stat"]["thresholds"]["calibrated_t_5"],
                "bonferroni": bonf,
                "max_t_independent": dep["max_t_quantiles_independent"]["q95"],
                "max_t_joint": dep["max_t_quantiles_joint"]["q95"],
                "n_tests": k,
            }
        )
        c = nv["part_c_alpha_null_maximum"][weighting]
        rows.append(
            {
                "statistic": "five-factor alpha t",
                "weighting": weighting.upper(),
                "null_sd": blk["ff5_alpha_t"]["summary"]["sd"],
                "nominal": 1.959964,
                "calibrated": blk["ff5_alpha_t"]["thresholds"]["calibrated_t_5"],
                "bonferroni": float(
                    stats.norm.isf(ALPHA / (2 * int(c["n_strategies"])))
                ),
                "max_t_independent": c["max_alpha_t_independent"]["q95"],
                "max_t_joint": c["max_alpha_t_joint"]["q95"],
                "n_tests": int(c["n_strategies"]),
            }
        )
    df = pd.DataFrame(rows)
    df["dependence_ratio"] = df["max_t_joint"] / df["max_t_independent"]
    df["bonferroni_error"] = df["bonferroni"] / df["max_t_joint"] - 1.0
    return df


def published_statistics() -> pd.DataFrame:
    doc = loaders.load_osap_signal_doc()
    panel = loaders.osap_longshort_panel(sample="original", doc=doc)
    summ = metrics.summarise_panel(panel, min_months=MIN_MONTHS)
    meta = doc.set_index("Acronym")[["Authors", "Year", "Journal", "Cat.Economic"]]
    return summ.join(meta, how="left")


def survival_table(t_obs: np.ndarray, null_sd: float, n_total: int) -> pd.DataFrame:
    """How many published predictors survive each correction.

    ``null_sd = 1.0`` reproduces the conventional calculation; passing the
    measured value is the calibrated version.  Both are reported side by side,
    because the whole point is the difference between them.
    """
    p = 2.0 * stats.norm.sf(np.abs(t_obs) / null_sd)
    k = p.size
    rows = []
    for res in (
        fwer.bonferroni(p, ALPHA),
        fwer.sidak(p, ALPHA),
        fwer.holm(p, ALPHA),
        fdr.benjamini_hochberg(p, ALPHA),
        fdr.benjamini_yekutieli(p, ALPHA),
        fdr.storey_qvalues(p, ALPHA, seed=SEED),
    ):
        rows.append(
            {
                "method": res.method,
                "error_rate": res.error_rate,
                "n_total_tests": k,
                "n_survive": res.n_reject,
                "share_survive": res.n_reject / k,
            }
        )
    for meth in ("bonferroni", "holm", "bhy"):
        res = haircuts.harvey_liu_zhu(p, n_total_tests=n_total, method=meth, alpha=ALPHA)
        rows.append(
            {
                "method": res.method,
                "error_rate": res.error_rate,
                "n_total_tests": n_total,
                "n_survive": res.n_reject,
                "share_survive": res.n_reject / k,
            }
        )
    rows.insert(
        0,
        {
            "method": "Uncorrected",
            "error_rate": "per test",
            "n_total_tests": k,
            "n_survive": int(np.sum(p <= ALPHA)),
            "share_survive": float(np.mean(p <= ALPHA)),
        },
    )
    return pd.DataFrame(rows)


def n_sensitivity(t_obs: np.ndarray, null_sd: float) -> pd.DataFrame:
    """Survival as a function of the assumed trial count."""
    rows = []
    for n in N_GRID:
        p = 2.0 * stats.norm.sf(np.abs(t_obs) / null_sd)
        n = int(max(n, p.size))
        row = {"n_total_tests": n}
        for meth in ("bonferroni", "holm", "bhy"):
            res = haircuts.harvey_liu_zhu(p, n_total_tests=n, method=meth, alpha=ALPHA)
            row[f"survive_{meth}"] = res.n_reject
        row["t_threshold_bonferroni"] = float(
            stats.norm.isf(ALPHA / (2 * n)) * null_sd
        )
        rows.append(row)
    return pd.DataFrame(rows)


def resampling_tests() -> dict:
    """Reality Check, SPA and Romano-Wolf on the common-sample panel.

    These need one shared sample, so they run on the months in which every
    included predictor has a return.  That is a different and longer window
    than the original papers used, so the numbers are not comparable with the
    t-statistics above; they answer a different question, namely whether the
    best of the 212 beats zero once the search over all 212 is accounted for.
    """
    panel = loaders.osap_longshort_panel(sample="full")
    complete = panel.notna().all(axis=1)
    # Take the longest run of months with no missing values anywhere.
    idx = np.flatnonzero(complete.to_numpy())
    if idx.size < 120:
        # Fall back to the predictors that are available over the last 40 years.
        recent = panel.loc[panel.index >= "1985-01-01"]
        keep = recent.columns[recent.notna().all(axis=0)]
        sub = recent[keep]
    else:
        sub = panel.loc[complete]
    d = sub.to_numpy(dtype=np.float64)
    rc = bootstrap_tests.white_reality_check(
        d, n_boot=N_BOOT, scheme="stationary", alpha=ALPHA, seed=SEED
    )
    spa = bootstrap_tests.hansen_spa(
        d, n_boot=N_BOOT, scheme="stationary", alpha=ALPHA, seed=SEED
    )
    obs, cent, bl = bootstrap_tests.bootstrap_centred_matrix(
        d, n_boot=N_BOOT, scheme="stationary", seed=SEED, studentised=True
    )
    rw = fwer.romano_wolf(obs, cent, alpha=ALPHA)
    return {
        "n_months": int(d.shape[0]),
        "n_predictors": int(d.shape[1]),
        "sample_start": str(sub.index.min().date()),
        "sample_end": str(sub.index.max().date()),
        "block_length": float(bl),
        "reality_check_p": rc.extra["p_value"],
        "spa_p_consistent": spa.extra["p_value"],
        "spa_p_lower": spa.extra["p_lower"],
        "spa_p_upper": spa.extra["p_upper"],
        "romano_wolf_n_reject": rw.n_reject,
        "romano_wolf_critical_value": rw.critical_value,
        "best_predictor": str(sub.columns[int(np.argmax(d.mean(axis=0)))]),
    }


def main() -> int:
    ensure_dirs()
    cal = load_null_calibration()
    results = {}

    thr = threshold_table(cal)
    print(thr.to_string(index=False, float_format=lambda x: f"{x:8.3f}"), flush=True)
    results["thresholds"] = thr.to_dict(orient="records")

    pub = published_statistics()
    t_obs = pub["t_stat"].dropna().to_numpy()
    results["published"] = {
        "n": int(t_obs.size),
        "median_t": float(np.median(np.abs(t_obs))),
        "share_abs_t_gt_196": float(np.mean(np.abs(t_obs) > 1.96)),
        "share_abs_t_gt_300": float(np.mean(np.abs(t_obs) > 3.0)),
    }
    print(f"\npublished predictors: {results['published']}", flush=True)

    # Calibrated null scale for the mean-return t-statistic.  The published
    # long-short portfolios are equal-weighted, so the equal-weighted ticker
    # calibration is the matching one; the value-weighted number is reported as
    # a sensitivity.
    sd_ew = cal["empirical_null"]["ticker_ew"]["t_stat"]["summary"]["sd"]
    sd_vw = cal["empirical_null"]["ticker_vw"]["t_stat"]["summary"]["sd"]

    surv = {}
    for label, sd in (
        ("theoretical", 1.0),
        ("calibrated_ew", sd_ew),
        ("calibrated_vw", sd_vw),
    ):
        tab = survival_table(t_obs, sd, n_total=1_000)
        surv[label] = tab.to_dict(orient="records")
        print(f"\n--- survival, null sd = {sd:.3f} ({label}) ---", flush=True)
        print(tab.to_string(index=False), flush=True)
    results["survival"] = surv
    results["null_sd_used"] = {"ew": sd_ew, "vw": sd_vw}

    sens = {}
    for label, sd in (("theoretical", 1.0), ("calibrated_ew", sd_ew)):
        tab = n_sensitivity(t_obs, sd)
        sens[label] = tab.to_dict(orient="records")
        print(f"\n--- N sensitivity, null sd = {sd:.3f} ({label}) ---", flush=True)
        print(tab.to_string(index=False), flush=True)
    results["n_sensitivity"] = sens

    print("\n--- resampling tests on the common-sample panel ---", flush=True)
    results["resampling"] = resampling_tests()
    print(json.dumps(results["resampling"], indent=2), flush=True)

    hc = haircuts.haircuts(
        np.abs(t_obs),
        n_total_tests=1_000,
        names=list(pub["t_stat"].dropna().index),
        method="bhy",
    )
    results["haircut_summary"] = {
        "median_haircut": float(hc["haircut"].median()),
        "median_haircut_of_survivors": float(
            hc.loc[hc["survives_05"], "haircut"].median()
        ),
        "n_survive": int(hc["survives_05"].sum()),
    }
    hc.to_parquet(INTERIM / "published_haircuts.parquet")
    print(f"\nhaircuts: {results['haircut_summary']}", flush=True)

    out = RESULTS / "reevaluation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
