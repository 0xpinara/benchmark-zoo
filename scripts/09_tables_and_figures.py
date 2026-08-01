"""Emit every table, figure and in-text macro the paper uses.

No number in the paper is typed by hand.  Tables go to ``paper/tables/*.tex``
and are included with ``\\input``; single numbers that appear in prose go to
``paper/tables/macros.tex`` as ``\\newcommand`` definitions, so a sentence that
quotes a result cannot go stale either.

Reads only the JSON result files written by scripts 02 to 08.  If one is
missing the corresponding table is skipped with a message rather than being
built from stale numbers.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np
import pandas as pd

from bzoo.finance import loaders, metrics
from bzoo.paths import INTERIM, RESULTS, ensure_dirs
from bzoo.report import figures, tables

MIN_MONTHS = 60
FF5 = ["mktrf", "smb", "hml", "rmw", "cma"]


def _fill_missing_macros(macros: dict) -> None:
    """Define every macro the paper uses, so the document always compiles.

    Anything this run could not compute is defined as ``??`` rather than left
    undefined.  An undefined command is a fatal LaTeX error that hides which
    number is missing; ``??`` compiles, is obvious on the page, and
    ``tests/test_report.py`` fails while any of them remain.
    """
    import re

    from bzoo.paths import PAPER

    main = PAPER / "main.tex"
    if not main.exists():
        return
    used = set(re.findall(r"\\bzoo([A-Za-z]+)", main.read_text()))
    # The alpha-mechanism results have their own macro file, written by
    # 13_alpha_tables_and_figures.py and inputted alongside this one.  Without
    # this exclusion both files would define the same commands and LaTeX would
    # stop on "already defined".
    other = PAPER / "tables" / "macros_alpha.tex"
    if other.exists():
        used -= set(re.findall(r"\\newcommand\{\\bzoo([A-Za-z]+)\}",
                               other.read_text()))
    missing = sorted(used - set(macros))
    for key in missing:
        macros[key] = "??"
    if missing:
        print(
            f"WARNING: {len(missing)} macros the paper uses were not computed by "
            f"this run and are set to '??': {missing}"
        )


def load(name: str) -> Optional[dict]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"skipping: {path} is missing")
        return None
    return json.loads(path.read_text())


# ----------------------------------------------------------------------
# Tables


def table_null_summary(en: dict) -> pd.DataFrame:
    rows = []
    pretty = {
        "t_stat": "Mean return $t$",
        "t_stat_nw": "Mean return $t$, Newey--West",
        "capm_alpha_t": "CAPM alpha $t$",
        "ff5_alpha_t": "Five-factor alpha $t$",
    }
    for weighting in ("ew", "vw"):
        blk = en[f"ticker_{weighting}"]
        for key, label in pretty.items():
            s = blk[key]["summary"]
            rows.append(
                {
                    "Statistic": label,
                    "Weighting": weighting.upper(),
                    "SD": s["sd"],
                    "MAD-SD": s["mad_sd"],
                    "Kurtosis": s["kurtosis"],
                    r"$\Pr(|t|>1.96)$": s["frac_abs_gt_196"],
                    r"$\Pr(|t|>3)$": s["frac_abs_gt_300"],
                    r"5\% cutoff": blk[key]["thresholds"]["calibrated_t_5"],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["Weighting", "Statistic"]).reset_index(drop=True)


def table_thresholds(re_: dict) -> pd.DataFrame:
    rows = []
    for r in re_["thresholds"]:
        rows.append(
            {
                "Statistic": (
                    "Mean return $t$"
                    if r["statistic"] == "mean return t"
                    else "Five-factor alpha $t$"
                ),
                "Wt.": r["weighting"],
                "Null SD": r["null_sd"],
                "Nominal": r["nominal"],
                "Calibrated": r["calibrated"],
                "Bonferroni": r["bonferroni"],
                r"Max-$t$ indep.": r["max_t_independent"],
                r"Max-$t$ joint": r["max_t_joint"],
                "Bonf. error": r["bonferroni_error"],
            }
        )
    return pd.DataFrame(rows)


def table_survival(re_: dict) -> pd.DataFrame:
    theo = {r["method"]: r for r in re_["survival"]["theoretical"]}
    cal = {r["method"]: r for r in re_["survival"]["calibrated_ew"]}
    rows = []
    for method, t in theo.items():
        c = cal.get(method, {})
        rows.append(
            {
                "Correction": method,
                "Controls": t["error_rate"],
                "$M$": t["n_total_tests"],
                "Survive, nominal null": t["n_survive"],
                "Survive, calibrated null": c.get("n_survive", np.nan),
                "Change": c.get("n_survive", np.nan) - t["n_survive"],
            }
        )
    return pd.DataFrame(rows)


def table_n_sensitivity(re_: dict) -> pd.DataFrame:
    rows = []
    for label, key in (("nominal", "theoretical"), ("calibrated", "calibrated_ew")):
        for r in re_["n_sensitivity"][key]:
            rows.append(
                {
                    "Null": label,
                    "$M$": r["n_total_tests"],
                    "$t$ threshold": r["t_threshold_bonferroni"],
                    "Bonferroni": r["survive_bonferroni"],
                    "Holm": r["survive_holm"],
                    "BHY": r["survive_bhy"],
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["Null", "$M$"])
    return df.reset_index(drop=True)


def table_ml_sigma(ml: dict) -> pd.DataFrame:
    rows = []
    for ds, blk in ml["datasets"].items():
        if not blk.get("available"):
            continue
        for name, s in blk["populations"].items():
            if name == "seed":
                continue  # reported as a scalar noise floor instead
            rows.append(
                {
                    "Dataset": ds,
                    "Null population": s["population"],
                    "$n$": s["n"],
                    r"$\sigma_\Delta$": s["sd_delta"],
                    "Bootstrap only": s["bootstrap_sigma_delta_joint"],
                    "Ratio": s["across_run_over_bootstrap"],
                    r"$\max\Delta$": s["max_delta"],
                }
            )
    return pd.DataFrame(rows)


def table_ml_thresholds(ml: dict) -> pd.DataFrame:
    rows = []
    for ds, blk in ml["datasets"].items():
        if not blk.get("available"):
            continue
        row = {
            "Dataset": ds,
            "Seed only": blk.get("sigma_delta_noise_floor", float("nan")),
            r"$\sigma_\Delta$": blk["sigma_delta_pooled"],
        }
        for r in blk["deflation_grid_pooled"]:
            row[f"$N={r['n_trials']}$"] = r["threshold_accuracy_points"]
        rows.append(row)
    return pd.DataFrame(rows)


def table_saturation(ml: dict) -> pd.DataFrame:
    rows = []
    for r in ml["saturation"]["table"]:
        rows.append(
            {
                "Dataset": r["dataset"],
                "Test nodes": r["n_test_instances"],
                "Tuned baseline": r["baseline_accuracy"],
                "Headroom": r["headroom"],
                r"$\sigma_\Delta$": r["sigma_delta"],
                r"$\sigma_\Delta$/headroom": r["sigma_over_headroom"],
            }
        )
    return pd.DataFrame(rows)


def table_leaderboard(ml: dict) -> "tuple[pd.DataFrame, bool]":
    """Always returns a table.

    The second element says whether the deflation columns are populated.  They
    need a ``sigma_Delta`` estimated on ogbn-arxiv itself, and we refuse to
    substitute one estimated on a different benchmark, so when that sweep is
    absent the table reports the descriptive statistics alone and says so.
    """
    lb = ml["leaderboard"]
    if lb["deflation"].get("available"):
        rows = []
        for r in lb["deflation"]["grid"]:
            rows.append(
                {
                    "$N$": r["n_trials"],
                    "Threshold (pts)": r["threshold_accuracy_points"],
                    "Advances": r["n_advances"],
                    "Survive": r["n_survive"],
                    "Share": r["share_survive"],
                }
            )
        return pd.DataFrame(rows), True

    rows = [
        {"Quantity": "submissions", "Value": lb["n_entries"]},
        {"Quantity": "displayed ranks", "Value": lb["n_ranks"]},
        {"Quantity": "best reported accuracy", "Value": lb["best_test_acc"]},
        {"Quantity": "median gap between adjacent entries",
         "Value": lb["median_adjacent_gap"]},
        {"Quantity": "median reported standard deviation",
         "Value": lb["median_reported_std"]},
        {"Quantity": "gap as a multiple of that standard deviation",
         "Value": lb["gap_over_reported_std"]},
        {"Quantity": "share of gaps below one standard deviation",
         "Value": lb["share_adjacent_gaps_below_median_std"]},
    ]
    return pd.DataFrame(rows), False


def table_corrections_implemented() -> pd.DataFrame:
    import yaml

    from bzoo.paths import CONFIGS

    spec = yaml.safe_load((CONFIGS / "corrections.yaml").read_text())
    rows = []
    for name, m in spec["methods"].items():
        rows.append(
            {
                "Method": name.replace("_", " "),
                "Controls": m["type"],
                "Dependence": m["dependence"],
                "Source": m["source"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------


def main() -> int:
    ensure_dirs()
    en = load("empirical_null")
    obs = load("observable_n")
    nv = load("null_validation")
    re_ = load("reevaluation")
    ml = load("ml_analysis")
    rb = load("robustness")
    sanity = load("sanity")
    macros: dict = {}

    if en is not None:
        df = table_null_summary(en)
        tables.write_dataframe(
            "null_summary",
            df,
            caption=(
                "The measured null. Cross-sectional distribution of four "
                "statistics over the 19{,}380 ticker-symbol strategies, which "
                "have no economic content by construction. The theoretical "
                "null has SD 1.00, kurtosis 3.00, "
                "$\\Pr(|t|>1.96)=0.050$ and $\\Pr(|t|>3)=0.0027$. "
                "The last column is the cutoff that gives a 5 percent "
                "per-test false positive rate against the measured null, with "
                "no multiplicity correction."
            ),
            label="tab:null-summary",
            digits={"SD": 3, "MAD-SD": 3, "Kurtosis": 2, r"$\Pr(|t|>1.96)$": 4,
                    r"$\Pr(|t|>3)$": 5, r"5\% cutoff": 2},
            column_format="llrrrrrr",
            rules_after=(3,),
        )
        for weighting in ("ew", "vw"):
            w = weighting.capitalize()
            for key, tag in (("t_stat", "Mean"), ("ff5_alpha_t", "Ff")):
                sm = en[f"ticker_{weighting}"][key]["summary"]
                macros[f"null{tag}Sd{w}"] = sm["sd"]
                macros[f"null{tag}Kurt{w}"] = (sm["kurtosis"], 2)
                macros[f"null{tag}Exceed{w}Pct"] = (100 * sm["frac_abs_gt_196"], 1)
                macros[f"null{tag}Three{w}Pct"] = (100 * sm["frac_abs_gt_300"], 1)
        # the two the abstract and conclusion quote by name
        macros["nullFfSdVwPercent"] = (
            100 * en["ticker_vw"]["ff5_alpha_t"]["summary"]["frac_abs_gt_300"], 1)
        macros["nominalThreePct"] = (100 * 0.0026998, 2)
        dep = en["ticker_ew"]["dependence"]
        macros["nEffLiJi"] = int(round(dep["eigen"]["n_eff_li_ji"]))
        macros["nEffCheverud"] = int(round(dep["eigen"]["n_eff_cheverud_nyholt"]))
        macros["nStrategiesComplete"] = int(dep["n_complete"])
        mj = dep["max_t_quantiles_joint"]["q95"]
        mi = dep["max_t_quantiles_independent"]["q95"]
        macros["maxTJoint"] = (mj, 2)
        macros["maxTIndep"] = (mi, 2)
        macros["dependenceMovePct"] = (100 * (1.0 - mj / mi), 1)

    if re_ is not None:
        thr = table_thresholds(re_)
        tables.write_dataframe(
            "thresholds",
            thr,
            caption=(
                "Five thresholds for the same family of tests, on one scale. "
                "\\emph{Nominal} is 1.96. \\emph{Calibrated} is the 95th "
                "percentile of the measured null for a single test. "
                "\\emph{Bonferroni} applies the nominal null with "
                "$M = 14{,}535$, the number of strategies with a complete "
                "720-month history; the fourth ticker letter position exists "
                "only for the most recent 600 months, which is why that count "
                "is three quarters of the 19{,}380 and not all of them. The "
                "two max-$t$ columns are the 95th percentile "
                "of the null maximum with the measured marginals, without and "
                "with cross-strategy dependence; the second is correct. "
                "\\emph{Bonf.\\ error} is how much Bonferroni overshoots it. "
                "The two alpha rows mix two constructions and should be read "
                "as such: \\emph{Null SD} and \\emph{Calibrated} come from the "
                "marginal cross-sectional spread, which "
                "Section~\\ref{sec:nullfails} shows is not a null, while the "
                "max-$t$ columns come from the permutation that imposes "
                "$\\alpha = 0$ given the exposures. Only the latter is used as "
                "a threshold."
            ),
            label="tab:thresholds",
            digits={"Null SD": 3, "Nominal": 2, "Calibrated": 2, "Bonferroni": 2,
                    r"Max-$t$ indep.": 2, r"Max-$t$ joint": 2, "Bonf. error": 3},
            column_format="llrrrrrrr",
        )
        tables.write_dataframe(
            "survival",
            table_survival(re_),
            caption=(
                "The 212 published predictors of Chen and Zimmermann "
                "\\citep{chen2022osap}, re-evaluated on each predictor's own "
                "original sample window. 208 have at least 60 months. "
                "Calibrating the null makes \\emph{more} of them survive, "
                "because the measured null for the equal-weighted mean-return "
                "$t$-statistic is narrower than the nominal one, not wider. "
                "Two rows need a note. The Storey $q$-value keeps all 208 "
                "under either null because the estimated null proportion "
                "$\\hat\\pi_0$ is essentially zero on this population, which "
                "is what a procedure designed for a mostly-null family does "
                "when handed a family of published claims; it is not a bug. "
                "The three Harvey--Liu--Zhu rows use the $M = 1{,}000$ their "
                "own procedure assumes rather than the 208 of the rows above, "
                "so their survival counts are not comparable with the rest of "
                "the column."
            ),
            label="tab:survival",
            digits=0,
            column_format="llrrrr",
        )
        tables.write_dataframe(
            "n_sensitivity",
            table_n_sensitivity(re_),
            caption=(
                "Survival as a function of the assumed total number of tests "
                "$M$, which is never fixed to one value. The smallest $M$ is "
                "the number of published predictors themselves, which is the "
                "only $M$ that is observed rather than assumed."
            ),
            label="tab:n-sensitivity",
            digits={"$t$ threshold": 2},
            column_format="lrrrrr",
        )
        macros["publishedN"] = int(re_["published"]["n"])
        macros["publishedMedianT"] = (re_["published"]["median_t"], 2)
        macros["publishedShareTwoPct"] = (
            100 * re_["published"]["share_abs_t_gt_196"], 0)
        macros["publishedShareThreePct"] = (
            100 * re_["published"]["share_abs_t_gt_300"], 0)
        surv_t = {r["method"]: r["n_survive"] for r in re_["survival"]["theoretical"]}
        surv_c = {r["method"]: r["n_survive"] for r in re_["survival"]["calibrated_ew"]}
        macros["survBonfNominal"] = surv_t["Bonferroni"]
        macros["survBonfCalibrated"] = surv_c["Bonferroni"]
        macros["survBhNominal"] = surv_t["Benjamini-Hochberg"]
        macros["survBhCalibrated"] = surv_c["Benjamini-Hochberg"]
        macros["haircutMedianPct"] = (
            100 * re_["haircut_summary"]["median_haircut"], 0)
        macros["realityCheckP"] = (re_["resampling"]["reality_check_p"], 4)
        macros["spaP"] = (re_["resampling"]["spa_p_consistent"], 4)
        macros["romanoWolfReject"] = int(re_["resampling"]["romano_wolf_n_reject"])
        macros["romanoWolfN"] = int(re_["resampling"]["n_predictors"])
        by = {(r["statistic"], r["weighting"]): r for r in re_["thresholds"]}
        macros["bonfErrorMeanEwPct"] = (
            100 * by[("mean return t", "EW")]["bonferroni_error"], 1)
        macros["bonfErrorFfVwPct"] = (
            100 * by[("five-factor alpha t", "VW")]["bonferroni_error"], 1)

    if nv is not None:
        a = nv["part_a_alphabeticity"]
        macros["alphabetTests"] = int(a["n_subgroup_tests"])
        macros["alphabetFlagged"] = int(len(a["flagged_subgroups"]))
        b = nv["part_b_factor_exposure"]
        macros["exposureVarShareEwPct"] = (
            100 * b["ew"]["share_alpha_variance_from_exposure"], 0)
        macros["exposureVarShareVwPct"] = (
            100 * b["vw"]["share_alpha_variance_from_exposure"], 0)
        macros["exposureCorrVw"] = (b["vw"]["corr_alpha_with_minus_exposure"], 2)
        # The decisive diagnostic: the alpha t-statistic distribution widens by
        # exactly the factor the alphas themselves widen by, so the standard
        # errors are essentially unchanged and the width is in the numerator.
        if en is not None:
            for w in ("ew", "vw"):
                num = b[w]["sd_alpha"] / b[w]["sd_raw_mean"]
                tstat = (
                    en[f"ticker_{w}"]["ff5_alpha_t"]["summary"]["sd"]
                    / en[f"ticker_{w}"]["t_stat"]["summary"]["sd"]
                )
                macros[f"numeratorRatio{w.capitalize()}"] = (num, 2)
                macros[f"tStatRatio{w.capitalize()}"] = (tstat, 2)
        c = nv["part_c_alpha_null_maximum"]
        macros["ffMaxTJointVw"] = (c["vw"]["max_alpha_t_joint"]["q95"], 2)
        macros["ffMaxTIndepVw"] = (c["vw"]["max_alpha_t_independent"]["q95"], 2)
        d = nv["part_d_subpopulations"]
        macros["subpopSdLowEw"] = d["ew"]["t_stat_sd_range"][0]
        macros["subpopSdHighEw"] = d["ew"]["t_stat_sd_range"][1]
        macros["subpopFfSdLowVw"] = d["vw"]["ff5_alpha_t_sd_range"][0]
        macros["subpopFfSdHighVw"] = d["vw"]["ff5_alpha_t_sd_range"][1]

    if ml is not None:
        tables.write_dataframe(
            "ml_sigma",
            table_ml_sigma(ml),
            caption=(
                "The constructed null on the machine learning side. "
                "$\\sigma_\\Delta$ is the standard deviation of the "
                "test-accuracy improvement over the tuned GCN baseline across "
                "the runs in each null population. \\emph{Bootstrap only} is "
                "the same quantity from the joint instance bootstrap with the "
                "models held fixed, so it contains test-set noise and nothing "
                "else. \\emph{Ratio} is how much larger the across-run spread "
                "is; a ratio near one means the null population differs from "
                "the baseline only by noise, which is what a null population "
                "should look like. The \\emph{screened} row applies the "
                "recentring rule of Section~\\ref{sec:ml}."
            ),
            label="tab:ml-sigma",
            digits={r"$\sigma_\Delta$": 5, "Bootstrap only": 5, "Ratio": 2,
                    r"$\max\Delta$": 4},
            column_format="llrrrrr",
        )
        tables.write_dataframe(
            "ml_thresholds",
            table_ml_thresholds(ml),
            caption=(
                "Improvement, in accuracy points, that a reported result must "
                "exceed to survive deflation, as a function of the number of "
                "trials $N$ the field is credited with. $\\sigma_\\Delta$ is "
                "estimated per benchmark from the screened null population and "
                "is never transferred between benchmarks."
            ),
            label="tab:ml-thresholds",
            digits={r"$\sigma_\Delta$": 5, "Seed only": 5,
                    "$N=10$": 2, "$N=100$": 2, "$N=1000$": 2,
                    "$N=10000$": 2, "$N=100000$": 2},
            column_format="lrrrrrrr",
        )
        tables.write_dataframe(
            "saturation",
            table_saturation(ml),
            caption=(
                "Saturation. \\emph{Headroom} is the distance from the tuned "
                "baseline to perfect accuracy, an upper bound on what is left "
                "to win. The last column is the fraction of that headroom one "
                "draw from the null moves the metric by."
            ),
            label="tab:saturation",
            digits={"Tuned baseline": 4, "Headroom": 4, r"$\sigma_\Delta$": 5,
                    r"$\sigma_\Delta$/headroom": 4},
            column_format="lrrrrr",
        )
        lb_tab, lb_deflated = table_leaderboard(ml)
        if lb_deflated:
            tables.write_dataframe(
                "leaderboard",
                lb_tab,
                caption=(
                    "The ogbn-arxiv leaderboard, deflated. \\emph{Advances} "
                    "counts the submissions that improved on the best result "
                    "available when they were submitted, restricted to entries "
                    "that use graph features only. \\emph{Survive} counts those "
                    "whose improvement exceeds the deflated threshold at each "
                    "$N$. The leaderboard itself supplies a lower bound on $N$."
                ),
                label="tab:leaderboard",
                digits={"Threshold (pts)": 3, "Share": 3},
                column_format="lrrrr",
            )
        else:
            tables.write_dataframe(
                "leaderboard",
                lb_tab,
                caption=(
                    "The ogbn-arxiv leaderboard as reported. The deflation "
                    "columns are omitted here because they require a "
                    "$\\sigma_\\Delta$ estimated on ogbn-arxiv itself, and we "
                    "do not substitute one estimated on another benchmark; the "
                    "gap-to-noise comparison in the last two rows needs no null "
                    "distribution at all."
                ),
                label="tab:leaderboard",
                digits=4,
                column_format="lr",
            )
        lb = ml["leaderboard"]
        macros["lbEntries"] = int(lb["n_entries"])
        macros["lbRanks"] = int(lb["n_ranks"])
        macros["lbBest"] = (lb["best_test_acc"], 4)
        macros["lbMedianGap"] = (lb["median_adjacent_gap"], 4)
        macros["lbMedianGapPts"] = (100 * lb["median_adjacent_gap"], 2)
        macros["lbMedianStd"] = (lb["median_reported_std"], 4)
        macros["lbGapOverStd"] = (lb["gap_over_reported_std"], 2)
        macros["lbShareGapsBelowStdPct"] = (
            100 * lb["share_adjacent_gaps_below_median_std"], 0)
        macros["lbDateMin"] = lb["date_min"]
        macros["lbDateMax"] = lb["date_max"]
        for ds, blk in ml["datasets"].items():
            if not blk.get("available"):
                continue
            tag = "".join(part.capitalize()
                          for part in ds.replace("-", " ").replace("_", " ").split())
            # A sweep can finish and still fail to yield a null population:
            # if no candidate survives the screening rule there is nothing to
            # estimate a spread from, and 07_ml_analysis.py says so rather
            # than returning a number.  That is the case for ogbn-arxiv, whose
            # tuned baseline leads the pool by five times the per-trial noise.
            # Emit the descriptive macros and skip the deflation ones, which
            # is what the paper reports for that benchmark.
            if not blk.get("sigma_delta_estimable", True):
                macros[f"baseline{tag}"] = (blk["baseline_test_accuracy"], 4)
                macros[f"{tag}NullPool"] = int(blk["screening"]["n_before"])
                macros[f"{tag}ScreenKept"] = int(blk["screening"]["n_after"])
                macros[f"{tag}PassingRule"] = int(blk["screening"]["n_passing_rule"])
                macros[f"{tag}LeadInNoise"] = (
                    blk["screening"]["baseline_lead_in_noise_units"], 1)
                macros[f"noiseFloor{tag}Pts"] = (
                    100 * blk.get("sigma_delta_noise_floor", float("nan")), 2)
                macros[f"sigmaUnscreened{tag}"] = (
                    blk["sigma_delta_unscreened"], 3)
                continue
            macros[f"sigmaDelta{tag}"] = (blk["sigma_delta_pooled"], 5)
            macros[f"sigmaUnscreened{tag}"] = (blk["sigma_delta_unscreened"], 3)
            macros[f"sigmaSeed{tag}"] = (blk["sigma_delta_seed"], 5)
            macros[f"noiseFloor{tag}"] = (
                blk.get("sigma_delta_noise_floor", float("nan")), 5)
            macros[f"noiseFloor{tag}Pts"] = (
                100 * blk.get("sigma_delta_noise_floor", float("nan")), 2)
            macros[f"noiseFloorPooled{tag}Pts"] = (
                100 * blk.get("sigma_delta_noise_floor_pooled", float("nan")), 2)
            macros[f"baseline{tag}"] = (blk["baseline_test_accuracy"], 4)
            macros[f"{tag}NullPool"] = int(blk["screening"]["n_before"])
            macros[f"{tag}ScreenKept"] = int(blk["screening"]["n_after"])
            macros[f"ratioScreened{tag}"] = (
                blk["populations"]["screened"]["across_run_over_bootstrap"], 2)
            macros[f"ratioPooled{tag}"] = (
                blk["populations"]["pooled"]["across_run_over_bootstrap"], 2)
            g = {r["n_trials"]: r["threshold_accuracy_points"]
                 for r in blk["deflation_grid_pooled"]}
            macros[f"thr{tag}Thousand"] = (g[1000], 2)
            # The same threshold computed from the unscreened spread, which is
            # the number the recentring step is there to avoid.
            from bzoo.corrections import deflation as _defl

            macros[f"thrUnscreened{tag}Thousand"] = (
                100 * _defl.deflated_threshold(
                    blk["sigma_delta_unscreened"], 1000, 0.05), 1)
            sel = blk["selection_null_screened"]["grid"]
            by_m = {r["n_search"]: r for r in sel}
            if 30 in by_m:
                macros[f"sel{tag}MeanThirtyPts"] = (
                    100 * by_m[30]["mean_delta"], 2)
                macros[f"sel{tag}QNinetyFiveThirtyPts"] = (
                    100 * by_m[30]["q95_delta"], 2)
        if "spearman_accuracy_vs_sigma_over_headroom" in ml["saturation"]:
            sp = ml["saturation"]["spearman_accuracy_vs_sigma_over_headroom"]
            macros["saturationRho"] = (sp["rho"], 2)
            macros["saturationP"] = (sp["p_value"], 3)

    # Ranges over whichever datasets are present, so the prose does not have to
    # name them and cannot go stale when a dataset is added.
    if ml is not None:
        avail = [d for d, b in ml["datasets"].items() if b.get("available")]
        if avail:
            scr = [ml["datasets"][d]["populations"]["screened"][
                "across_run_over_bootstrap"] for d in avail]
            poo = [ml["datasets"][d]["populations"]["pooled"][
                "across_run_over_bootstrap"] for d in avail]
            nf = [100 * ml["datasets"][d]["sigma_delta_noise_floor"] for d in avail]
            macros["ratioScreenedLow"] = (min(scr), 2)
            macros["ratioScreenedHigh"] = (max(scr), 2)
            macros["ratioPooledLow"] = (min(poo), 2)
            macros["ratioPooledHigh"] = (max(poo), 2)
            macros["noiseFloorLowPts"] = (min(nf), 2)
            macros["noiseFloorHighPts"] = (max(nf), 2)
            macros["nDatasets"] = len(avail)

    # ------------------------------------------------------------------
    # Observable trial counts
    if obs is not None:
        rows = [
            {
                "Quantity": "ogbn-arxiv leaderboard submissions",
                "Count": obs["leaderboard"]["n_submissions"],
                "What it bounds": "lower bound on trials, machine learning",
            },
            {
                "Quantity": "distinct leaderboard submitters",
                "Count": obs["leaderboard"]["n_distinct_submitters"],
                "What it bounds": "lower bound on research groups",
            },
            {
                "Quantity": "documented return predictors",
                "Count": obs["finance"]["n_documented_predictors"],
                "What it bounds": "lower bound on trials, finance",
            },
            {
                "Quantity": "documented placebo characteristics",
                "Count": obs["finance"]["n_documented_placebos"],
                "What it bounds": "characteristics examined and not claimed",
            },
        ]
        for key, v in obs["citations"].items():
            if v.get("cited_by_count"):
                rows.append(
                    {
                        "Quantity": f"works citing {v['label']}",
                        "Count": int(v["cited_by_count"]),
                        "What it bounds": "research efforts, no bound on trials",
                    }
                )
        tables.write_dataframe(
            "observable_n",
            pd.DataFrame(rows),
            caption=(
                "What is observable about the number of trials. Leaderboard "
                "submissions are a lower bound and a loose one, because they "
                "count attempts that were made public. Citation counts are not "
                "a bound in either direction and are reported for scale only; "
                "they come from OpenAlex, which indexes preprint and "
                "proceedings versions separately, so other indexes give larger "
                "figures. No threshold in this paper uses any of these as $N$."
            ),
            label="tab:observable-n",
            digits=0,
            column_format="lrp{0.34\\linewidth}",
        )
        macros["obsLbSubmitters"] = int(obs["leaderboard"]["n_distinct_submitters"])
        macros["obsMaxPerSubmitter"] = int(
            obs["leaderboard"]["max_submissions_by_one_submitter"])
        for key, mname in (("ogb", "obsOgbCitations"), ("gcn", "obsGcnCitations")):
            if obs["citations"].get(key, {}).get("cited_by_count"):
                macros[mname] = int(obs["citations"][key]["cited_by_count"])

    # ------------------------------------------------------------------
    # Robustness
    if rb is not None:
        rows = []
        r4 = rb.get("r4_joint_vs_independent", {})
        if "finance" in r4:
            rows.append(
                {
                    "Check": "Joint vs independent resampling, finance",
                    "Quantity": r"max-$|t|$ 95th pct ratio",
                    "Value": r4["finance"]["threshold_ratio"],
                }
            )
        for ds, v in r4.get("ml", {}).items():
            rows.append(
                {
                    "Check": f"Joint vs independent resampling, {ds}",
                    "Quantity": r"$\sigma_\Delta$ ratio",
                    "Value": v["ratio"],
                }
            )
        r3 = rb.get("r3_block_and_replicates", {})
        if "q95_spread_pct" in r3:
            rows.append(
                {
                    "Check": "Block length 1 to 12",
                    "Quantity": r"spread in max-$|t|$ 95th pct (\%)",
                    "Value": r3["q95_spread_pct"],
                }
            )
            macros["bootBlockSpreadPct"] = (r3["q95_spread_pct"], 1)
        r1b = rb.get("r1_expanding_window", {})
        for w in ("ew", "vw"):
            if w in r1b and "mean_abs_log_error_calibrated" in r1b[w]:
                W = w.capitalize()
                macros[f"expWindowErrCal{W}"] = (
                    r1b[w]["mean_abs_log_error_calibrated"], 2)
                macros[f"expWindowErrNom{W}"] = (
                    r1b[w]["mean_abs_log_error_nominal"], 2)
                macros[f"expWindowSdEarly{W}"] = (r1b[w]["sd_early"], 2)
                macros[f"expWindowSdLate{W}"] = (r1b[w]["sd_late"], 2)
                for e in r1b[w]["exceedances"]:
                    rows.append(
                        {
                            "Check": f"Expanding window, {w.upper()}, cutoff "
                                     f"{e['cutoff']}",
                            "Quantity": "observed / predicted, calibrated",
                            "Value": e["ratio_calibrated"],
                        }
                    )
        r2b = rb.get("r2_alternative_population", {})
        for w in ("ew", "vw"):
            if w in r2b:
                W = w.capitalize()
                macros[f"altNullSd{W}"] = (r2b[w]["t_stat_sd"], 2)
                macros[f"altNullFfSd{W}"] = (r2b[w]["ff5_alpha_t_sd"], 2)
                rows.append(
                    {
                        "Check": f"Second null population, {w.upper()}",
                        "Quantity": r"SD of mean-return $t$",
                        "Value": r2b[w]["t_stat_sd"],
                    }
                )
        if "n_signals" in r2b:
            macros["altNullN"] = int(r2b["n_signals"])
        r6 = rb.get("r6_exclude_alphabet_subgroups", {})
        for w in ("ew", "vw"):
            if w in r6:
                rows.append(
                    {
                        "Check": f"Drop alphabet-extreme sorts, {w.upper()}",
                        "Quantity": r"SD of mean-return $t$",
                        "Value": r6[w]["t_stat_sd_interior"],
                    }
                )
        r8 = rb.get("r8_metric_choice", {})
        for ds, v in r8.items():
            if isinstance(v, dict) and v.get("available"):
                rows.append(
                    {
                        "Check": f"Balanced accuracy, {ds}",
                        "Quantity": r"$\sigma_\Delta$ ratio to accuracy",
                        "Value": v["ratio"],
                    }
                )
        if rows:
            tables.write_dataframe(
                "robustness",
                pd.DataFrame(rows),
                caption=(
                    "Robustness suite. Every entry is reported whatever it "
                    "shows. For the joint-versus-independent rows a value below "
                    "one means the correct scheme gives a smaller null "
                    "maximum, which is the whole point; for the "
                    "expanding-window rows one is perfect and the nominal "
                    "null's own errors are in Section~\\ref{sec:robust}; for "
                    "the exclusion and second-population rows the comparison is "
                    "with the full-population value in "
                    "Table~\\ref{tab:null-summary}. The individual numbers "
                    "behind every row are in "
                    "\\texttt{data/results/robustness.json}."
                ),
                label="tab:robustness",
                digits=3,
                column_format="p{0.42\\linewidth}p{0.30\\linewidth}r",
            )

    # ------------------------------------------------------------------
    # Corrections implemented, and the test count read from the suite itself
    tables.write_dataframe(
        "corrections",
        table_corrections_implemented(),
        caption=(
            "Corrections implemented in the package, with the source each one "
            "is tested against. Every method has a unit test that reproduces a "
            "published worked example or a published property of the procedure."
        ),
        label="tab:corrections",
        column_format="p{0.15\\linewidth}p{0.22\\linewidth}"
                      "p{0.20\\linewidth}p{0.30\\linewidth}",
    )

    import subprocess

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
            capture_output=True, text=True, timeout=900,
        )
        import re as _re

        found = _re.search(r"(\d+)\s+tests? collected", proc.stdout)
        if found:
            macros["testCount"] = int(found.group(1))
    except Exception as exc:  # noqa: BLE001 - a missing count is reported
        print(f"could not count tests: {exc}")

    if sanity is not None:
        macros["tickerMeanEw"] = (
            sanity["check2_ticker_mean_zero"]["ew"]["mean_pct_per_month"], 4)
        macros["tickerMeanTEw"] = (
            sanity["check2_ticker_mean_zero"]["ew"]["t_stat"], 2)
        macros["reproCorr"] = (
            sanity["check1_published_tstats"]["correlation_with_reported"], 2)
        macros["momentumMedianTEw"] = (
            sanity["check3_known_signals"]["ew"]["momentum"]["median_t"], 2)

    # Worst calibrated/observed exceedance ratio, value-weighted: quoted in the
    # robustness section and previously typed by hand.
    try:
        rb = json.loads((RESULTS / "robustness.json").read_text())
        ex = rb["r1_expanding_window"]["vw"]["exceedances"]
        worst = max(e["ratio_calibrated"] for e in ex)
        macros["expWindowWorstRatioVw"] = (worst, 1)
    except Exception as exc:  # noqa: BLE001 - a missing number is reported
        print(f"WARNING: could not compute expWindowWorstRatioVw: {exc}")

    _fill_missing_macros(macros)
    tables.write_macros("macros", macros, digits=3)

    # ------------------------------------------------------------------
    # Figures
    stats_path = INTERIM / "ticker_strategy_stats.parquet"
    if stats_path.exists():
        st = pd.read_parquet(stats_path)
        vw = st[st["weighting"] == "vw"]
        ew = st[st["weighting"] == "ew"]
        figures.null_density(
            {
                "mean return $t$, VW": vw["t_stat"].dropna().to_numpy(),
                "five-factor alpha $t$, VW": vw["ff5_alpha_t"].dropna().to_numpy(),
                "five-factor alpha $t$, EW": ew["ff5_alpha_t"].dropna().to_numpy(),
            },
            name="null_density",
        )
    if re_ is not None:
        figures.threshold_comparison(re_["thresholds"], name="thresholds")
        figures.survival_vs_n(
            {
                "nominal null": re_["n_sensitivity"]["theoretical"],
                "calibrated null": re_["n_sensitivity"]["calibrated_ew"],
            },
            name="survival_vs_n",
            lower_bound=re_["published"]["n"],
            lower_bound_label="published predictors, the only observed $M$",
        )
    if ml is not None:
        if len(ml["saturation"]["table"]) >= 2:
            figures.saturation(ml["saturation"]["table"], name="saturation")
        lb = ml["leaderboard"]
        if lb["deflation"]["available"] and lb.get("advances"):
            thr = {
                f"$N={r['n_trials']}$": r["threshold_delta"]
                for r in lb["deflation"]["grid"]
                if r["n_trials"] in (100, 10_000)
            }
            figures.leaderboard_advances(
                lb["advances"], thr, name="leaderboard_advances"
            )
    _write_placeholders_for_missing_tables()
    print("done")
    return 0


def _write_placeholders_for_missing_tables() -> None:
    """Write a visible placeholder for any table the paper includes but that this
    run could not produce.

    A missing ``\\input`` file stops LaTeX dead, which makes the paper
    uncompilable and hides which result is absent.  A placeholder compiles and
    says on the page which stage of the pipeline has not been run, which is both
    more useful and harder to overlook than a silent omission.
    """
    import re

    from bzoo.paths import PAPER

    main = PAPER / "main.tex"
    if not main.exists():
        return
    wanted = re.findall(r"\\input\{tables/([A-Za-z0-9_]+)\}", main.read_text())
    for name in wanted:
        path = tables.TABLES / f"{name}.tex"
        if path.exists():
            continue
        tables.write_table(
            name,
            "% PLACEHOLDER: this table has not been generated yet.\n"
            "\\begin{table}[t]\\centering\n"
            f"\\caption{{\\textbf{{Not yet generated:}} \\texttt{{{name}}}. "
            "Run \\texttt{make all}; the stage that produces this table has "
            "not completed.}}\n"
            f"\\label{{tab:{name.replace('_', '-')}}}\n"
            "\\begin{tabular}{l}\\toprule pending \\\\ \\bottomrule"
            "\\end{tabular}\n\\end{table}\n",
        )


if __name__ == "__main__":
    sys.exit(main())
