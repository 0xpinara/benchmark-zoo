"""Tables, figures and macros for the alpha-mechanism results.

Reads ``data/results/alpha_mechanism.json`` and writes, into ``paper/``:

  tables/across_models.tex      A1, the dose-response across benchmark models
  tables/exposure_deciles.tex   A3, the within-population dose-response
  tables/alt_population.tex     the second no-content population
  tables/macros_alpha.tex       every number the new prose quotes
  figures/across_models.{pdf,png}
  figures/exposure_dose_response.{pdf,png}

Kept separate from ``09_tables_and_figures.py`` so that the two macro files
do not overwrite each other; ``main.tex`` inputs both.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bzoo.report import figures, tables
from bzoo.paths import RESULTS

MODEL_LABELS = {
    "K0_raw_mean": "Raw mean return",
    "K1_capm": "CAPM",
    "K3_ff3": "Fama--French 3",
    "K4_carhart": "Carhart 4",
    "K5_ff5": "Fama--French 5",
    "K6_ff5_mom": "FF5 + momentum",
}
FACTOR_LABELS = {"mktrf": "MKT", "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA"}


def table_across_models(d: dict) -> None:
    rows = []
    for w in ("vw", "ew"):
        for r in d["a1_across_models"][w]:
            rows.append(
                {
                    "Model": MODEL_LABELS[r["model"]],
                    "$K$": r["k"],
                    "Wt.": w.upper(),
                    "SD": r["sd"],
                    "MAD-SD": r["mad_sd"],
                    "Kurt.": r["kurtosis"],
                    "$\\Pr(|t|>1.96)$": r["frac_abs_gt_196"],
                    "$\\Pr(|t|>3)$": r["frac_abs_gt_300"],
                    "5\\% cutoff": r["cutoff_5pct"],
                    "SD($\\hat\\alpha$)": r["sd_alpha_bps"],
                    "Med. SE": r["median_alpha_se_bps"],
                    "SE infl.": r["se_inflation_from_estimating_beta"],
                }
            )
    df = pd.DataFrame(rows)
    tables.write_dataframe(
        "across_models",
        df,
        caption=(
            "Dose-response across benchmark models, on the 19{,}380 ticker "
            "strategies. $K$ is the number of factors; every row is the same "
            "strategies over the same 714 months, so only the model changes. "
            "SD($\\hat\\alpha$) and the median standard error are in basis "
            "points per month. \\emph{SE infl.} is "
            "$\\sqrt{1+\\bar f'S_f^{-1}\\bar f}$, the exact finite-sample cost "
            "of estimating $\\beta$ rather than knowing it, which is a "
            "Sharpe-ratio quantity and does not shrink with $T$. The spread of "
            "the alpha widens by a third while the standard error moves by "
            "about a percent, so the widening is entirely in the numerator."
        ),
        label="tab:across-models",
        digits={
            "SD": 3, "MAD-SD": 3, "Kurt.": 2, "$\\Pr(|t|>1.96)$": 4,
            "$\\Pr(|t|>3)$": 4, "5\\% cutoff": 2, "SD($\\hat\\alpha$)": 2,
            "Med. SE": 2, "SE infl.": 4, "$K$": 0,
        },
        rules_after=(5,),
    )


def table_exposure_deciles(d: dict) -> None:
    rows = []
    for r in d["a3_within_population"]["vw"]["by_abs_exposure"]:
        rows.append(
            {
                "Decile": r["bin"],
                "$n$": r["n"],
                "Median $|\\hat\\beta'\\bar f|$": r["key_median"] * 100.0,
                "SD($t_\\alpha$)": r["sd_alpha_t"],
                "SD($\\hat\\alpha$)": r["sd_alpha_bps"],
                "Median SE($\\hat\\alpha$)": r["median_alpha_se_bps"],
                "$\\Pr(|t_\\alpha|>3)$": r["frac_abs_t_gt_300"],
                "Median $R^2$": r["median_r2"],
            }
        )
    tables.write_dataframe(
        "exposure_deciles",
        pd.DataFrame(rows),
        caption=(
            "Within-population dose-response, value-weighted. Strategies are "
            "sorted into deciles by the absolute size of their exposure term "
            "$|\\hat\\beta'\\bar f|$; all quantities except the last two are in "
            "basis points per month. Across the deciles the numerator of the "
            "$t$-statistic rises by a factor of two while its denominator "
            "rises by a fifth, which is what identifies the mechanism as "
            "exposure rather than a standard-error effect. The bottom decile "
            "is a within-population placebo: strategies with almost no "
            "exposure have an alpha $t$-statistic of almost exactly the "
            "textbook width."
        ),
        label="tab:exposure-deciles",
        digits={
            "Decile": 0, "$n$": 0, "Median $|\\hat\\beta'\\bar f|$": 2,
            "SD($t_\\alpha$)": 3, "SD($\\hat\\alpha$)": 2,
            "Median SE($\\hat\\alpha$)": 2, "$\\Pr(|t_\\alpha|>3)$": 4,
            "Median $R^2$": 3,
        },
    )


def table_alt_population(d: dict) -> None:
    alt, a1 = d["alt_population"], d["a1_across_models"]
    rows = []
    for w in ("vw", "ew"):
        by = {r["model"]: r for r in a1[w]}
        rows.append(
            {
                "Population": "Ticker letters", "Wt.": w.upper(),
                "$n$": by["K5_ff5"]["n_strategies"],
                "SD($t_{\\bar r}$)": by["K0_raw_mean"]["sd"],
                "SD($t_\\alpha$)": by["K5_ff5"]["sd"],
                "Ratio": by["K5_ff5"]["sd"] / by["K0_raw_mean"]["sd"],
                "SD($\\bar r$)": by["K0_raw_mean"]["sd_alpha_bps"],
                "SD($\\hat\\alpha$)": by["K5_ff5"]["sd_alpha_bps"],
                "Ratio ": by["K5_ff5"]["sd_alpha_bps"] / by["K0_raw_mean"]["sd_alpha_bps"],
            }
        )
        rows.append(
            {
                "Population": "Higher-moment, 3--5 yr", "Wt.": w.upper(),
                "$n$": alt[w]["n"],
                "SD($t_{\\bar r}$)": alt[w]["sd_mean_t"],
                "SD($t_\\alpha$)": alt[w]["sd_alpha_t"],
                "Ratio": alt[w]["widening_t_ratio"],
                "SD($\\bar r$)": alt[w]["sd_rbar_bps"],
                "SD($\\hat\\alpha$)": alt[w]["sd_alpha_bps"],
                "Ratio ": alt[w]["widening_numerator_ratio"],
            }
        )
    tables.write_dataframe(
        "alt_population",
        pd.DataFrame(rows),
        caption=(
            "Two no-content populations, built differently, widened the same "
            "way. Dispersions of $\\bar r$ and $\\hat\\alpha$ are in basis "
            "points per month; each \\emph{Ratio} column is the alpha entry "
            "divided by the mean-return entry to its left. The second "
            "population's alpha $t$-statistics sit near the textbook width in "
            "\\emph{absolute} terms only because its raw-return null is "
            "unusually narrow to begin with: risk adjustment widens its "
            "numerator by as much as it widens the ticker population's. Two "
            "independent constructions, one mechanism."
        ),
        label="tab:alt-population",
        digits={
            "$n$": 0, "SD($t_{\\bar r}$)": 3, "SD($t_\\alpha$)": 3, "Ratio": 3,
            "SD($\\bar r$)": 2, "SD($\\hat\\alpha$)": 2, "Ratio ": 3,
        },
        rules_after=(3,),
    )


def table_corrections_on_null(c: dict) -> None:
    """The direct test: run the corrections on a population with no content."""
    rows = []
    labels = [("uncorrected", "Uncorrected ($|t|>1.96$)"),
              ("bonferroni", "Bonferroni"), ("sidak", "\\v{S}id\\'ak"),
              ("holm", "Holm"), ("benjamini_hochberg", "Benjamini--Hochberg"),
              ("benjamini_yekutieli", "Benjamini--Yekutieli")]
    vw = c["c1_survival"]["vw"]
    for key, label in labels:
        rows.append({
            "Correction": label,
            "Mean return $t$": vw["mean_return_t"]["whole_population"][key],
            "Five-factor $\\alpha$ $t$": vw["ff5_alpha_t"]["whole_population"][key],
        })
    tables.write_dataframe(
        "corrections_on_null",
        pd.DataFrame(rows),
        caption=(
            "Multiplicity corrections applied to 19{,}380 strategies that have "
            "no economic content by construction, value-weighted, with the "
            "family size set to the full 19{,}380 and the nominal standard "
            "normal used as the null. Every rejection in this table is a false "
            "one. A 5 percent family-wise procedure promises at most a 5 "
            "percent chance of \\emph{any} rejection across the family; on the "
            "raw mean return Bonferroni delivers that and rejects nothing, and "
            "on the five-factor alpha it rejects 69 times. The multiplicity "
            "arithmetic is not what fails --- the null it is applied to is."
        ),
        label="tab:corrections-on-null",
        digits=0,
        column_format="lrr",
    )


def figure_across_models(d: dict) -> None:
    figures.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    for w, mark in (("vw", "o"), ("ew", "s")):
        rows = d["a1_across_models"][w]
        ax.plot([r["k"] for r in rows], [r["sd"] for r in rows],
                marker=mark, label=w.upper())
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(6.0, 1.005, "textbook null", color="0.4", fontsize=7,
            va="bottom", ha="right")
    # Hand-placed so the labels do not collide with the guide line or with
    # each other; the Carhart point sits below its neighbours, so its label
    # goes underneath and the rest go above.
    offsets = {"K0_raw_mean": (6, 6), "K1_capm": (0, 8), "K3_ff3": (-2, 8),
               "K4_carhart": (0, -14), "K5_ff5": (-4, 8), "K6_ff5_mom": (0, -14)}
    aligns = {"K0_raw_mean": "left", "K6_ff5_mom": "right"}
    for r in d["a1_across_models"]["vw"]:
        ax.annotate(MODEL_LABELS[r["model"]].replace("--", "-"),
                    (r["k"], r["sd"]), textcoords="offset points",
                    xytext=offsets[r["model"]],
                    ha=aligns.get(r["model"], "center"), fontsize=6, color="0.3")
    ax.set_xlabel("Number of factors in the benchmark model, $K$")
    ax.set_ylabel(r"SD of the alpha $t$-statistic")
    ax.set_ylim(0.80, 1.55)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    rows = d["a3_within_population"]["vw"]["by_abs_exposure"]
    x = [r["bin"] for r in rows]
    ax.plot(x, [r["sd_alpha_bps"] for r in rows], marker="o",
            label=r"SD($\hat\alpha$), the numerator")
    ax.plot(x, [r["median_alpha_se_bps"] for r in rows], marker="s",
            label=r"median SE($\hat\alpha$), the denominator")
    ax.set_xlabel(r"Decile of $|\hat\beta'\bar f|$, value-weighted")
    ax.set_ylabel("Basis points per month")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tw = ax.twinx()
    tw.plot(x, [r["frac_abs_t_gt_300"] for r in rows], marker="^",
            color="0.35", ls=":")
    tw.set_ylabel(r"$\Pr(|t_\alpha|>3)$ (dotted)", color="0.35", fontsize=8)
    tw.tick_params(axis="y", colors="0.35")

    fig.tight_layout()
    figures.save(fig, "across_models")


def figure_exposure_dose_response(d: dict) -> None:
    """Persistence, on its own.

    This used to carry the exposure deciles in a left panel, which duplicated
    the right panel of the across-models figure and paired the paper's
    strongest exhibit with its most qualified one.  The decile comparison
    against R^2 now lives in the table, where the numbers can be read off.
    """
    figures.setup_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    p = d["a4_sqrt_t"]["vw"]["persistence_fit"]
    tt = [r["T"] for r in d["a4_sqrt_t"]["vw"]["windows"]]
    ax.plot(tt, p["observed_sd_alpha_bps"], marker="o",
            label=r"observed SD($\hat\alpha$)")
    ax.plot(tt, p["pure_noise_sd_alpha_bps"], marker="s", ls="--",
            label=r"pure estimation noise ($\propto 1/\sqrt{T}$)")
    dj = d["a4_sqrt_t"]["vw"]["disjoint"]
    ax.axhline(dj["sd_persistent_bps"], color="0.4", ls=":", lw=1)
    ax.text(tt[-1], dj["sd_persistent_bps"] + 0.4,
            "persists across decades  ", color="0.4", fontsize=7,
            va="bottom", ha="right")
    ax.set_xlabel("Window length $T$ (months, most recent)")
    ax.set_ylabel("Basis points per month")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    figures.save(fig, "exposure_dose_response")


def write_macros(d: dict, c: dict) -> None:
    vw = {r["model"]: r for r in d["a1_across_models"]["vw"]}
    ew = {r["model"]: r for r in d["a1_across_models"]["ew"]}
    a2, a4 = d["a2_slope_one"]["vw"], d["a4_sqrt_t"]["vw"]
    a3 = d["a3_within_population"]["vw"]["by_abs_exposure"]
    a5, a6, a7, a8 = (d["a5_attribution"]["vw"], d["a6_placebo"]["vw"],
                      d["a7_the_winner"]["vw"], d["a8_remedy"])
    alt = d["alt_population"]
    dc = a2["decomposition"]

    values = {
        # A1
        "amSdRawVw": vw["K0_raw_mean"]["sd"],
        "amSdCapmVw": vw["K1_capm"]["sd"],
        "amSdFfThreeVw": vw["K3_ff3"]["sd"],
        "amSdCarhartVw": vw["K4_carhart"]["sd"],
        "amSdFfFiveVw": vw["K5_ff5"]["sd"],
        "amSdFfSixVw": vw["K6_ff5_mom"]["sd"],
        "amSdRawEw": ew["K0_raw_mean"]["sd"],
        "amSdFfFiveEw": ew["K5_ff5"]["sd"],
        "amSeInflFfFive": (vw["K5_ff5"]["se_inflation_from_estimating_beta"], 3),
        "amMedSeRawVw": (vw["K0_raw_mean"]["median_alpha_se_bps"], 2),
        "amMedSeFfFiveVw": (vw["K5_ff5"]["median_alpha_se_bps"], 2),
        "amSdAlphaRawVw": (vw["K0_raw_mean"]["sd_alpha_bps"], 2),
        "amSdAlphaFfFiveVw": (vw["K5_ff5"]["sd_alpha_bps"], 2),
        "amCutoffFfFiveVw": (vw["K5_ff5"]["cutoff_5pct"], 2),
        "amCutoffRawVw": (vw["K0_raw_mean"]["cutoff_5pct"], 2),
        "amThreeFfFiveVwPct": (vw["K5_ff5"]["frac_abs_gt_300"] * 100, 1),
        "amThreeRawVwPct": (vw["K0_raw_mean"]["frac_abs_gt_300"] * 100, 2),
        # A2
        "amSlopeVw": (a2["slope"], 3),
        "amSlopeSeVw": (a2["slope_se_white"], 3),
        "amSlopeEw": (d["a2_slope_one"]["ew"]["slope"], 3),
        "amSlopeRsqVw": (a2["r2"], 3),
        "amVarAlphaVw": (dc["var_alpha"] * 1e4, 2),
        "amVarRbarVw": (dc["var_rbar"] * 1e4, 2),
        "amVarExposureVw": (dc["var_exposure"] * 1e4, 2),
        "amVarCovTermVw": (dc["minus_two_cov"] * 1e4, 2),
        "amCovTermVw": (-0.5 * dc["minus_two_cov"] * 1e4, 2),
        "amSdRatioVw": (dc["sd_ratio_alpha_over_rbar"], 3),
        "amIncreaseExposurePct": (dc["increase_share_from_exposure_var"] * 100, 0),
        "amIncreaseCovPct": (dc["increase_share_from_covariance"] * 100, 0),
        # A3
        "amDecileOneSd": (a3[0]["sd_alpha_t"], 3),
        "amDecileTenSd": (a3[-1]["sd_alpha_t"], 3),
        "amDecileOneThreePct": (a3[0]["frac_abs_t_gt_300"] * 100, 1),
        "amDecileTenThreePct": (a3[-1]["frac_abs_t_gt_300"] * 100, 1),
        "amDecileNumeratorRatio": (a3[-1]["sd_alpha_bps"] / a3[0]["sd_alpha_bps"], 2),
        "amDecileDenominatorRatio": (
            a3[-1]["median_alpha_se_bps"] / a3[0]["median_alpha_se_bps"], 2),
        # A4.  The nested-window fit is kept as a descriptive curve only; the
        # inferential numbers all come from the disjoint-block test, whose
        # points do not share data and therefore support a standard error.
        "amNoiseOnlyAlphaBps": (a4["persistence_fit"]["pure_noise_sd_alpha_bps"][-1], 1),
        "amObservedAlphaBpsLong": (a4["persistence_fit"]["observed_sd_alpha_bps"][-1], 1),
        "amBlockMonths": a4["disjoint"]["block_months"],
        "amNBlocks": a4["disjoint"]["n_blocks"],
        "amPersistentBps": (a4["disjoint"]["sd_persistent_bps"], 2),
        "amPersistentLoBps": (a4["disjoint"]["sd_persistent_ci95_bps"][0], 2),
        "amPersistentHiBps": (a4["disjoint"]["sd_persistent_ci95_bps"][1], 2),
        "amPersistentSharePct": (a4["disjoint"]["share_of_within_that_persists"] * 100, 1),
        "amWithinBlockSdBps": (a4["disjoint"]["within_block_var_bps2"] ** 0.5, 1),
        "amPersistentBootP": (a4["disjoint"]["bootstrap_p_cov_le_zero"], 3),
        # A5
        "amShareRmwCmaPct": (
            (a5["factor_terms"]["rmw"]["share_of_var_exposure"]
             + a5["factor_terms"]["cma"]["share_of_var_exposure"]) * 100, 0),
        "amShareSmbPct": (a5["factor_terms"]["smb"]["share_of_var_exposure"] * 100, 0),
        "amShareMktPct": (a5["factor_terms"]["mktrf"]["share_of_var_exposure"] * 100, 0),
        "amAlphabetRsq": (a5["alphabet_regressions"]["exposure"]["r2"], 3),
        # A6
        "amPlaceboIid": (a6["iid_normal"]["sd_alpha_t_mean"], 3),
        "amPlaceboBoot": (a6["residual_bootstrap"]["sd_alpha_t_mean"], 3),
        # A7
        "amWinnerName": a7["signalname"].replace("_", r"\_"),
        "amWinnerAlphaT": (a7["alpha_t"], 2),
        "amWinnerMeanT": (a7["mean_return_t"], 2),
        "amWinnerAlphaBps": (a7["alpha_bps_per_month"], 1),
        "amWinnerRbarBps": (a7["mean_return_bps_per_month"], 1),
        "amWinnerExposureBps": (a7["exposure_bps_per_month"], 1),
        "amWinnerAbsAlphaT": (abs(a7["alpha_t"]), 2),
        # A8
        "amPubN": a8["n_predictors"],
        "amPubAlphaOnly": a8["at_measured_cutoffs"]["n_alpha_only"],
        "amPubMeanOnly": a8["at_measured_cutoffs"]["n_mean_only"],
        "amPubAlphaOnlyThree": a8["at_300"]["n_alpha_only"],
        "amPubCorr": (a8["corr_mean_t_alpha_t"], 2),
        # The second population
        "amAltN": alt["n_higher_moment"],
        "amAltSdMeanTVw": (alt["vw"]["sd_mean_t"], 3),
        "amAltSdAlphaTVw": (alt["vw"]["sd_alpha_t"], 3),
        "amAltSdMeanTEw": (alt["ew"]["sd_mean_t"], 3),
        "amAltSdAlphaTEw": (alt["ew"]["sd_alpha_t"], 3),
        "amAltWideningVw": (alt["vw"]["widening_numerator_ratio"], 2),
        "amTickerWideningVw": (
            vw["K5_ff5"]["sd_alpha_bps"] / vw["K0_raw_mean"]["sd_alpha_bps"], 2),
    }
    vwa = c["c1_survival"]["vw"]["ff5_alpha_t"]
    vwm = c["c1_survival"]["vw"]["mean_return_t"]
    split = c["c2_variance_split"]["vw"]
    cov = c["c3_split_sample_covariance"]["vw"]
    fam = {r["family_size"]: r for r in vwa["by_family_size"]}
    values.update({
        # C1
        "amBonfCrit": (vwa["whole_population"]["bonferroni_critical_value"], 2),
        "amBonfFalseAlpha": vwa["whole_population"]["bonferroni"],
        "amBonfFalseMean": vwm["whole_population"]["bonferroni"],
        "amBhFalseAlpha": vwa["whole_population"]["benjamini_hochberg"],
        "amBhFalseMean": vwm["whole_population"]["benjamini_hochberg"],
        "amUncorrFalseAlpha": vwa["whole_population"]["uncorrected"],
        "amMeasuredBonfCrit": (vwa["measured_null"]["bonferroni_critical_value"], 2),
        "amMeasuredBonfFalse": vwa["measured_null"]["bonferroni"],
        "amFamFiftyBonf": (fam[50]["expected_bonferroni"], 2),
        "amFamFiftyAnyPct": (fam[50]["any_false_rejection_bonferroni"] * 100, 0),
        "amFamTenAnyPct": (fam[10]["any_false_rejection_bonferroni"] * 100, 0),
        # C2
        "amCOne": (split["c1_sample_specific"], 2),
        "amCTwoT": (split["c2T_persistent"], 3),
        "amCOneSharePct": (split["share_of_excess_from_c1"] * 100, 0),
        "amCTwoSharePct": (split["share_of_excess_from_c2T"] * 100, 0),
        "amTEqualYears": (split["T_where_c2T_equals_c1"] / 12.0, 0),
        # C3
        "amCovInSampleBps": (cov["full_sample_in_sample"]["cov_bps2"], 1),
        "amCovOosBps": (cov["second_half_beta_from_first_half"]["cov_bps2"], 1),
        "amCovOosLoBps": (cov["oos_cov_ci95_bps2"][0], 1),
        "amCovOosHiBps": (cov["oos_cov_ci95_bps2"][1], 1),
        # Numbers that were typed by hand in the prose until now.
        "amObservedAlphaBpsShort": (
            d["a4_sqrt_t"]["vw"]["persistence_fit"]["observed_sd_alpha_bps"][0], 1),
        "amCorrSquaredVw": (d["a2_slope_one"]["vw"]["corr_squared"], 2),
        "amWinnerExposureSharePct": (
            100 * abs(a7["exposure_bps_per_month"] / a7["alpha_bps_per_month"]), 0),
        "amTimesNominal": (
            d["a1_across_models"]["vw"][4]["frac_abs_gt_300"] / 0.0027, 0),
    })
    tables.write_macros("macros_alpha", values, digits=3)


def main() -> int:
    d = json.loads((RESULTS / "alpha_mechanism.json").read_text())
    c = json.loads((RESULTS / "corrections_on_null.json").read_text())
    table_across_models(d)
    table_corrections_on_null(c)
    table_exposure_deciles(d)
    table_alt_population(d)
    figure_across_models(d)
    figure_exposure_dose_response(d)
    write_macros(d, c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
