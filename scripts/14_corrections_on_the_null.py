"""Does a multiplicity correction actually fix the exposure channel?

The paper asserts that no multiplicity correction is a remedy for alphas that
are real rather than spurious.  That assertion was never tested, which is
indefensible when the test is one line: we have twelve corrections and 19,380
strategies that are null by construction.  This script runs it, and splits the
variance of the alpha t-statistic into the part a correction can absorb and the
part it cannot.

Three questions, each writing one block of
``data/results/corrections_on_null.json``.

C1  Survival under correction.  Apply the classical corrections to the ticker
    population's alpha t-statistics and count how many of the 19,380 empty
    strategies survive each, against the same corrections applied to the
    mean-return t-statistics.  Reported at the full family size and at the
    smaller family sizes a working researcher actually screens: a correction
    over 19,380 tests is not what someone comparing fifty candidate signals
    does.  This is the number that says when the channel bites.

C2  The variance split.  Write the alpha of a no-content strategy as

        alpha_hat = -beta' mu_f  -  beta'(fbar - mu_f)  +  ebar + O(1/T)
                    \___________/   \__________________/   \____/
                     persistent      sample-specific        noise

    In t-units the first term contributes a variance that grows like T, because
    the quantity is fixed while SE shrinks like 1/sqrt(T).  The second
    contributes a variance that does *not* grow, because fbar converges to mu_f
    at exactly the rate SE shrinks.  So

        Var(t_alpha) = 1 + c1 + c2 * T

    and the distinction is the whole argument.  A constant c1 is a pure scale
    inflation, which is exactly what a null calibrated on the measured
    population absorbs: multiply the threshold and you are done.  A growing
    c2*T is not, because no fixed threshold contains it.  If c2 is small
    relative to c1 then "no correction is a remedy" is too strong and has to
    become "no correction calibrated on the *nominal* null is a remedy".

    ``Var(-beta' mu_f)`` is the between-block covariance from the disjoint-block
    estimate in script 12, which is unbiased for it because estimation noise is
    independent across disjoint blocks.  Everything else follows.

C3  Is Cov(rbar, beta'fbar) real or mechanical?  The decomposition in the paper
    turns on this covariance being negative, but in a population that is null
    by construction E[rbar | beta] = 0 and the covariance should vanish in
    expectation.  A reliably negative value means either the population is not
    null in the way we claim, or beta_hat is absorbing something from rbar
    because both are estimated on the same months.  We rule the second out by
    estimating beta on one half of the sample and forming the exposure term on
    the other, so the two are estimated on disjoint data.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.corrections import fdr, fwer
from bzoo.finance import loaders, metrics
from bzoo.paths import RESULTS, ensure_dirs

MIN_MONTHS = 60
SEED = 20260801
FF5 = ["mktrf", "smb", "hml", "rmw", "cma"]
WEIGHTINGS = ("vw", "ew")

# Family sizes a reader might plausibly be in.  The point of the table is that
# the answer depends on this and the paper has to say which one it means.
FAMILY_SIZES = (1, 10, 50, 200, 1000, 19380)


def _import_mechanism():
    """Reuse script 12's fitting helpers rather than reimplementing them."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "alpha_mechanism", Path(__file__).resolve().parent / "12_alpha_mechanism.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# C1


def c1_survival(am, panels: dict, factors: pd.DataFrame) -> dict:
    """How many empty strategies survive each correction?"""
    out = {}
    for w in WEIGHTINGS:
        res_a = am.fit_model(panels[w], factors, FF5)
        block = {}
        for stat, t in (("ff5_alpha_t", res_a["alpha_t"].to_numpy()),
                        ("mean_return_t", res_a["mean_t"].to_numpy())):
            t = t[np.isfinite(t)]
            n = t.size
            # Two-sided p-values against the nominal standard normal, which is
            # what a researcher applying a correction would use.
            p = 2.0 * stats.norm.sf(np.abs(t))
            rows = []
            for m in FAMILY_SIZES:
                # A family of size m drawn from the population: rather than
                # subsample once, use the exact expected count, which for a
                # per-test rule is n * Pr(reject) and for a family-wise rule is
                # the count that would survive if the m tests were these m.
                # We report both the whole-population application (m = n) and,
                # for smaller m, the expected number of false rejections in a
                # family of that size, which is what a researcher experiences.
                if m == 1:
                    surv = float(np.mean(p < 0.05) * m)
                    bonf = surv
                    holm_c = surv
                    bh = surv
                else:
                    thresh_bonf = 0.05 / m
                    bonf = float(np.mean(p < thresh_bonf) * m)
                    # Holm and BH on a random family of size m, averaged over
                    # many draws, is the honest way to get the expected count.
                    rng = np.random.default_rng(SEED)
                    n_draw = 200 if m < n else 1
                    bh_counts, holm_counts = [], []
                    for _ in range(n_draw):
                        idx = (rng.choice(n, size=m, replace=False)
                               if m < n else np.arange(n))
                        pm = p[idx]
                        holm_counts.append(int(fwer.holm(pm, 0.05).n_reject))
                        bh_counts.append(
                            int(fdr.benjamini_hochberg(pm, 0.05).n_reject))
                    holm_c = float(np.mean(holm_counts))
                    bh = float(np.mean(bh_counts))
                    surv = float(np.mean(p < 0.05) * m)
                rows.append({
                    "family_size": m,
                    "expected_uncorrected": surv,
                    "expected_bonferroni": bonf,
                    "expected_holm": holm_c,
                    "expected_bh": bh,
                    "any_false_rejection_bonferroni": float(
                        1.0 - (1.0 - np.mean(p < 0.05 / m)) ** m),
                })
            block[stat] = {
                "n": int(n),
                "sd_t": float(t.std(ddof=1)),
                "frac_p_below_05": float(np.mean(p < 0.05)),
                "by_family_size": rows,
                # The whole-population application, which is the direct test of
                # the paper's claim.
                "whole_population": {
                    "n_tests": int(n),
                    "uncorrected": int((p < 0.05).sum()),
                    "bonferroni": int(fwer.bonferroni(p, 0.05).n_reject),
                    "sidak": int(fwer.sidak(p, 0.05).n_reject),
                    "holm": int(fwer.holm(p, 0.05).n_reject),
                    "benjamini_hochberg": int(
                        fdr.benjamini_hochberg(p, 0.05).n_reject),
                    "benjamini_yekutieli": int(
                        fdr.benjamini_yekutieli(p, 0.05).n_reject),
                    "max_abs_t": float(np.max(np.abs(t))),
                    "bonferroni_critical_value": float(
                        stats.norm.isf(0.05 / (2 * n))),
                },
                # The same Bonferroni, with the *measured* null substituted for
                # the standard normal.  This is the comparison that says which
                # part of the correction is at fault: the multiplicity
                # arithmetic, or the null it is applied to.
                "measured_null": {
                    "sd": float(t.std(ddof=1)),
                    "bonferroni_critical_value": float(
                        t.std(ddof=1) * stats.norm.isf(0.05 / (2 * n))),
                    "bonferroni": int(
                        (np.abs(t) > t.std(ddof=1) * stats.norm.isf(0.05 / (2 * n))).sum()),
                    "uncorrected_cutoff": float(np.quantile(np.abs(t), 0.95)),
                    "uncorrected": int(
                        (np.abs(t) > np.quantile(np.abs(t), 0.95)).sum()),
                },
            }
        out[w] = block
    return out


# ----------------------------------------------------------------------
# C2


def c2_variance_split(am, panels: dict, factors: pd.DataFrame, mech: dict) -> dict:
    """Split Var(t_alpha) into 1 + c1 + c2*T."""
    out = {}
    for w in WEIGHTINGS:
        res = am.fit_model(panels[w], factors, FF5)
        months = am.regression_sample(panels[w], factors, FF5)
        t_full = int(len(months))

        se = res["alpha_se"].to_numpy(dtype=np.float64)
        alpha = res["alpha"].to_numpy(dtype=np.float64)
        ok = np.isfinite(se) & np.isfinite(alpha)
        se, alpha = se[ok], alpha[ok]

        mean_se2 = float(np.mean(se ** 2))            # in (% / month)^2
        var_alpha = float(np.var(alpha, ddof=1))

        # Var(-beta' mu_f): the component common to disjoint decades.
        dj = mech["a4_sqrt_t"][w]["disjoint"]
        var_persistent = float(dj["between_block_cov_bps2"]) / 1e4

        # What is left, after removing the persistent part and the pure
        # estimation noise, is the sample-specific exposure.
        var_sample_specific = var_alpha - var_persistent - mean_se2

        c2T = var_persistent / mean_se2
        c1 = var_sample_specific / mean_se2
        out[w] = {
            "T": t_full,
            "n": int(alpha.size),
            "observed_var_t_alpha": float(np.var(res["alpha_t"].dropna(), ddof=1)),
            "implied_var_t_alpha": 1.0 + c1 + c2T,
            "mean_se_squared_bps2": mean_se2 * 1e4,
            "var_alpha_bps2": var_alpha * 1e4,
            "var_persistent_bps2": var_persistent * 1e4,
            "var_sample_specific_bps2": var_sample_specific * 1e4,
            "one": 1.0,
            "c1_sample_specific": c1,
            "c2T_persistent": c2T,
            "c2_per_month": c2T / t_full,
            "share_of_excess_from_c1": c1 / (c1 + c2T),
            "share_of_excess_from_c2T": c2T / (c1 + c2T),
            # How long a sample would have to be before the growing component
            # matters as much as the fixed one.
            "T_where_c2T_equals_c1": float(c1 / (c2T / t_full)),
            # And before the growing component alone pushes the median empty
            # strategy past a Bonferroni bar at the full family size.
            "T_where_c2T_alone_gives_sd_equal_bonferroni": float(
                (stats.norm.isf(0.05 / (2 * 19380)) ** 2 - 1.0) / (c2T / t_full)),
        }
    return out


# ----------------------------------------------------------------------
# C3


def c3_split_sample_covariance(am, panels: dict, factors: pd.DataFrame) -> dict:
    """Is Cov(rbar, beta'fbar) real, or an artefact of a shared sample?

    In-sample, beta_hat and rbar are estimated on the same months, so any
    covariance between rbar and beta_hat'fbar could be mechanical. Estimating
    beta on the first half and forming the exposure on the second half makes
    the two independent under the null, so a covariance that survives is real.
    """
    out = {}
    for w in WEIGHTINGS:
        months = am.regression_sample(panels[w], factors, FF5)
        half = len(months) // 2
        first, second = months[:half], months[half:]

        res_in = am.fit_model(panels[w], factors, FF5)
        res_1 = am.fit_model(panels[w].loc[first], factors, FF5, min_months=60)
        res_2 = am.fit_model(panels[w].loc[second], factors, FF5, min_months=60)
        common = res_1.index.intersection(res_2.index)

        beta_1 = res_1.loc[common, [f"beta_{c}" for c in FF5]].to_numpy()
        fbar_2 = factors.loc[second, FF5].mean().to_numpy()
        exposure_oos = beta_1 @ fbar_2               # beta from half 1, fbar from half 2
        rbar_2 = res_2.loc[common, "rbar"].to_numpy()

        beta_2 = res_2.loc[common, [f"beta_{c}" for c in FF5]].to_numpy()
        exposure_is = beta_2 @ fbar_2                # both from half 2

        def summary(x, y):
            c = float(np.cov(x, y, ddof=1)[0, 1])
            return {
                "cov": c,
                "cov_bps2": c * 1e4,
                "corr": float(np.corrcoef(x, y)[0, 1]),
                "sd_x_bps": float(np.std(x, ddof=1) * 100),
                "sd_y_bps": float(np.std(y, ddof=1) * 100),
            }

        rng = np.random.default_rng(SEED)
        boot = np.array([
            np.cov(*(lambda i: (rbar_2[i], exposure_oos[i]))(
                rng.integers(0, common.size, common.size)), ddof=1)[0, 1]
            for _ in range(2000)
        ])

        out[w] = {
            "n": int(common.size),
            "n_months_half": int(half),
            "full_sample_in_sample": summary(
                res_in["rbar"].to_numpy(), res_in["exposure"].to_numpy()),
            "second_half_in_sample": summary(rbar_2, exposure_is),
            "second_half_beta_from_first_half": summary(rbar_2, exposure_oos),
            "oos_cov_ci95_bps2": [float(np.quantile(boot, 0.025) * 1e4),
                                  float(np.quantile(boot, 0.975) * 1e4)],
            "oos_bootstrap_mass_below_zero": float(np.mean(boot < 0.0)),
        }
    return out


# ----------------------------------------------------------------------


def main() -> int:
    ensure_dirs()
    am = _import_mechanism()
    mech = json.loads((RESULTS / "alpha_mechanism.json").read_text())
    factors = loaders.download_factors()
    panels = {w: loaders.mined_return_panel("ticker", w) for w in WEIGHTINGS}

    results: dict = {"config": {"min_months": MIN_MONTHS, "seed": SEED}}
    steps = [
        ("c1_survival", lambda: c1_survival(am, panels, factors)),
        ("c2_variance_split", lambda: c2_variance_split(am, panels, factors, mech)),
        ("c3_split_sample_covariance",
         lambda: c3_split_sample_covariance(am, panels, factors)),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===", flush=True)
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a failed check is reported
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {results[name]['error']}", flush=True)
            continue
        print(json.dumps(results[name], indent=2)[:2000], flush=True)

    out = RESULTS / "corrections_on_null.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
