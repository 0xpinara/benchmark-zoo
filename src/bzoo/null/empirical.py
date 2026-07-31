"""Describing an empirical null distribution and comparing it with the
theoretical one.

The theoretical null for a t-statistic against a zero mean is standard
normal.  Whether the *observed* distribution of t-statistics in a population
of strategies that truly have no edge matches it is an empirical question,
and it is the question this module answers.  Efron (2004), JASA 99, 96-104,
makes the same point in the microarray setting and calls the measured
version the empirical null; the finance and machine learning literatures
generally assume the theoretical one.

Three separate things can go wrong, and they are worth keeping apart because
they have different consequences:

1. **Location and scale.**  If the null t-statistics have standard deviation
   1.4 rather than 1.0, every nominal threshold understates the true one by
   40 percent.  Reported as ``sd`` below.
2. **Shape.**  Even with the right variance, heavier tails put more mass past
   any given cutoff.  Handled in :mod:`bzoo.null.tails`.
3. **Dependence.**  This does not affect the marginal distribution at all,
   but it changes the distribution of the *maximum*, which is what every
   multiplicity correction actually depends on.  Handled in
   :mod:`bzoo.null.dependence`.

A population can therefore have a perfectly standard normal marginal null
and still make Bonferroni badly wrong, in either direction.  Reporting only
the variance inflation, as is common, misses this.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NullSummary:
    """Marginal summary of an empirical null distribution of statistics."""

    n: int
    mean: float
    sd: float
    median: float
    mad_sd: float
    skew: float
    kurtosis: float
    q01: float
    q05: float
    q95: float
    q99: float
    max: float
    mean_ci_low: float
    mean_ci_high: float
    ks_stat: float
    ks_pvalue: float
    frac_abs_gt_196: float
    frac_abs_gt_258: float
    frac_abs_gt_300: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def summarise_null(t: np.ndarray, n_boot_ci: int = 2000, seed: int = 0) -> NullSummary:
    """Marginal summary of a vector of null statistics.

    ``mad_sd`` is the median absolute deviation rescaled by 1.4826 so that it
    estimates the standard deviation of a normal.  It is reported next to
    ``sd`` because the two diverge exactly when the tails are the problem,
    and a reader who sees only one cannot tell which.

    The confidence interval on the mean is a bootstrap over strategies.  It
    is *not* a valid test of whether the population mean is zero, because the
    strategies are heavily dependent; the correct version resamples months
    instead, and lives in ``scripts/04_known_null_validation.py``.
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    n = t.size
    if n < 10:
        raise ValueError("need at least ten statistics")

    rng = np.random.default_rng(seed)
    boot_means = t[rng.integers(0, n, size=(n_boot_ci, n))].mean(axis=1)
    ks = stats.kstest(t, "norm")

    return NullSummary(
        n=n,
        mean=float(t.mean()),
        sd=float(t.std(ddof=1)),
        median=float(np.median(t)),
        mad_sd=float(stats.median_abs_deviation(t, scale="normal")),
        skew=float(stats.skew(t)),
        kurtosis=float(stats.kurtosis(t, fisher=False)),
        q01=float(np.quantile(t, 0.01)),
        q05=float(np.quantile(t, 0.05)),
        q95=float(np.quantile(t, 0.95)),
        q99=float(np.quantile(t, 0.99)),
        max=float(t.max()),
        mean_ci_low=float(np.quantile(boot_means, 0.025)),
        mean_ci_high=float(np.quantile(boot_means, 0.975)),
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        frac_abs_gt_196=float(np.mean(np.abs(t) > 1.959964)),
        frac_abs_gt_258=float(np.mean(np.abs(t) > 2.575829)),
        frac_abs_gt_300=float(np.mean(np.abs(t) > 3.0)),
    )


def exceedance_table(
    t: np.ndarray,
    cutoffs: "tuple[float, ...]" = (2.0, 2.5, 3.0, 3.5, 4.0),
    two_sided: bool = True,
) -> pd.DataFrame:
    """Observed versus Gaussian exceedance probabilities at fixed cutoffs.

    The ratio column is the quantity a reader should look at: how many times
    more often the empirical null crosses a cutoff than the theoretical null
    says it should.  A ratio of 1 means the theoretical null is right at that
    cutoff, which can happen at 2.0 and fail at 3.0.
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    rows = []
    for c in cutoffs:
        if two_sided:
            emp = float(np.mean(np.abs(t) > c))
            theo = float(2.0 * stats.norm.sf(c))
        else:
            emp = float(np.mean(t > c))
            theo = float(stats.norm.sf(c))
        # Wilson interval, which stays inside [0,1] for the small
        # probabilities we are dealing with.
        lo, hi = _wilson(emp, t.size)
        rows.append(
            {
                "cutoff": c,
                "empirical": emp,
                "empirical_lo": lo,
                "empirical_hi": hi,
                "gaussian": theo,
                "ratio": emp / theo if theo > 0 else np.nan,
                "n_exceed": int(round(emp * t.size)),
            }
        )
    return pd.DataFrame(rows)


def _wilson(p: float, n: int, z: float = 1.959964) -> "tuple[float, float]":
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def calibrated_threshold(
    t_null: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Pointwise threshold read off the empirical null.

    The value a single statistic must exceed for a per-test false positive
    rate of ``alpha`` when the null is the measured one rather than the
    assumed one.  This is *not* multiplicity-corrected; it is the input the
    corrections then adjust.
    """
    t_null = np.asarray(t_null, dtype=float)
    t_null = t_null[np.isfinite(t_null)]
    if two_sided:
        return float(np.quantile(np.abs(t_null), 1.0 - alpha))
    return float(np.quantile(t_null, 1.0 - alpha))


def empirical_pvalues(
    t_obs: np.ndarray, t_null: np.ndarray, two_sided: bool = True
) -> np.ndarray:
    """p-values of observed statistics against an empirical null sample.

    ``(1 + #{null >= obs}) / (1 + n_null)``, the usual plus-one convention,
    which keeps p-values strictly positive so that step-down procedures and
    log transforms behave.  The floor is ``1/(n_null+1)``: with 19,380 null
    strategies no p-value below about 5e-5 can be resolved, and the
    calibrated tail model of :mod:`bzoo.null.tails` is what extends past it.
    """
    t_obs = np.asarray(t_obs, dtype=float)
    ref = np.sort(np.abs(t_null) if two_sided else np.asarray(t_null, dtype=float))
    ref = ref[np.isfinite(ref)]
    x = np.abs(t_obs) if two_sided else t_obs
    n_ge = ref.size - np.searchsorted(ref, x, side="left")
    return (1.0 + n_ge) / (1.0 + ref.size)


def variance_inflation(
    t_null: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """Standard deviation of the null statistics, with a bootstrap interval.

    Interpretation: a value ``s`` means the correct pointwise two-sided
    5 percent threshold is ``1.96 * s`` rather than ``1.96``, *if* the null
    is otherwise normal.  Whether it is, is what
    :func:`summarise_null` and :mod:`bzoo.null.tails` are for.

    The bootstrap here resamples strategies, so the interval understates
    uncertainty in the presence of cross-strategy dependence.  It is
    reported as a lower bound on the width and labelled as such.
    """
    t_null = np.asarray(t_null, dtype=float)
    t_null = t_null[np.isfinite(t_null)]
    n = t_null.size
    rng = np.random.default_rng(seed)
    sds = np.empty(n_boot)
    for b in range(n_boot):
        sds[b] = t_null[rng.integers(0, n, size=n)].std(ddof=1)
    return {
        "sd": float(t_null.std(ddof=1)),
        "sd_lo": float(np.quantile(sds, 0.025)),
        "sd_hi": float(np.quantile(sds, 0.975)),
        "mad_sd": float(stats.median_abs_deviation(t_null, scale="normal")),
        "n": int(n),
    }


def conditional_summary(
    t_by_group: Dict[str, np.ndarray],
    seed: int = 0,
) -> pd.DataFrame:
    """Marginal null summaries for several subsamples, one row each.

    Used for the decade splits and the volatility-regime splits.  If the
    standard deviation moves systematically across regimes, then a single
    unconditional threshold is the wrong object and a conditional one is
    needed; that is a finding, not a robustness check.
    """
    rows = []
    for name, t in t_by_group.items():
        s = summarise_null(t, seed=seed)
        row = s.to_dict()
        row["group"] = name
        rows.append(row)
    cols = ["group"] + [c for c in rows[0] if c != "group"]
    return pd.DataFrame(rows)[cols]
