"""Harvey-Liu-Zhu haircuts.

Harvey, Liu and Zhu (2016), Review of Financial Studies 29, 5-68, apply
Bonferroni, Holm and Benjamini-Hochberg-Yekutieli to the t-statistics of
published return predictors, and Harvey and Liu (2015), Journal of Portfolio
Management 42(1), 13-28, turn the adjusted p-value back into a "haircut" on
the reported Sharpe ratio.

The translation is:

1. observed two-sided p-value ``p = 2 (1 - Phi(|t|))``;
2. adjusted p-value ``p_adj`` from one of the three procedures;
3. adjusted t-statistic ``t_adj = Phi^{-1}(1 - p_adj/2)``;
4. haircut ``h = 1 - t_adj / t``.

Step 4 works because, for a fixed sample length, the Sharpe ratio is
proportional to the t-statistic, so the proportional cut in t is the
proportional cut in the Sharpe ratio.

The one input that is not observed is the total number of tests the field
ran.  Harvey, Liu and Zhu estimate it structurally from the distribution of
published t-statistics, because unpublished tests are invisible.  We take it
as an argument and report results across a grid, which is the same admission
made explicit.  Their headline recommendation of a t-statistic near 3.0
corresponds to a particular point on that grid.

Assumption when only some p-values are observed
-----------------------------------------------
Holm and BHY need all ``M`` p-values, and we only see the published ones.
Following Harvey, Liu and Zhu, we assume the observed set are the ``K``
smallest of the ``M`` total, that is, that unpublished tests were
unsuccessful.  This is optimistic about the field in a specific direction:
it puts the unobserved tests where they do the least damage.  Stated in the
paper as a limitation.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .base import CorrectionResult, check_alpha, check_pvalues


def t_to_pvalue(t: np.ndarray) -> np.ndarray:
    return 2.0 * stats.norm.sf(np.abs(np.asarray(t, dtype=float)))


def pvalue_to_t(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    return stats.norm.isf(p / 2.0)


def _adjusted_bonferroni(p_sorted: np.ndarray, m_total: int) -> np.ndarray:
    return np.minimum(1.0, m_total * p_sorted)


def _adjusted_holm(p_sorted: np.ndarray, m_total: int) -> np.ndarray:
    """Holm adjusted p-values when the observed set is the smallest ``K`` of ``M``."""
    k = p_sorted.size
    ranks = np.arange(1, k + 1)
    raw = (m_total - ranks + 1) * p_sorted
    return np.minimum(1.0, np.maximum.accumulate(raw))


def _adjusted_bhy(p_sorted: np.ndarray, m_total: int) -> np.ndarray:
    """Benjamini-Hochberg-Yekutieli adjusted p-values, same convention."""
    k = p_sorted.size
    c_m = float(np.sum(1.0 / np.arange(1, m_total + 1)))
    ranks = np.arange(1, k + 1)
    raw = m_total * c_m * p_sorted / ranks
    # Step-up: enforce monotonicity from the largest observed p downwards.
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    return np.minimum(1.0, adj)


ADJUSTERS = {
    "bonferroni": _adjusted_bonferroni,
    "holm": _adjusted_holm,
    "bhy": _adjusted_bhy,
}


def haircuts(
    t_stats: Sequence[float],
    n_total_tests: int,
    names: "Sequence[str] | None" = None,
    method: str = "bonferroni",
    marginal_sd: float = 1.0,
) -> pd.DataFrame:
    """Adjusted t-statistics and Sharpe ratio haircuts for a set of factors.

    Parameters
    ----------
    t_stats:
        Reported t-statistics.  Signs are ignored; the procedures are
        two-sided.
    n_total_tests:
        ``M``, the assumed total number of tests including unpublished ones.
        Must be at least the number of statistics supplied.
    marginal_sd:
        Standard deviation of the null t-statistic distribution.  The
        default of 1.0 reproduces the published calculation.  Passing a
        calibrated value is the whole point of this project: the observed
        p-value becomes ``2(1 - Phi(|t|/sd))``, which shifts every
        downstream number.

    Returns
    -------
    Frame with one row per factor: ``t_obs``, ``p_obs``, ``p_adj``,
    ``t_adj``, ``haircut``, and ``survives`` at the conventional 5 percent
    level.
    """
    t = np.abs(np.asarray(list(t_stats), dtype=float))
    if names is None:
        names = [f"factor_{i}" for i in range(t.size)]
    if n_total_tests < t.size:
        raise ValueError("n_total_tests must be at least the number of statistics")
    if method not in ADJUSTERS:
        raise ValueError(f"method must be one of {sorted(ADJUSTERS)}")
    if marginal_sd <= 0:
        raise ValueError("marginal_sd must be positive")

    p_obs = 2.0 * stats.norm.sf(t / marginal_sd)
    order = np.argsort(p_obs, kind="mergesort")
    p_adj_sorted = ADJUSTERS[method](p_obs[order], int(n_total_tests))
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted

    t_adj = pvalue_to_t(p_adj) * marginal_sd
    with np.errstate(invalid="ignore", divide="ignore"):
        haircut = 1.0 - t_adj / t
    return pd.DataFrame(
        {
            "name": list(names),
            "t_obs": t,
            "p_obs": p_obs,
            "p_adj": p_adj,
            "t_adj": t_adj,
            "haircut": haircut,
            "survives_05": p_adj <= 0.05,
        }
    ).set_index("name")


def harvey_liu_zhu(
    p: np.ndarray,
    n_total_tests: int,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> CorrectionResult:
    """The three HLZ adjustments in the common :class:`CorrectionResult` form."""
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    if method not in ADJUSTERS:
        raise ValueError(f"method must be one of {sorted(ADJUSTERS)}")
    order = np.argsort(p, kind="mergesort")
    adj_sorted = ADJUSTERS[method](p[order], int(n_total_tests))
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return CorrectionResult(
        method=f"Harvey-Liu-Zhu ({method})",
        n_tests=p.size,
        alpha=alpha,
        reject=adj <= alpha,
        adjusted_p=adj,
        critical_value=None,
        error_rate="FDR" if method == "bhy" else "FWER",
        extra={"n_total_tests": int(n_total_tests)},
    )


def threshold_grid(
    n_total_grid: Sequence[int],
    alpha: float = 0.05,
    marginal_sd: float = 1.0,
    methods: Sequence[str] = ("bonferroni", "holm", "bhy"),
) -> pd.DataFrame:
    """t-statistic threshold for a single new factor, as a function of ``M``.

    This is the sensitivity table the plan insists on: no single ``M`` is
    ever asserted.  For a single new candidate, Holm and Bonferroni coincide
    (the first step of Holm is the Bonferroni step), so the Holm column is
    reported for completeness rather than as new information.
    """
    rows = []
    for m in n_total_grid:
        row: Dict[str, float] = {"n_total_tests": int(m)}
        for meth in methods:
            if meth == "bhy":
                c_m = float(np.sum(1.0 / np.arange(1, int(m) + 1)))
                p_star = alpha / (m * c_m)
            else:
                p_star = alpha / m
            row[f"t_{meth}"] = float(stats.norm.isf(p_star / 2.0) * marginal_sd)
        rows.append(row)
    return pd.DataFrame(rows)
