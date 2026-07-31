"""Family-wise error rate procedures.

Contents
--------
``bonferroni``       Bonferroni (1936), the single-step p-value correction.
``sidak``            Sidak (1967), the exact version under independence.
``holm``             Holm (1979), Scandinavian Journal of Statistics 6, 65-70.
``romano_wolf``      Romano and Wolf (2005), Econometrica 73, 1237-1282.
``westfall_young``   Westfall and Young (1993), Resampling-Based Multiple
                     Testing, Wiley.  Both the max-T and the min-P variants.

The two resampling procedures need a ``(B, K)`` matrix of centred
statistics: replicate ``b``, hypothesis ``k``.  The columns must come from
the *same* resample draw, otherwise the dependence between hypotheses is
destroyed.  See :mod:`bzoo.resample.instance` for how we build it.
"""

from __future__ import annotations

import numpy as np

from .base import (
    CorrectionResult,
    check_alpha,
    check_pvalues,
    check_resample_matrix,
)


def bonferroni(p: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni correction.

    Reject hypothesis ``k`` if ``p_k <= alpha / K``.  Equivalently, the
    adjusted p-value is ``min(1, K * p_k)``.
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    adj = np.minimum(1.0, k * p)
    return CorrectionResult(
        method="Bonferroni",
        n_tests=k,
        alpha=alpha,
        reject=p <= alpha / k,
        adjusted_p=adj,
        critical_value=alpha / k,
        error_rate="FWER",
    )


def sidak(p: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Sidak (1967) correction: exact FWER control under independence.

    Reject if ``p_k <= 1 - (1 - alpha) ** (1 / K)``.  Slightly less
    conservative than Bonferroni; we report it because the deflation
    p-value of :mod:`bzoo.corrections.deflation` is its statistic-scale
    twin, and it is useful to see the two agree.
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    crit = 1.0 - (1.0 - alpha) ** (1.0 / k)
    adj = np.minimum(1.0, 1.0 - (1.0 - p) ** k)
    return CorrectionResult(
        method="Sidak",
        n_tests=k,
        alpha=alpha,
        reject=p <= crit,
        adjusted_p=adj,
        critical_value=crit,
        error_rate="FWER",
    )


def holm(p: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Holm (1979) step-down procedure.

    Sort p-values ascending.  Walk down the list comparing ``p_(i)`` with
    ``alpha / (K - i + 1)``; stop at the first failure and retain every
    remaining hypothesis.  Uniformly more powerful than Bonferroni and
    valid under arbitrary dependence.
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]

    # Step-down thresholds alpha / (K - i) for i = 0, ..., K-1.
    thresh = alpha / (k - np.arange(k))
    passed = p_sorted <= thresh
    # Reject the first run of passes only.
    if not passed[0]:
        n_rej = 0
    else:
        fail = np.flatnonzero(~passed)
        n_rej = int(fail[0]) if fail.size else k

    reject_sorted = np.zeros(k, dtype=bool)
    reject_sorted[:n_rej] = True

    # Adjusted p-values: running maximum of (K - i) * p_(i), capped at 1.
    # The running maximum enforces monotonicity, which the raw products do
    # not have.
    adj_sorted = np.minimum(np.maximum.accumulate((k - np.arange(k)) * p_sorted), 1.0)

    reject = np.empty(k, dtype=bool)
    adj = np.empty(k, dtype=float)
    reject[order] = reject_sorted
    adj[order] = adj_sorted
    return CorrectionResult(
        method="Holm",
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=adj,
        critical_value=None,
        error_rate="FWER",
    )


def _stepdown_maxstat(
    obs: np.ndarray,
    boot: np.ndarray,
    alpha: float,
    method_name: str,
) -> CorrectionResult:
    """Shared engine for Romano-Wolf and Westfall-Young max-T.

    ``boot`` holds centred statistics, so its columns already have mean
    (approximately) zero under the null.  The step-down logic is:

    1. take the largest observed statistic;
    2. build the null distribution of the maximum over the hypotheses
       still under consideration;
    3. reject if the observed statistic exceeds its ``1 - alpha`` quantile;
    4. drop the rejected hypothesis and repeat, otherwise stop.

    Step 2 is what makes the procedure dependence-aware: the maximum is
    taken within each replicate, so correlated hypotheses do not each get
    charged the full multiplicity penalty.
    """
    boot, obs = check_resample_matrix(boot, obs)
    b, k = boot.shape
    alpha = check_alpha(alpha)

    order = np.argsort(-obs, kind="mergesort")  # descending
    remaining = list(order)
    reject_sorted = np.zeros(k, dtype=bool)
    adj_sorted = np.ones(k, dtype=float)
    first_crit = None
    running_p = 0.0

    step = 0
    while remaining:
        idx = np.asarray(remaining, dtype=int)
        max_null = boot[:, idx].max(axis=1)
        j = remaining[0]  # largest observed statistic among those remaining
        # p-value = fraction of replicates whose maximum reaches the observation.
        # The +1 in numerator and denominator keeps the p-value strictly
        # positive, which matters for the step-down monotonicity below.
        p_step = (1.0 + np.sum(max_null >= obs[j])) / (b + 1.0)
        crit = np.quantile(max_null, 1.0 - alpha)
        if step == 0:
            first_crit = float(crit)
        # Enforce monotone adjusted p-values down the step-down path.
        running_p = max(running_p, p_step)
        adj_sorted[step] = running_p
        if obs[j] > crit:
            reject_sorted[step] = True
            remaining.pop(0)
            step += 1
        else:
            # Everything left keeps the current adjusted p-value.
            adj_sorted[step:] = running_p
            break

    reject = np.zeros(k, dtype=bool)
    adj = np.ones(k, dtype=float)
    reject[order] = reject_sorted
    adj[order] = adj_sorted
    return CorrectionResult(
        method=method_name,
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=adj,
        critical_value=first_crit,
        error_rate="FWER",
        extra={"n_replicates": b},
    )


def romano_wolf(obs: np.ndarray, boot: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Romano-Wolf (2005) step-down FWER control.

    Parameters
    ----------
    obs:
        Observed statistics, one per hypothesis, large values against the
        null.
    boot:
        ``(B, K)`` matrix of *centred* bootstrap statistics.  Centring is
        the caller's job because it depends on the statistic: for a mean we
        subtract the sample mean, for a studentised statistic we subtract
        the observed value.

    Notes
    -----
    Romano and Wolf (2005), Algorithm 4.1.  The single-step version of the
    same construction is White's Reality Check, see
    :func:`bzoo.corrections.bootstrap_tests.white_reality_check`.
    """
    return _stepdown_maxstat(obs, boot, alpha, "Romano-Wolf")


def westfall_young_maxt(
    obs: np.ndarray, perm: np.ndarray, alpha: float = 0.05
) -> CorrectionResult:
    """Westfall-Young max-T with permutation replicates.

    Identical arithmetic to :func:`romano_wolf`; the difference is where
    the replicate matrix comes from.  Permutation replicates are exactly
    null by construction (under the randomisation hypothesis), whereas
    bootstrap replicates are only asymptotically null.  We keep the two
    entry points separate so that the reported method name says which
    resampling scheme produced the number.
    """
    return _stepdown_maxstat(obs, perm, alpha, "Westfall-Young (max-T)")


def westfall_young_minp(
    p_obs: np.ndarray, p_perm: np.ndarray, alpha: float = 0.05
) -> CorrectionResult:
    """Westfall-Young min-P.

    Works on the p-value scale, which makes it valid when the marginal
    distributions differ across hypotheses (unequal sample sizes, unequal
    variances).  max-T implicitly assumes the statistics are comparable
    across hypotheses; min-P does not.

    Parameters
    ----------
    p_obs:
        ``K`` observed marginal p-values.
    p_perm:
        ``(B, K)`` matrix of marginal p-values recomputed on each
        permutation replicate.
    """
    p_obs = check_pvalues(p_obs)
    p_perm = np.asarray(p_perm, dtype=float)
    if p_perm.ndim != 2 or p_perm.shape[1] != p_obs.size:
        raise ValueError("p_perm must have shape (B, K) matching p_obs")
    b = p_perm.shape[0]
    k = p_obs.size

    order = np.argsort(p_obs, kind="mergesort")  # ascending: most significant first
    remaining = list(order)
    reject_sorted = np.zeros(k, dtype=bool)
    adj_sorted = np.ones(k, dtype=float)
    running_p = 0.0
    first_crit = None
    step = 0

    while remaining:
        idx = np.asarray(remaining, dtype=int)
        min_null = p_perm[:, idx].min(axis=1)
        j = remaining[0]
        p_step = (1.0 + np.sum(min_null <= p_obs[j])) / (b + 1.0)
        crit = np.quantile(min_null, alpha)
        if step == 0:
            first_crit = float(crit)
        running_p = max(running_p, p_step)
        adj_sorted[step] = running_p
        if p_obs[j] < crit:
            reject_sorted[step] = True
            remaining.pop(0)
            step += 1
        else:
            adj_sorted[step:] = running_p
            break

    reject = np.zeros(k, dtype=bool)
    adj = np.ones(k, dtype=float)
    reject[order] = reject_sorted
    adj[order] = adj_sorted
    return CorrectionResult(
        method="Westfall-Young (min-P)",
        n_tests=k,
        alpha=check_alpha(alpha),
        reject=reject,
        adjusted_p=adj,
        critical_value=first_crit,
        error_rate="FWER",
        extra={"n_replicates": b},
    )
