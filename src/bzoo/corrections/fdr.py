"""False discovery rate procedures.

``benjamini_hochberg``  Benjamini and Hochberg (1995), JRSS-B 57, 289-300.
``benjamini_yekutieli`` Benjamini and Yekutieli (2001), Annals of Statistics
                        29, 1165-1188.
``storey_qvalues``      Storey (2002), JRSS-B 64, 479-498, plus the bootstrap
                        choice of the tuning parameter from Storey, Taylor
                        and Siegmund (2004), JRSS-B 66, 187-205.

FDR is the right error rate for this project's finance half, where the
question is "what fraction of the predictors we call significant are
false", and FWER is the right one for the machine learning half, where the
question is "is this single reported improvement real".  We report both
everywhere rather than choosing.
"""

from __future__ import annotations

import numpy as np

from .base import CorrectionResult, check_alpha, check_pvalues


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg (1995) step-up procedure.

    Sort ascending, find the largest ``i`` with ``p_(i) <= i * alpha / K``,
    and reject hypotheses ``1..i``.  Controls FDR at ``alpha`` under
    independence and under positive regression dependence
    (Benjamini and Yekutieli 2001, Theorem 1.2).
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, k + 1)

    passed = p_sorted <= ranks * alpha / k
    n_rej = int(np.flatnonzero(passed).max() + 1) if passed.any() else 0

    reject_sorted = np.zeros(k, dtype=bool)
    reject_sorted[:n_rej] = True

    # q-values: running minimum from the largest p-value downwards.
    q_sorted = np.minimum.accumulate((k / ranks * p_sorted)[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    reject = np.empty(k, dtype=bool)
    q = np.empty(k, dtype=float)
    reject[order] = reject_sorted
    q[order] = q_sorted
    return CorrectionResult(
        method="Benjamini-Hochberg",
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=q,
        critical_value=float(p_sorted[n_rej - 1]) if n_rej else None,
        error_rate="FDR",
    )


def benjamini_yekutieli(p: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Yekutieli (2001) procedure for arbitrary dependence.

    BH with ``alpha`` replaced by ``alpha / c(K)`` where
    ``c(K) = sum_{i=1}^{K} 1/i`` is the harmonic number.  Valid under any
    dependence structure, at the cost of a factor of about ``log K``: with
    K = 20,000 the penalty is roughly 10.5, so it is very conservative on
    populations of the size we work with.  We report it as the pessimistic
    bound rather than as a recommendation.
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    c_k = float(np.sum(1.0 / np.arange(1, k + 1)))

    inner = benjamini_hochberg(p, alpha / c_k)
    q = np.minimum(inner.adjusted_p * c_k, 1.0)
    return CorrectionResult(
        method="Benjamini-Yekutieli",
        n_tests=k,
        alpha=alpha,
        reject=inner.reject,
        adjusted_p=q,
        critical_value=inner.critical_value,
        error_rate="FDR",
        extra={"harmonic_c": c_k},
    )


def estimate_pi0(
    p: np.ndarray,
    lambdas: "np.ndarray | None" = None,
    method: str = "bootstrap",
    n_boot: int = 100,
    seed: int = 0,
) -> "tuple[float, dict]":
    """Estimate the null proportion pi0 for Storey's procedure.

    The estimator is ``pi0(lambda) = #{p > lambda} / (K * (1 - lambda))``:
    p-values above ``lambda`` are assumed to be almost all null, and the
    null ones are uniform, so we scale up the count.  Small ``lambda``
    gives low variance and upward bias, large ``lambda`` the reverse.

    ``method="bootstrap"`` picks ``lambda`` by minimising the estimated
    mean squared error against ``min_lambda pi0(lambda)``, which is
    Storey, Taylor and Siegmund (2004), Section 6 / the ``qvalue`` package
    default.  ``method="fixed"`` uses ``lambda = 0.5``, the value in the
    original Storey (2002) paper.
    """
    p = check_pvalues(p)
    k = p.size
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)
    lambdas = np.asarray(lambdas, dtype=float)

    def _pi0_grid(pv: np.ndarray) -> np.ndarray:
        counts = np.array([np.sum(pv > lam) for lam in lambdas], dtype=float)
        return counts / (pv.size * (1.0 - lambdas))

    grid = np.minimum(_pi0_grid(p), 1.0)

    if method == "fixed":
        lam = 0.5
        pi0 = float(min(1.0, np.sum(p > lam) / (k * (1.0 - lam))))
        return pi0, {"lambda": lam, "method": "fixed"}

    if method != "bootstrap":
        raise ValueError("method must be 'bootstrap' or 'fixed'")

    rng = np.random.default_rng(seed)
    min_pi0 = float(np.min(grid))
    boot = np.empty((n_boot, lambdas.size), dtype=float)
    for b in range(n_boot):
        sample = p[rng.integers(0, k, size=k)]
        boot[b] = np.minimum(_pi0_grid(sample), 1.0)
    mse = np.mean((boot - min_pi0) ** 2, axis=0)
    best = int(np.argmin(mse))
    pi0 = float(min(1.0, grid[best]))
    return pi0, {"lambda": float(lambdas[best]), "method": "bootstrap", "pi0_min": min_pi0}


def storey_qvalues(
    p: np.ndarray,
    alpha: float = 0.05,
    pi0: "float | None" = None,
    seed: int = 0,
) -> CorrectionResult:
    """Storey (2002) q-values, controlling the positive FDR.

    Same step-up shape as BH but with the multiplicity factor ``K``
    replaced by ``pi0 * K``.  When most hypotheses are truly null,
    ``pi0`` is close to 1 and the two agree; when many are non-null it is
    noticeably more powerful.  On the mined finance populations ``pi0`` is
    an interesting number in its own right, so we return it in ``extra``.
    """
    p = check_pvalues(p)
    alpha = check_alpha(alpha)
    k = p.size
    info: dict = {}
    if pi0 is None:
        pi0, info = estimate_pi0(p, seed=seed)

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, k + 1)
    q_sorted = np.minimum.accumulate((pi0 * k / ranks * p_sorted)[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    reject_sorted = q_sorted <= alpha
    reject = np.empty(k, dtype=bool)
    q = np.empty(k, dtype=float)
    reject[order] = reject_sorted
    q[order] = q_sorted

    info["pi0"] = float(pi0)
    return CorrectionResult(
        method="Storey q-value",
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=q,
        critical_value=None,
        error_rate="pFDR",
        extra=info,
    )
