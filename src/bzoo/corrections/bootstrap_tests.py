"""Bootstrap tests for the best of many candidates.

``white_reality_check``  White (2000), Econometrica 68, 1097-1126.
``hansen_spa``           Hansen (2005), JBES 23, 365-380.

Both test the same null hypothesis: no candidate beats the benchmark,

    H0:  max_k  E[d_k] <= 0,

where ``d_k`` is the per-period (finance) or per-instance (ML) performance
difference between candidate ``k`` and the benchmark.  They differ in two
ways.  SPA studentises, so a candidate with a noisy metric does not
dominate the maximum; and SPA recentres the bootstrap distribution so that
candidates far below the benchmark do not inflate the p-value.  White's
statistic has neither property, which is why adding obviously bad
candidates to a Reality Check makes it harder to reject.  That contrast is
the reason we report both.

Both functions take the ``(T, K)`` differential matrix and resample *rows*,
so all candidates always share the same resampled sample.  There is no way
to pass in per-candidate draws, which is deliberate.
"""

from __future__ import annotations

import numpy as np

from ..resample.stationary import optimal_block_length, stationary_bootstrap_indices
from .base import CorrectionResult, check_alpha


def _row_indices(
    n_obs: int,
    n_boot: int,
    scheme: str,
    block_length: "float | None",
    rng: np.random.Generator,
) -> "tuple[np.ndarray, float]":
    if scheme == "iid":
        return rng.integers(0, n_obs, size=(n_boot, n_obs), dtype=np.int64), 1.0
    if scheme == "stationary":
        if block_length is None:
            raise ValueError("stationary scheme needs a block length")
        return (
            stationary_bootstrap_indices(n_obs, n_boot, block_length, rng),
            float(block_length),
        )
    raise ValueError("scheme must be 'iid' or 'stationary'")


def _bootstrap_means(d: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """``(B, K)`` bootstrap means, one shared row draw per replicate."""
    return d[indices].mean(axis=1)


def white_reality_check(
    d: np.ndarray,
    n_boot: int = 1000,
    scheme: str = "stationary",
    block_length: "float | None" = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> CorrectionResult:
    """White's (2000) Reality Check p-value for the best candidate.

    Parameters
    ----------
    d:
        ``(T, K)`` performance differentials, candidate minus benchmark.
    scheme:
        ``"stationary"`` for time series, ``"iid"`` for exchangeable
        instances.  With ``"stationary"`` and ``block_length=None`` the
        block length is chosen by :func:`optimal_block_length` on the
        column with the largest mean.

    Returns
    -------
    A :class:`CorrectionResult` whose ``reject`` array is ``True`` only for
    the single best candidate, and only if the joint null is rejected.  The
    Reality Check is a test of one composite hypothesis, not a per-candidate
    procedure; the step-down version that is per-candidate is
    :func:`bzoo.corrections.fwer.romano_wolf`.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        raise ValueError("d must be (T, K)")
    t, k = d.shape
    alpha = check_alpha(alpha)
    rng = np.random.default_rng(seed)

    dbar = d.mean(axis=0)
    if scheme == "stationary" and block_length is None:
        block_length = optimal_block_length(d[:, int(np.argmax(dbar))])

    idx, bl = _row_indices(t, n_boot, scheme, block_length, rng)
    boot = _bootstrap_means(d, idx)

    v_obs = np.sqrt(t) * dbar.max()
    v_boot = (np.sqrt(t) * (boot - dbar)).max(axis=1)
    p_value = float((1.0 + np.sum(v_boot >= v_obs)) / (n_boot + 1.0))

    reject = np.zeros(k, dtype=bool)
    if p_value <= alpha:
        reject[int(np.argmax(dbar))] = True

    return CorrectionResult(
        method="White Reality Check",
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=None,
        critical_value=float(np.quantile(v_boot, 1.0 - alpha)) / np.sqrt(t),
        error_rate="FWER",
        extra={
            "p_value": p_value,
            "statistic": float(v_obs),
            "block_length": bl,
            "scheme": scheme,
            "n_replicates": n_boot,
        },
    )


def hansen_spa(
    d: np.ndarray,
    n_boot: int = 1000,
    scheme: str = "stationary",
    block_length: "float | None" = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> CorrectionResult:
    """Hansen's (2005) test for superior predictive ability.

    Returns the consistent p-value ``p_c`` in ``extra["p_value"]`` and the
    lower and upper bounds ``p_l`` and ``p_u`` alongside it, following
    Hansen (2005), Section 3.  The three differ only in how the bootstrap
    distribution is recentred:

    ==========  ============================================  ==============
    version     recentring ``g_k``                             ordering
    ==========  ============================================  ==============
    lower       ``max(0, dbar_k)``                             smallest p
    consistent  ``dbar_k * 1{dbar_k > -A_k}``                   in between
    upper       ``dbar_k``                                      largest p
    ==========  ============================================  ==============

    with ``A_k = omega_k sqrt(2 log log T) / sqrt(T)``, the
    law-of-the-iterated-logarithm rate: a candidate is dropped from the null
    distribution only once its mean is far enough below zero that it is
    almost surely a genuinely inferior model.  The rate is what makes the
    classification consistent, so that the p-value has the right limit
    whether or not any candidate is truly bad.

    The upper version keeps every candidate in the null distribution, which
    is exactly White's recentring; that is why the Reality Check p-value
    tracks ``p_u``.

    The test statistic is ``max(max_k sqrt(T) dbar_k / omega_k, 0)``, the
    studentised form Hansen recommends.  ``omega_k`` is estimated from the
    same bootstrap draws that produce the null distribution.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        raise ValueError("d must be (T, K)")
    t, k = d.shape
    alpha = check_alpha(alpha)
    rng = np.random.default_rng(seed)

    dbar = d.mean(axis=0)
    if scheme == "stationary" and block_length is None:
        block_length = optimal_block_length(d[:, int(np.argmax(dbar))])

    idx, bl = _row_indices(t, n_boot, scheme, block_length, rng)
    boot = _bootstrap_means(d, idx)

    # omega_k: bootstrap estimate of the standard deviation of sqrt(T)*dbar_k.
    omega = np.sqrt(t) * boot.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)

    stat = float(np.max(np.maximum(0.0, np.sqrt(t) * dbar / omega)))

    # Threshold on the dbar scale; log(log(T)) needs T > e, which any usable
    # sample satisfies, but guard it anyway.
    lil = np.sqrt(2.0 * np.log(np.log(max(float(t), 3.0))))
    a_k = omega * lil / np.sqrt(t)
    g_lower = np.maximum(0.0, dbar)
    g_cons = np.where(dbar > -a_k, dbar, 0.0)
    g_upper = dbar

    p_values = {}
    for name, g in (("lower", g_lower), ("consistent", g_cons), ("upper", g_upper)):
        z = np.sqrt(t) * (boot - g) / omega
        z_max = np.maximum(0.0, z).max(axis=1)
        p_values[name] = float((1.0 + np.sum(z_max >= stat)) / (n_boot + 1.0))

    reject = np.zeros(k, dtype=bool)
    if p_values["consistent"] <= alpha:
        reject[int(np.argmax(np.sqrt(t) * dbar / omega))] = True

    return CorrectionResult(
        method="Hansen SPA",
        n_tests=k,
        alpha=alpha,
        reject=reject,
        adjusted_p=None,
        critical_value=None,
        error_rate="FWER",
        extra={
            "p_value": p_values["consistent"],
            "p_lower": p_values["lower"],
            "p_upper": p_values["upper"],
            "statistic": stat,
            "block_length": bl,
            "scheme": scheme,
            "n_replicates": n_boot,
        },
    )


def bootstrap_centred_matrix(
    d: np.ndarray,
    n_boot: int = 1000,
    scheme: str = "stationary",
    block_length: "float | None" = None,
    studentised: bool = True,
    seed: int = 0,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Build the inputs Romano-Wolf needs from a differential matrix.

    Returns ``(observed, centred_replicates, block_length)``.  Centring
    subtracts the observed column means, which is the bootstrap analogue of
    imposing the null.  With ``studentised=True`` both are divided by the
    bootstrap standard deviation, which is what makes the step-down
    procedure comparable across candidates with different noise levels.
    """
    d = np.asarray(d, dtype=float)
    t = d.shape[0]
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    if scheme == "stationary" and block_length is None:
        block_length = optimal_block_length(d[:, int(np.argmax(dbar))])
    idx, bl = _row_indices(t, n_boot, scheme, block_length, rng)
    boot = _bootstrap_means(d, idx)

    obs = np.sqrt(t) * dbar
    cent = np.sqrt(t) * (boot - dbar)
    if studentised:
        sd = cent.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        obs = obs / sd
        cent = cent / sd
    return obs, cent, bl
