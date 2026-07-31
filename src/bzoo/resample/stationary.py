"""Stationary bootstrap of Politis and Romano (1994), JASA 89, 1303-1313.

Used for the finance testbed, where the sampling unit is a month and
returns are mildly autocorrelated.  Blocks have geometric length with mean
``block_length``, and the series is wrapped around so that every
observation has the same probability of being drawn.  That wrapping is what
makes the resampled series stationary, which the fixed-block bootstrap is
not.

The automatic block length is Politis and White (2004), Econometric Reviews
23, 53-70, with the correction in Patton, Politis and White (2009).
"""

from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(
    n_obs: int,
    n_boot: int,
    block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``(n_boot, n_obs)`` indices for the stationary bootstrap.

    Each replicate starts at a uniform position.  At every step we either
    continue the current block (probability ``1 - 1/block_length``) or jump
    to a new uniform position.  Indices wrap modulo ``n_obs``.
    """
    if n_obs < 2:
        raise ValueError("need at least two observations")
    if block_length < 1.0:
        raise ValueError("block_length must be at least 1")

    p_new = 1.0 / float(block_length)
    idx = np.empty((n_boot, n_obs), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, size=n_boot)
    new_block = rng.random((n_boot, n_obs - 1)) < p_new
    fresh = rng.integers(0, n_obs, size=(n_boot, n_obs - 1))
    for t in range(1, n_obs):
        cont = (idx[:, t - 1] + 1) % n_obs
        idx[:, t] = np.where(new_block[:, t - 1], fresh[:, t - 1], cont)
    return idx


def _lam(s: np.ndarray) -> np.ndarray:
    """Flat-top lag window of Politis and Romano (1995), used by the
    automatic block-length rule."""
    out = np.zeros_like(s, dtype=float)
    a = np.abs(s)
    out[a <= 0.5] = 1.0
    mid = (a > 0.5) & (a <= 1.0)
    out[mid] = 2.0 * (1.0 - a[mid])
    return out


def optimal_block_length(x: np.ndarray, k_n: "int | None" = None) -> float:
    """Automatic block length for the stationary bootstrap.

    Implements Politis and White (2004) as corrected by Patton, Politis and
    White (2009).  Returns ``b_opt = (2 * G^2 / D)^(1/3) * n^(1/3)``, where
    ``G`` and ``D`` are built from the estimated autocovariances up to a
    data-dependent lag.

    We clip the answer to ``[1, n/3]``: the rule is derived asymptotically
    and can return absurd lengths on short or nearly white-noise series,
    and a block longer than a third of the sample destroys the point of
    resampling.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return 1.0
    x = x - x.mean()

    # Maximum lag considered, following the reference implementation.
    m_max = int(np.ceil(np.sqrt(n)))
    k_n = int(max(5, np.sqrt(np.log10(n)))) if k_n is None else k_n
    var = float(np.dot(x, x) / n)
    if var <= 0.0:
        return 1.0

    rho = np.empty(m_max + 1, dtype=float)
    for lag in range(m_max + 1):
        rho[lag] = float(np.dot(x[: n - lag], x[lag:]) / n) / var

    # m_hat: smallest lag beyond which k_n consecutive autocorrelations are
    # all inside the 2/sqrt(n) band.
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = 1
    for lag in range(1, m_max + 1):
        window = rho[lag : min(lag + k_n, m_max + 1)]
        if window.size and np.all(np.abs(window) < crit):
            m_hat = lag
            break
        m_hat = lag
    m = min(2 * m_hat, m_max)
    m = max(m, 1)

    lags = np.arange(-m, m + 1)
    gamma = np.array([var * rho[abs(int(l))] for l in lags], dtype=float)
    w = _lam(lags / m)
    g_hat = float(np.sum(w * np.abs(lags) * gamma))
    d_hat = 2.0 * (float(np.sum(w * gamma))) ** 2  # stationary-bootstrap constant

    if d_hat <= 0.0 or g_hat == 0.0:
        return 1.0
    b_opt = (2.0 * g_hat ** 2 / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b_opt, 1.0, max(1.0, n / 3.0)))
