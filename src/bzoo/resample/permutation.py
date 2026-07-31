"""Permutation replicates, the non-parametric alternative.

The closed-form deflation of :mod:`bzoo.corrections.deflation` assumes the
maximum of the trial statistics is approximately normal.  Bounded metrics
near a ceiling are skewed, so that assumption has to be checked rather than
asserted.  These routines give a replicate matrix that makes no
distributional assumption at all, so the two can be compared on the same
data.

Two schemes:

``sign_flip_replicates``
    For paired differences.  Under the null that model and baseline have the
    same expected metric, the sign of each per-instance difference is
    exchangeable, so multiplying each instance's difference by a random
    +/-1 gives an exactly null replicate.  The same sign vector is applied
    to every model, which preserves the cross-model dependence.

``block_sign_flip_replicates``
    The same idea for the finance testbed, flipping the sign of whole
    blocks of consecutive months rather than single months, so that
    autocorrelation within a block survives.
"""

from __future__ import annotations

import numpy as np


def sign_flip_replicates(
    diffs: np.ndarray, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_perm, n_models)`` null means from shared sign flips.

    Parameters
    ----------
    diffs:
        ``(n_units, n_models)`` paired differences.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.ndim != 2:
        raise ValueError("diffs must be (n_units, n_models)")
    n_units = diffs.shape[0]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_units))
    return (signs @ diffs) / n_units


def block_sign_flip_replicates(
    diffs: np.ndarray, n_perm: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Sign flips applied to blocks of consecutive rows.

    ``block_length`` should be at least the horizon over which the series
    is autocorrelated.  With ``block_length=1`` this reduces to
    :func:`sign_flip_replicates`.
    """
    diffs = np.asarray(diffs, dtype=float)
    n_units = diffs.shape[0]
    block_length = max(1, int(block_length))
    n_blocks = int(np.ceil(n_units / block_length))
    block_signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_blocks))
    signs = np.repeat(block_signs, block_length, axis=1)[:, :n_units]
    return (signs @ diffs) / n_units


def studentise(replicates: np.ndarray, observed: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Divide replicates and observation by the replicate standard deviation.

    Max-T over raw statistics favours whichever hypothesis happens to have
    the largest variance.  Studentising removes that, and is what Hansen
    (2005) does inside the SPA test.
    """
    replicates = np.asarray(replicates, dtype=float)
    observed = np.asarray(observed, dtype=float).ravel()
    sd = replicates.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)  # a constant column can never reject
    return replicates / sd, observed / sd
