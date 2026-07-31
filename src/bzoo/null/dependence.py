"""How many independent tests are we really running?

Corrections that take a trial count N as input implicitly assume the N tests
are independent.  They never are.  Two mined strategies that overlap in
holdings, or two model configurations that differ in one hyperparameter, are
close to the same test run twice, and charging the family two units of
multiplicity is wrong.

Three ways of putting a number on it, all reported:

``effective_n_eigen``
    From the eigenvalue spectrum of the correlation matrix.  Cheap, needs
    only second moments, and there are two published variants that can
    disagree by a factor of two, so we report both.
``effective_n_sidak``
    From the distribution of the maximum statistic.  Defines the effective
    count as the ``N_eff`` that makes an independence-based correction give
    the same threshold as the observed maximum.  This is the one to prefer:
    it is defined in terms of the quantity the corrections actually use, and
    it needs no assumption about where the dependence comes from.
``max_stat_distribution``
    The maximum distribution itself, which is the primitive object the other
    two summarise.

The eigenvalue methods are the ones in common use, and they answer a
different question: how many dimensions does the correlation matrix have.
That coincides with the multiplicity question only when the statistics are
jointly normal with equal variances.  We report them for comparability with
the existing literature and use the Sidak-implied count for the results.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def correlation_eigenvalues(
    panel: np.ndarray,
    max_columns: Optional[int] = None,
    seed: int = 0,
    demean: bool = True,
) -> np.ndarray:
    """Eigenvalues of the column correlation matrix of a ``(T, K)`` panel.

    For ``K > T`` the correlation matrix is singular with at least ``K - T``
    zero eigenvalues; those are kept, because both effective-count formulas
    are defined over all ``K`` of them.

    When ``K`` is large, forming the ``K x K`` matrix is the expensive step
    (19,380 columns is 3 GB in float64).  We instead take the eigenvalues of
    the much smaller ``T x T`` Gram matrix of the standardised data, which
    are the same non-zero eigenvalues, and pad with zeros.  ``max_columns``
    subsamples columns instead, for the sensitivity check.
    """
    x = np.asarray(panel, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("panel must be 2-D (T, K)")
    if max_columns is not None and x.shape[1] > max_columns:
        rng = np.random.default_rng(seed)
        cols = rng.choice(x.shape[1], size=max_columns, replace=False)
        x = x[:, np.sort(cols)]

    keep = np.isfinite(x).all(axis=0)
    x = x[:, keep]
    t, k = x.shape
    if demean:
        x = x - x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    good = sd > 0
    x = x[:, good] / sd[good]
    k = x.shape[1]

    # Correlation matrix is (1/(T-1)) Z'Z with Z standardised; its non-zero
    # eigenvalues equal those of (1/(T-1)) Z Z', which is only T x T.
    gram = (x @ x.T) / (t - 1)
    ev_small = np.linalg.eigvalsh(gram)[::-1]
    ev = np.zeros(k)
    m = min(k, ev_small.size)
    ev[:m] = np.clip(ev_small[:m], 0.0, None)
    # Eigenvalues of a correlation matrix sum to K; rescale to remove the
    # small numerical drift so that the formulas below stay exact.
    total = ev.sum()
    if total > 0:
        ev = ev * (k / total)
    return ev


def effective_n_eigen(eigenvalues: np.ndarray) -> Dict[str, float]:
    """Effective number of independent tests from an eigenvalue spectrum.

    Two published variants:

    Cheverud (2001), Heredity 87, 52-58, and Nyholt (2004), American Journal
    of Human Genetics 74, 765-769::

        M_eff = 1 + (M - 1) * (1 - Var(lambda) / M)

    Li and Ji (2005), Heredity 95, 221-227::

        M_eff = sum_i [ I(|lambda_i| >= 1) + (|lambda_i| - floor(|lambda_i|)) ]

    They can differ by a factor of two on the same matrix, which is the main
    reason we do not rely on either for the headline numbers.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    m = ev.size
    if m < 2:
        raise ValueError("need at least two eigenvalues")
    var_l = float(np.var(ev, ddof=1))
    cheverud = 1.0 + (m - 1.0) * (1.0 - var_l / m)
    a = np.abs(ev)
    li_ji = float(np.sum((a >= 1.0).astype(float) + (a - np.floor(a)) * (a < 1.0)))
    return {
        "m": float(m),
        "var_eigenvalue": var_l,
        "n_eff_cheverud_nyholt": float(cheverud),
        "n_eff_li_ji": li_ji,
        "ratio_cheverud": float(cheverud / m),
        "ratio_li_ji": float(li_ji / m),
        "top1_share": float(ev[0] / m),
        "top10_share": float(ev[: min(10, m)].sum() / m),
    }


def max_stat_distribution(
    replicates: np.ndarray, two_sided: bool = True
) -> np.ndarray:
    """Per-replicate maximum over columns of a ``(B, K)`` null matrix."""
    r = np.asarray(replicates, dtype=float)
    if r.ndim != 2:
        raise ValueError("replicates must be (B, K)")
    return (np.abs(r) if two_sided else r).max(axis=1)


def effective_n_sidak(
    max_stats: np.ndarray,
    n_tests: int,
    alpha: float = 0.05,
    two_sided: bool = True,
    marginal_sd: float = 1.0,
    marginal_sample: "np.ndarray | None" = None,
) -> Dict[str, float]:
    """Effective test count implied by the observed maximum distribution.

    Take the ``1 - alpha`` quantile ``q`` of the maximum statistic.  Under
    independence, Sidak says the maximum of ``N_eff`` tests exceeds ``q``
    with probability ``alpha`` when

        ``1 - (1 - p1(q))^N_eff = alpha``,

    where ``p1(q)`` is the marginal probability of a single statistic
    exceeding ``q``.  Solving,

        ``N_eff = log(1 - alpha) / log(1 - p1(q))``.

    Using the *empirical* marginal for ``p1`` separates the two effects: the
    marginal shape is already in ``p1``, so ``N_eff`` measures dependence
    alone.  We also return the version that uses a Gaussian marginal, which
    is what someone applying Bonferroni without calibration is implicitly
    using.

    ``marginal_sd`` scales the Gaussian marginal, so passing the measured
    standard deviation gives the intermediate case: calibrated scale, assumed
    shape.  ``marginal_sample`` goes one step further and estimates ``p1``
    directly from a sample of individual null statistics, which assumes
    nothing about the marginal at all.  That version is the one that isolates
    dependence, because the marginal is then held fixed by construction; the
    other two mix dependence with marginal misspecification, in the way that
    someone applying Bonferroni to a nominal null implicitly does.
    """
    max_stats = np.asarray(max_stats, dtype=float)
    max_stats = max_stats[np.isfinite(max_stats)]
    q = float(np.quantile(max_stats, 1.0 - alpha))

    def solve(p1: float) -> float:
        if p1 <= 0.0:
            return float("inf")
        if p1 >= 1.0:
            return 1.0
        return float(np.log1p(-alpha) / np.log1p(-p1))

    tail = 2.0 if two_sided else 1.0
    p1_gauss = float(tail * stats.norm.sf(q))
    p1_scaled = float(tail * stats.norm.sf(q / marginal_sd))

    out = {
        "n_tests": float(n_tests),
        "max_quantile": q,
        "p1_gaussian": p1_gauss,
        "n_eff_gaussian_marginal": solve(p1_gauss),
        "p1_scaled": p1_scaled,
        "n_eff_scaled_marginal": solve(p1_scaled),
        "ratio_gaussian": solve(p1_gauss) / n_tests,
        "ratio_scaled": solve(p1_scaled) / n_tests,
        "alpha": alpha,
    }
    if marginal_sample is not None:
        m = np.asarray(marginal_sample, dtype=float)
        m = m[np.isfinite(m)]
        vals = np.abs(m) if two_sided else m
        # Plus-one convention so p1 cannot be exactly zero.
        p1_emp = float((1.0 + np.sum(vals > q)) / (vals.size + 1.0))
        out["p1_empirical"] = p1_emp
        out["n_marginal_sample"] = float(vals.size)
        out["n_eff_empirical_marginal"] = solve(p1_emp)
        out["ratio_empirical"] = solve(p1_emp) / n_tests
    return out


def effective_n_from_panel(
    panel: np.ndarray,
    n_boot: int = 2000,
    block_length: Optional[float] = None,
    alpha: float = 0.05,
    seed: int = 0,
    two_sided: bool = True,
) -> Dict[str, float]:
    """Effective test count from a return panel, via a block bootstrap.

    Resamples months jointly across strategies, recomputes every
    t-statistic on each replicate, and takes the maximum.  This is the
    permutation/bootstrap max-T route referred to in the research plan, and
    it is the version we prefer, because it never forms a correlation matrix
    and so makes no linearity assumption about how the strategies co-move.
    """
    from ..resample.stationary import (
        optimal_block_length,
        stationary_bootstrap_indices,
    )

    x = np.asarray(panel, dtype=np.float64)
    keep = np.isfinite(x).all(axis=0)
    x = x[:, keep]
    t, k = x.shape
    if block_length is None:
        block_length = optimal_block_length(x[:, 0])

    rng = np.random.default_rng(seed)
    # Centre columns so the bootstrap imposes the zero-mean null exactly.
    xc = x - x.mean(axis=0)

    maxima = np.empty(n_boot)
    batch = max(1, int(2e7 // max(1, t * k)))  # keep peak memory near 150 MB
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = stationary_bootstrap_indices(t, b, block_length, rng)
        for i in range(b):
            s = xc[idx[i]]
            m = s.mean(axis=0)
            sd = s.std(axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                tt = m / (sd / np.sqrt(t))
            tt = np.where(np.isfinite(tt), tt, 0.0)
            maxima[done + i] = np.max(np.abs(tt) if two_sided else tt)
        done += b

    out = effective_n_sidak(maxima, k, alpha=alpha, two_sided=two_sided)
    out["block_length"] = float(block_length)
    out["n_replicates"] = float(n_boot)
    out["max_mean"] = float(maxima.mean())
    out["max_sd"] = float(maxima.std(ddof=1))
    out["max_q95"] = float(np.quantile(maxima, 0.95))
    out["max_q99"] = float(np.quantile(maxima, 0.99))
    return out


def max_t_permutation(
    panel: np.ndarray,
    n_perm: int = 20000,
    block_length: int = 1,
    seed: int = 0,
    two_sided: bool = True,
    batch: int = 2000,
    joint: bool = True,
    return_marginals: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Null distribution of the maximum t-statistic, by block sign flipping.

    Under the null that every strategy has zero expected return, the sign of a
    month's cross-section is exchangeable.  Flipping the sign of whole blocks
    of months, with the *same* sign applied to every strategy in a block,
    gives an exactly null replicate that preserves the cross-sectional
    dependence and the within-block autocorrelation.

    Holding the standard deviation at its sample value - which Hansen (2005)
    notes is the usual choice, since a sign flip barely changes it - turns the
    whole calculation into one matrix product per batch of replicates:

        ``t*_bk = (s_b . x_k) / (sqrt(T) sigma_k)``

    That is thousands of times faster than recomputing t-statistics inside a
    loop, which matters at 19,380 strategies, and it is what makes the
    conditional splits by decade and volatility regime affordable.

    ``joint=True`` applies one sign vector to the whole cross-section, which
    is the correct scheme.  ``joint=False`` draws an independent sign vector
    per strategy, which leaves every strategy's marginal distribution exactly
    unchanged but makes the columns independent.  Comparing the two maxima
    isolates the effect of dependence at matched marginals, which is what
    :func:`effective_n_sidak` needs and what the joint-versus-independent
    robustness check reports.

    Returns the ``n_perm`` maxima, and with ``return_marginals=True`` also a
    pooled sample of the individual studentised statistics, for use as the
    empirical marginal.
    """
    x = np.asarray(panel, dtype=np.float64)
    keep = np.isfinite(x).all(axis=0)
    x = x[:, keep]
    t, k = x.shape
    if k == 0:
        raise ValueError("no complete columns in the panel")
    xc = x - x.mean(axis=0)
    sd = xc.std(axis=0, ddof=1)
    good = sd > 0
    xc = xc[:, good] / (sd[good] * np.sqrt(t))
    xc = xc.astype(np.float32)

    block_length = max(1, int(block_length))
    n_blocks = int(np.ceil(t / block_length))
    rng = np.random.default_rng(seed)

    k_use = xc.shape[1]
    out = np.empty(n_perm, dtype=np.float64)
    marg = [] if return_marginals else None
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        if joint:
            bs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(b, n_blocks))
            signs = np.repeat(bs, block_length, axis=1)[:, :t]
            stats_mat = signs @ xc  # (b, K) studentised means
        else:
            # An independent sign vector per strategy.  Done column-block by
            # column-block so the (b, T, K) intermediate is never formed.
            stats_mat = np.empty((b, k_use), dtype=np.float32)
            step = max(1, int(4e7 // (b * t)))
            for j0 in range(0, k_use, step):
                j1 = min(k_use, j0 + step)
                bs = rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float32),
                    size=(b, n_blocks, j1 - j0),
                )
                sg = np.repeat(bs, block_length, axis=1)[:, :t, :]
                stats_mat[:, j0:j1] = np.einsum(
                    "btj,tj->bj", sg, xc[:, j0:j1], optimize=True
                )
        vals = np.abs(stats_mat) if two_sided else stats_mat
        out[done : done + b] = vals.max(axis=1)
        if return_marginals:
            # Keep a bounded random slice, enough to estimate tail
            # probabilities down to about 1e-7.
            marg.append(vals[:, rng.integers(0, k_use, size=min(k_use, 200))].ravel())
        done += b
    if return_marginals:
        return out, np.concatenate(marg).astype(np.float64)
    return out


def dependence_report(
    panel: np.ndarray,
    n_boot: int = 1000,
    max_columns_for_eigen: Optional[int] = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """One table with every effective-count estimate for a panel."""
    ev = correlation_eigenvalues(panel, max_columns=max_columns_for_eigen, seed=seed)
    eig = effective_n_eigen(ev)
    boot = effective_n_from_panel(panel, n_boot=n_boot, alpha=alpha, seed=seed)

    rows = [
        {
            "method": "Cheverud-Nyholt eigenvalue",
            "n_eff": eig["n_eff_cheverud_nyholt"],
            "n_tests": eig["m"],
            "ratio": eig["ratio_cheverud"],
        },
        {
            "method": "Li-Ji eigenvalue",
            "n_eff": eig["n_eff_li_ji"],
            "n_tests": eig["m"],
            "ratio": eig["ratio_li_ji"],
        },
        {
            "method": "Sidak-implied, bootstrap max-t",
            "n_eff": boot["n_eff_gaussian_marginal"],
            "n_tests": boot["n_tests"],
            "ratio": boot["ratio_gaussian"],
        },
    ]
    return pd.DataFrame(rows)
