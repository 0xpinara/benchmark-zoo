"""Tail behaviour of the empirical null.

Multiplicity corrections live in the tail.  A correction that pushes the
threshold from 1.96 to 4.5 depends entirely on the shape of the null past
4.0, where a finite sample has very few observations.  Fitting a generalised
Pareto distribution to the exceedances is the standard way to extrapolate
there: by the Pickands-Balkema-de Haan theorem the conditional distribution
of exceedances over a high threshold converges to a GPD, whatever the parent
distribution is, so the two fitted parameters summarise the tail without
assuming a parent family.

Interpretation of the shape parameter ``xi``:

* ``xi < 0``  the tail ends at a finite point; lighter than exponential
* ``xi = 0``  exponential tail; this is the Gaussian case in the limit
* ``xi > 0``  polynomial tail, heavier than any normal

The Gaussian null is in the ``xi = 0`` (Gumbel) domain of attraction, so a
fitted ``xi`` that is reliably positive is direct evidence that the
theoretical null understates extreme statistics, and a reliably negative one
is evidence that it overstates them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GPDFit:
    threshold: float
    n_exceed: int
    n_total: int
    rate: float  # P(X > threshold)
    xi: float  # shape
    beta: float  # scale
    xi_se: float
    beta_se: float
    ks_pvalue: float
    log_likelihood: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def tail_probability(self, x: float) -> float:
        """P(X > x) for ``x`` above the fitting threshold.

        ``rate * (1 + xi (x - u)/beta)^(-1/xi)``, the standard peaks-over-
        threshold formula, with the ``xi -> 0`` limit handled explicitly.
        """
        if x < self.threshold:
            raise ValueError("x must be at or above the fitting threshold")
        z = (x - self.threshold) / self.beta
        if abs(self.xi) < 1e-8:
            return float(self.rate * np.exp(-z))
        base = 1.0 + self.xi * z
        if base <= 0:
            return 0.0
        return float(self.rate * base ** (-1.0 / self.xi))

    def quantile(self, p_tail: float) -> float:
        """Value ``x`` with ``P(X > x) = p_tail``, inverting the above."""
        if not 0.0 < p_tail < self.rate:
            raise ValueError("p_tail must lie in (0, rate)")
        r = p_tail / self.rate
        if abs(self.xi) < 1e-8:
            return float(self.threshold - self.beta * np.log(r))
        return float(self.threshold + self.beta / self.xi * (r ** (-self.xi) - 1.0))


def fit_gpd(
    x: np.ndarray,
    threshold: Optional[float] = None,
    quantile: float = 0.90,
) -> GPDFit:
    """Fit a GPD to the upper tail of ``x`` by maximum likelihood.

    Parameters
    ----------
    threshold:
        Fitting threshold ``u``.  If ``None``, the ``quantile`` empirical
        quantile is used.  The choice is a bias-variance trade-off: too low
        and the asymptotic approximation does not hold, too high and there is
        nothing left to fit.  :func:`threshold_stability` is the diagnostic
        for it, and we report a stability plot rather than defending a single
        value.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_total = x.size
    u = float(np.quantile(x, quantile)) if threshold is None else float(threshold)
    exc = x[x > u] - u
    if exc.size < 50:
        raise ValueError(f"only {exc.size} exceedances above {u}; need at least 50")

    xi, loc, beta = stats.genpareto.fit(exc, floc=0.0)
    ll = float(np.sum(stats.genpareto.logpdf(exc, xi, loc=0.0, scale=beta)))
    ks = stats.kstest(exc, "genpareto", args=(xi, 0.0, beta))

    # Asymptotic standard errors from the observed information matrix,
    # computed by central differences on the profile log-likelihood.  The
    # closed-form Fisher information for the GPD is only valid for xi > -0.5,
    # and numerical differencing behaves better near the boundary.
    xi_se, beta_se = _gpd_standard_errors(exc, xi, beta)

    return GPDFit(
        threshold=u,
        n_exceed=int(exc.size),
        n_total=int(n_total),
        rate=float(exc.size / n_total),
        xi=float(xi),
        beta=float(beta),
        xi_se=xi_se,
        beta_se=beta_se,
        ks_pvalue=float(ks.pvalue),
        log_likelihood=ll,
    )


def _gpd_standard_errors(
    exc: np.ndarray, xi: float, beta: float
) -> "tuple[float, float]":
    def nll(params: np.ndarray) -> float:
        s, sc = params
        if sc <= 0:
            return np.inf
        v = stats.genpareto.logpdf(exc, s, loc=0.0, scale=sc)
        if not np.all(np.isfinite(v)):
            return np.inf
        return -float(np.sum(v))

    theta = np.array([xi, beta])
    h = np.array([max(1e-4, 1e-3 * abs(xi)), 1e-3 * beta])
    hess = np.zeros((2, 2))
    f0 = nll(theta)
    for i in range(2):
        for j in range(2):
            if i == j:
                ei = np.zeros(2)
                ei[i] = h[i]
                hess[i, i] = (nll(theta + ei) - 2 * f0 + nll(theta - ei)) / h[i] ** 2
            else:
                ei = np.zeros(2)
                ej = np.zeros(2)
                ei[i] = h[i]
                ej[j] = h[j]
                hess[i, j] = (
                    nll(theta + ei + ej)
                    - nll(theta + ei - ej)
                    - nll(theta - ei + ej)
                    + nll(theta - ei - ej)
                ) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
        return float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


def threshold_stability(
    x: np.ndarray, quantiles: "tuple[float, ...]" = (0.80, 0.85, 0.90, 0.95, 0.975)
) -> pd.DataFrame:
    """Fit the GPD at several thresholds and report the parameters.

    A tail that really is GPD gives a shape parameter that is stable across
    thresholds, up to sampling error.  If ``xi`` drifts monotonically, the
    fit is picking up the body of the distribution rather than the tail, and
    the extrapolated quantiles should not be trusted.
    """
    rows = []
    for q in quantiles:
        try:
            fit = fit_gpd(x, quantile=q)
        except ValueError:
            continue
        row = fit.to_dict()
        row["quantile"] = q
        rows.append(row)
    if not rows:
        raise ValueError("no threshold gave enough exceedances")
    cols = ["quantile"] + [c for c in rows[0] if c != "quantile"]
    return pd.DataFrame(rows)[cols]


def tail_comparison(
    x: np.ndarray,
    cutoffs: "tuple[float, ...]" = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
    quantile: float = 0.90,
    two_sided: bool = True,
) -> pd.DataFrame:
    """Exceedance probability at each cutoff under three models.

    The three columns are the empirical frequency, the fitted GPD, and the
    Gaussian.  Past the largest observed statistic only the last two are
    defined, and the gap between them is what the multiplicity correction is
    sensitive to.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    vals = np.abs(x) if two_sided else x
    fit = fit_gpd(vals, quantile=quantile)
    rows = []
    for c in cutoffs:
        emp = float(np.mean(vals > c))
        gpd = fit.tail_probability(c) if c >= fit.threshold else np.nan
        gauss = float(2.0 * stats.norm.sf(c)) if two_sided else float(stats.norm.sf(c))
        rows.append(
            {
                "cutoff": c,
                "empirical": emp,
                "gpd": gpd,
                "gaussian": gauss,
                "gpd_over_gaussian": gpd / gauss if gauss > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
