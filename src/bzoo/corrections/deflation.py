"""Deflation: what the best of N trials looks like when nothing works.

The construction is Bailey and Lopez de Prado (2014), Journal of Portfolio
Management 40(5), 94-107, written there for Sharpe ratios.  Two pieces:

1. the expected maximum of N trial statistics under the null, which is an
   extreme-value approximation and needs only the cross-trial standard
   deviation ``sigma`` and the trial count ``N``;
2. a p-value for the selected statistic that treats that expected maximum
   as the benchmark to beat.

The transfer to machine learning replaces the Sharpe ratio with the metric
improvement ``Delta_k = theta(M_k) - theta(B*)``.  Everything about the
arithmetic is unchanged.  What is *not* transferred is the numerical value
of ``sigma``, which is domain specific and has to be estimated from a null
population in the domain at hand; see :mod:`bzoo.null`.

Note on which N to use.  The formulas below assume the N trials are
independent.  When they are not - and in both of our domains they are
strongly dependent - passing the raw trial count overstates the correction.
Pass the effective count from :func:`bzoo.null.dependence.effective_n_sidak`
instead, and report both.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def expected_max_normal(sigma: float, n_trials: float) -> float:
    """Expected maximum of ``N`` iid ``Normal(0, sigma^2)`` draws.

    Uses the Gumbel approximation

    .. math::
        E[\\max_{n \\le N} X_n] \\approx \\sigma
        \\left[(1-\\gamma)\\Phi^{-1}(1 - 1/N)
              + \\gamma\\,\\Phi^{-1}(1 - 1/(Ne))\\right],

    with ``gamma`` the Euler-Mascheroni constant.  This is the expression in
    Bailey and Lopez de Prado (2014), Section 3, derived from the
    Fisher-Tippett limit.

    Accuracy, measured against :func:`expected_max_normal_exact`: the
    approximation is high by 2.6 percent at ``N = 5``, 2.3 percent at
    ``N = 10``, 0.9 percent at ``N = 100``, 0.4 percent at ``N = 1000`` and
    0.1 percent at ``N = 10^6``.  It is *low* by 7.9 percent at ``N = 2``.
    So it is fine for the trial counts this project reports on, and it should
    not be used for a handful of trials; use the exact version there.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    n = float(n_trials)
    if n < 2:
        raise ValueError("need at least two trials")
    q1 = stats.norm.ppf(1.0 - 1.0 / n)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2))


def expected_max_normal_exact(sigma: float, n_trials: float) -> float:
    """Exact ``E[max]`` of ``N`` iid ``Normal(0, sigma^2)`` draws.

    Numerical integration of ``N x phi(x) Phi(x)^(N-1)``.  Used to state how
    good the Gumbel approximation is rather than assuming it, and as the
    reference value in the tests.
    """
    from scipy.integrate import quad

    n = float(n_trials)
    if n < 1:
        raise ValueError("need at least one trial")

    def integrand(x: float) -> float:
        return n * x * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1.0)

    val, _ = quad(integrand, -12.0, 12.0, limit=400)
    return float(sigma * val)


def expected_max_monte_carlo(
    sigma: float, n_trials: int, n_sim: int = 20000, seed: int = 0
) -> "tuple[float, float]":
    """Simulated ``E[max]`` and its standard error, an independent check."""
    rng = np.random.default_rng(seed)
    maxima = rng.normal(0.0, sigma, size=(n_sim, int(n_trials))).max(axis=1)
    return float(maxima.mean()), float(maxima.std(ddof=1) / np.sqrt(n_sim))


def deflated_threshold(sigma: float, n_trials: float, alpha: float = 0.05) -> float:
    """Critical value for the maximum of ``N`` iid ``Normal(0, sigma^2)``.

    Exact rather than asymptotic: ``P(max < c) = Phi(c/sigma)^N = 1 - alpha``
    gives ``c = sigma * Phi^{-1}((1-alpha)^(1/N))``.  We prefer this to the
    Gumbel approximation when a threshold is what we need, and keep
    :func:`expected_max_normal` for comparability with the published
    deflated Sharpe ratio.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = float(n_trials)
    return float(sigma * stats.norm.ppf((1.0 - alpha) ** (1.0 / n)))


def deflated_pvalue(observed: float, sigma: float, n_trials: float) -> float:
    """P(best of ``N`` trials >= ``observed``) under the iid normal null.

    ``1 - Phi(observed/sigma)**N``.  Computed through the survival function
    so that it does not underflow to zero when ``observed/sigma`` is large.
    """
    n = float(n_trials)
    if sigma <= 0:
        return 0.0 if observed > 0 else 1.0
    z = observed / sigma
    log_cdf = stats.norm.logcdf(z)
    return float(-np.expm1(n * log_cdf))


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe ratio of Bailey and Lopez de Prado.

    ``PSR(SR*) = Phi( (SR_hat - SR*) sqrt(T-1) / sqrt(1 - g3 SR_hat +
    (g4-1)/4 SR_hat^2) )``, where ``g3`` is skewness and ``g4`` is
    kurtosis on the raw (not excess) scale, so ``g4 = 3`` for a normal.

    ``sr_hat`` must be on the same time scale as ``n_obs``: if ``n_obs``
    counts months, pass the monthly Sharpe ratio, not the annualised one.
    Getting this wrong is the most common error with these formulas.
    """
    if n_obs < 3:
        raise ValueError("need at least three observations")
    denom_sq = 1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    sr_trials_sd: float,
    n_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> "dict[str, float]":
    """Deflated Sharpe ratio: PSR against the expected best of N trials.

    Returns the benchmark ``sr_star``, the deflated probability ``dsr``, and
    the undeflated ``psr`` so the effect of the correction is visible.
    """
    sr_star = expected_max_normal(sr_trials_sd, n_trials)
    return {
        "sr_hat": float(sr_hat),
        "sr_star": sr_star,
        "psr": probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurtosis, 0.0),
        "dsr": probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurtosis, sr_star),
        "n_trials": float(n_trials),
        "sr_trials_sd": float(sr_trials_sd),
    }


def deflated_improvement(
    delta_obs: float,
    sigma_delta: float,
    n_trials: float,
    alpha: float = 0.05,
) -> "dict[str, float]":
    """The machine learning version: deflate a metric improvement.

    Parameters
    ----------
    delta_obs:
        Reported improvement over the tuned baseline, on the metric scale
        (for example 0.006 for a 0.6 accuracy point gain).
    sigma_delta:
        Standard deviation of ``Delta`` across the constructed null
        population.  Estimated, never assumed.
    n_trials:
        Number of trials the field is credited with.  Report a range.
    """
    return {
        "delta_obs": float(delta_obs),
        "sigma_delta": float(sigma_delta),
        "n_trials": float(n_trials),
        "expected_max": expected_max_normal(sigma_delta, n_trials),
        "threshold": deflated_threshold(sigma_delta, n_trials, alpha),
        "p_value": deflated_pvalue(delta_obs, sigma_delta, n_trials),
        "survives": bool(delta_obs > deflated_threshold(sigma_delta, n_trials, alpha)),
    }


def implied_trials(observed: float, sigma: float, alpha: float = 0.05) -> float:
    """How many trials would make ``observed`` exactly borderline at ``alpha``.

    Solves ``1 - Phi(observed/sigma)^N = alpha`` for ``N``.  A useful way to
    report a result without picking N: "this improvement stops being
    significant once the field is credited with more than N trials".
    """
    if sigma <= 0:
        return float("inf")
    log_cdf = stats.norm.logcdf(observed / sigma)
    if log_cdf >= 0.0:
        return float("inf")
    return float(np.log1p(-alpha) / log_cdf)
