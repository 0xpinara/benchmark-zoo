"""Common interface for every multiplicity correction in this package.

There are two families of corrections here and they need different inputs:

* p-value corrections (Bonferroni, Holm, BH, BY, Storey) take a vector of
  marginal p-values and know nothing about the dependence between tests;
* resampling corrections (White's Reality Check, Hansen's SPA,
  Romano-Wolf, Westfall-Young) take a matrix of resampled statistics and
  learn the dependence from it.

Both return the same :class:`CorrectionResult` so that the reporting code
does not have to know which family it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class CorrectionResult:
    """Output of a correction.

    Attributes
    ----------
    method:
        Short name of the procedure, used as a table row label.
    n_tests:
        Number of hypotheses actually passed to the procedure.
    alpha:
        Nominal error level that was controlled.
    reject:
        Boolean array, ``True`` where the hypothesis is rejected.
    adjusted_p:
        Adjusted p-values (or q-values for FDR procedures).  ``None`` for
        procedures that do not produce one.
    critical_value:
        Threshold on the *statistic* scale, when the procedure defines a
        single one (Bonferroni, Westfall-Young max-T, deflation).  ``None``
        for step-down procedures, which do not have a single threshold.
    error_rate:
        Which error rate is controlled: ``"FWER"``, ``"FDR"`` or ``"pFDR"``.
    extra:
        Anything else worth reporting (estimated pi0, effective number of
        tests, bootstrap replicate count, ...).
    """

    method: str
    n_tests: int
    alpha: float
    reject: np.ndarray
    adjusted_p: Optional[np.ndarray] = None
    critical_value: Optional[float] = None
    error_rate: str = "FWER"
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def n_reject(self) -> int:
        return int(np.sum(self.reject))

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"CorrectionResult(method={self.method!r}, n_tests={self.n_tests}, "
            f"alpha={self.alpha}, n_reject={self.n_reject}, "
            f"error_rate={self.error_rate!r})"
        )


def check_pvalues(p: np.ndarray) -> np.ndarray:
    """Validate a p-value vector and return it as a 1-D float array."""
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("empty p-value vector")
    if not np.all(np.isfinite(p)):
        raise ValueError("p-values contain NaN or inf")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return alpha


def check_resample_matrix(boot: np.ndarray, obs: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Validate a ``(B, K)`` matrix of resampled statistics against ``K`` observed ones."""
    boot = np.asarray(boot, dtype=float)
    obs = np.asarray(obs, dtype=float).ravel()
    if boot.ndim != 2:
        raise ValueError("resampled statistics must be a 2-D array of shape (B, K)")
    if boot.shape[1] != obs.size:
        raise ValueError(
            f"observed statistics have length {obs.size} but the resample "
            f"matrix has {boot.shape[1]} columns"
        )
    if boot.shape[0] < 10:
        raise ValueError("need at least 10 resample replicates")
    if not np.all(np.isfinite(boot)) or not np.all(np.isfinite(obs)):
        raise ValueError("statistics contain NaN or inf")
    return boot, obs
