"""Multiplicity corrections with a common interface.

See :mod:`bzoo.corrections.base` for the shared return type and
``README.md`` for the table of methods and citations.
"""

from .base import CorrectionResult
from .bootstrap_tests import bootstrap_centred_matrix, hansen_spa, white_reality_check
from .deflation import (
    deflated_improvement,
    deflated_pvalue,
    deflated_sharpe_ratio,
    deflated_threshold,
    expected_max_normal,
    implied_trials,
    probabilistic_sharpe_ratio,
)
from .fdr import benjamini_hochberg, benjamini_yekutieli, estimate_pi0, storey_qvalues
from .fwer import (
    bonferroni,
    holm,
    romano_wolf,
    sidak,
    westfall_young_maxt,
    westfall_young_minp,
)
from .haircuts import haircuts as sharpe_haircuts
from .haircuts import harvey_liu_zhu, threshold_grid

__all__ = [
    "CorrectionResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni",
    "bootstrap_centred_matrix",
    "deflated_improvement",
    "deflated_pvalue",
    "deflated_sharpe_ratio",
    "deflated_threshold",
    "estimate_pi0",
    "expected_max_normal",
    "sharpe_haircuts",
    "hansen_spa",
    "harvey_liu_zhu",
    "holm",
    "implied_trials",
    "probabilistic_sharpe_ratio",
    "romano_wolf",
    "sidak",
    "storey_qvalues",
    "threshold_grid",
    "westfall_young_maxt",
    "westfall_young_minp",
    "white_reality_check",
]
