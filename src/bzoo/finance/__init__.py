"""Finance testbed: the population whose null is known by construction."""

from .loaders import (
    download_factors,
    download_osap,
    load_mined_names,
    load_mined_returns,
    load_osap_portfolios,
    load_osap_signal_doc,
    mined_return_panel,
    osap_longshort_panel,
)
from .metrics import factor_alphas, summarise_panel, two_sided_pvalues
from .partition import RULES as PARTITION_RULES
from .partition import (
    classify_name,
    partition as partition_signals,
    partition_summary,
)

__all__ = [
    "PARTITION_RULES",
    "classify_name",
    "download_factors",
    "download_osap",
    "factor_alphas",
    "load_mined_names",
    "load_mined_returns",
    "load_osap_portfolios",
    "load_osap_signal_doc",
    "mined_return_panel",
    "osap_longshort_panel",
    "partition_signals",
    "partition_summary",
    "summarise_panel",
    "two_sided_pvalues",
]
