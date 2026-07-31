"""Resampling schemes: time blocks for finance, instances for machine learning."""

from .instance import (
    bootstrap_metric_matrix,
    bootstrap_metric_matrix_independent,
    independent_instance_indices,
    joint_instance_indices,
    paired_differences,
)
from .permutation import (
    block_sign_flip_replicates,
    sign_flip_replicates,
    studentise,
)
from .stationary import optimal_block_length, stationary_bootstrap_indices

__all__ = [
    "block_sign_flip_replicates",
    "bootstrap_metric_matrix",
    "bootstrap_metric_matrix_independent",
    "independent_instance_indices",
    "joint_instance_indices",
    "optimal_block_length",
    "paired_differences",
    "sign_flip_replicates",
    "stationary_bootstrap_indices",
    "studentise",
]
