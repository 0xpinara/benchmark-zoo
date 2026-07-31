"""Instance-level bootstrap for machine learning metrics.

The finance testbed resamples time blocks.  Machine learning metrics are
averages over test instances, so the sampling unit is the instance.  The
only thing that needs care is that all models must be evaluated on the
*same* resampled instance set within a replicate.

Why that matters: the corrections in :mod:`bzoo.corrections` need the joint
null distribution of the model statistics.  Two models that agree on 95% of
the test set have highly correlated metrics, and the maximum over a
correlated family is far less extreme than the maximum over an independent
one.  Resampling each model separately makes the columns independent by
construction, so the estimated maximum is too large and the corrections
come out too conservative.  We provide the wrong version explicitly, as
:func:`independent_instance_indices`, so that the robustness section can
measure the size of the mistake instead of asserting it.
"""

from __future__ import annotations

import numpy as np


def joint_instance_indices(
    n_instances: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n_instances)`` indices, one draw per replicate.

    The same row is used for every model, which is the point.
    """
    if n_instances < 2:
        raise ValueError("need at least two instances")
    return rng.integers(0, n_instances, size=(n_boot, n_instances), dtype=np.int64)


def independent_instance_indices(
    n_instances: int, n_boot: int, n_models: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n_models, n_instances)`` indices, a separate draw per model.

    This is the implementation error the robustness section quantifies.  It is
    not used anywhere in the main results.

    It materialises the whole three-dimensional index array, which is
    ``n_boot * n_models * n_instances * 8`` bytes: fine for a 1,000-node test
    set and 95 GB for ogbn-arxiv with 4,000 replicates and 61 models.  Use
    :func:`bootstrap_metric_matrix_independent` on anything large; that draws
    one model's indices at a time and never holds more than one.
    """
    return rng.integers(
        0, n_instances, size=(n_boot, n_models, n_instances), dtype=np.int64
    )


def bootstrap_metric_matrix(
    scores: np.ndarray,
    indices: np.ndarray,
    max_bytes: int = 2_000_000_000,
) -> np.ndarray:
    """Average per-instance scores over bootstrap index draws.

    Parameters
    ----------
    scores:
        ``(n_instances, n_models)`` per-instance metric contributions.  For
        accuracy this is a 0/1 correctness matrix; for NDCG it is the
        per-query gain; for anything that is a plain mean over instances it
        is that quantity.
    indices:
        ``(B, n_instances)`` joint draws, or ``(B, n_models, n_instances)``
        independent draws.
    max_bytes:
        Ceiling on the size of the intermediate gather.  The obvious one-line
        implementation, ``scores[indices].mean(axis=1)``, materialises a
        ``(B, n_instances, n_models)`` array.  That is fine for a 1,000-node
        test set and is 95 terabytes for ogbn-arxiv with 48,603 test nodes,
        4,000 replicates and 61 models, so the replicates are processed in
        batches sized to this limit.

    Returns
    -------
    ``(B, n_models)`` matrix of resampled metrics.
    """
    scores = np.asarray(scores, dtype=np.float32)
    indices = np.asarray(indices)
    if scores.ndim != 2:
        raise ValueError("scores must be (n_instances, n_models)")
    n_inst, n_models = scores.shape

    if indices.ndim == 2:
        b = indices.shape[0]
        per_replicate = n_inst * n_models * scores.itemsize
        batch = max(1, int(max_bytes // max(1, per_replicate)))
        out = np.empty((b, n_models), dtype=np.float64)
        for start in range(0, b, batch):
            stop = min(b, start + batch)
            out[start:stop] = scores[indices[start:stop]].mean(axis=1)
        return out
    if indices.ndim == 3:
        if indices.shape[1] != n_models:
            raise ValueError("independent indices must have n_models in axis 1")
        b = indices.shape[0]
        per_replicate_col = n_inst * scores.itemsize
        batch = max(1, int(max_bytes // max(1, per_replicate_col)))
        out = np.empty((b, n_models), dtype=np.float64)
        for m in range(n_models):
            col = scores[:, m]
            for start in range(0, b, batch):
                stop = min(b, start + batch)
                out[start:stop, m] = col[indices[start:stop, m, :]].mean(axis=1)
        return out
    raise ValueError("indices must be 2-D (joint) or 3-D (independent)")


def bootstrap_metric_matrix_independent(
    scores: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    max_bytes: int = 2_000_000_000,
) -> np.ndarray:
    """``(n_boot, n_models)`` metrics under the *incorrect* per-model scheme.

    Equivalent to :func:`bootstrap_metric_matrix` with
    :func:`independent_instance_indices`, but the indices for one model are
    drawn, used and discarded before the next model's are drawn, so peak memory
    is one model's worth rather than all of them at once.  That is the
    difference between 1.5 GB and 95 GB on the large graph.

    This exists only so the robustness section can measure the size of the
    mistake.  Nothing in the main results calls it.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be (n_instances, n_models)")
    n_inst, n_models = scores.shape
    batch = max(1, int(max_bytes // max(1, n_inst * 8)))
    out = np.empty((n_boot, n_models), dtype=np.float64)
    for m in range(n_models):
        col = scores[:, m]
        for start in range(0, n_boot, batch):
            stop = min(n_boot, start + batch)
            idx = rng.integers(
                0, n_inst, size=(stop - start, n_inst), dtype=np.int64
            )
            out[start:stop, m] = col[idx].mean(axis=1)
    return out


def paired_differences(scores: np.ndarray, baseline_col: int) -> np.ndarray:
    """Per-instance differences against one column, keeping the pairing.

    Returns ``(n_instances, n_models - 1)``.  Pairing is what gives the
    difference a much smaller variance than the two metrics separately, and
    it is the object the Reality Check and SPA tests operate on.
    """
    scores = np.asarray(scores, dtype=float)
    keep = [j for j in range(scores.shape[1]) if j != baseline_col]
    return scores[:, keep] - scores[:, [baseline_col]]
