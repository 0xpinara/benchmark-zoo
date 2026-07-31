"""Tests for the two resampling schemes.

The claim these tests are here to protect is the one in the plan's warning:
joint resampling preserves cross-model dependence and per-model resampling
destroys it.  That is asserted as an implementation rule elsewhere; here it
is measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from bzoo.resample import instance, permutation, stationary


def test_stationary_indices_have_the_right_shape_and_range():
    rng = np.random.default_rng(0)
    idx = stationary.stationary_bootstrap_indices(100, 50, 10.0, rng)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_stationary_bootstrap_is_uniform_over_observations():
    """Wrapping is what makes the scheme stationary: every observation must
    be drawn with probability 1/n, including the ones at the ends."""
    rng = np.random.default_rng(1)
    n = 40
    idx = stationary.stationary_bootstrap_indices(n, 4000, 5.0, rng)
    counts = np.bincount(idx.ravel(), minlength=n) / idx.size
    assert np.allclose(counts, 1.0 / n, atol=0.004)


def test_longer_blocks_preserve_more_autocorrelation():
    """A block length of 1 gives an iid bootstrap, which destroys serial
    correlation; a long block keeps most of it."""
    rng = np.random.default_rng(2)
    n = 2000
    e = rng.standard_normal(n + 200)
    x = np.zeros(n)
    prev = 0.0
    for t in range(n):  # AR(1) with rho = 0.6
        prev = 0.6 * prev + e[t]
        x[t] = prev

    def mean_ac1(block_length: float) -> float:
        idx = stationary.stationary_bootstrap_indices(n, 200, block_length, rng)
        s = x[idx]
        s = s - s.mean(axis=1, keepdims=True)
        num = (s[:, 1:] * s[:, :-1]).sum(axis=1)
        den = (s ** 2).sum(axis=1)
        return float(np.mean(num / den))

    assert mean_ac1(1.0) < 0.1
    assert mean_ac1(50.0) > 0.4


def test_optimal_block_length_grows_with_persistence():
    rng = np.random.default_rng(3)
    n = 3000

    def ar1(rho: float) -> np.ndarray:
        e = rng.standard_normal(n)
        x = np.zeros(n)
        prev = 0.0
        for t in range(n):
            prev = rho * prev + e[t]
            x[t] = prev
        return x

    b_white = stationary.optimal_block_length(ar1(0.0))
    b_persistent = stationary.optimal_block_length(ar1(0.8))
    assert b_persistent > b_white
    assert 1.0 <= b_white <= n / 3
    assert 1.0 <= b_persistent <= n / 3


def _correlated_scores(rng, n_inst=2000, n_models=5, rho=0.9):
    """Per-instance 0/1 correctness for models that mostly agree."""
    latent = rng.standard_normal(n_inst)
    out = np.empty((n_inst, n_models))
    for m in range(n_models):
        z = np.sqrt(rho) * latent + np.sqrt(1 - rho) * rng.standard_normal(n_inst)
        out[:, m] = (z > -0.7).astype(float)
    return out


def test_joint_resampling_preserves_cross_model_correlation():
    rng = np.random.default_rng(4)
    scores = _correlated_scores(rng)
    joint = instance.joint_instance_indices(scores.shape[0], 500, rng)
    boot_joint = instance.bootstrap_metric_matrix(scores, joint)
    c_joint = np.corrcoef(boot_joint.T)
    off = c_joint[~np.eye(5, dtype=bool)]
    assert off.mean() > 0.6


def test_independent_resampling_destroys_cross_model_correlation():
    rng = np.random.default_rng(5)
    scores = _correlated_scores(rng)
    indep = instance.independent_instance_indices(scores.shape[0], 500, 5, rng)
    boot_indep = instance.bootstrap_metric_matrix(scores, indep)
    c_indep = np.corrcoef(boot_indep.T)
    off = c_indep[~np.eye(5, dtype=bool)]
    assert abs(off.mean()) < 0.15


def test_independent_resampling_inflates_the_maximum():
    """The consequence: the null maximum is larger under the wrong scheme, so
    the correction is too conservative."""
    rng = np.random.default_rng(6)
    scores = _correlated_scores(rng)
    joint = instance.bootstrap_metric_matrix(
        scores, instance.joint_instance_indices(scores.shape[0], 2000, rng)
    )
    indep = instance.bootstrap_metric_matrix(
        scores, instance.independent_instance_indices(scores.shape[0], 2000, 5, rng)
    )
    centre = scores.mean(axis=0)
    max_joint = (joint - centre).max(axis=1)
    max_indep = (indep - centre).max(axis=1)
    assert np.quantile(max_indep, 0.95) > np.quantile(max_joint, 0.95)


def test_bootstrap_metric_matrix_matches_a_direct_computation():
    rng = np.random.default_rng(7)
    scores = rng.random((50, 3))
    idx = instance.joint_instance_indices(50, 10, rng)
    fast = instance.bootstrap_metric_matrix(scores, idx)
    slow = np.array([[scores[idx[b], m].mean() for m in range(3)] for b in range(10)])
    assert np.allclose(fast, slow)


def test_paired_differences_drops_the_baseline_column():
    scores = np.arange(12, dtype=float).reshape(4, 3)
    d = instance.paired_differences(scores, baseline_col=1)
    assert d.shape == (4, 2)
    assert np.allclose(d[:, 0], scores[:, 0] - scores[:, 1])
    assert np.allclose(d[:, 1], scores[:, 2] - scores[:, 1])


def test_sign_flip_replicates_are_centred():
    rng = np.random.default_rng(8)
    diffs = rng.standard_normal((500, 4)) + 0.3  # a real, common effect
    rep = permutation.sign_flip_replicates(diffs, 4000, rng)
    # Under the sign-flip null every column has mean zero, whatever the
    # observed effect was.
    assert np.all(np.abs(rep.mean(axis=0)) < 0.02)


def test_sign_flip_replicates_inherit_cross_model_dependence():
    """The shared sign vector carries the dependence across columns.  With
    independent columns there is nothing to carry, so the test has to supply
    a common component; with a common component the replicates must be
    correlated, and separate sign vectors per column must not be."""
    rng = np.random.default_rng(80)
    latent = rng.standard_normal((600, 1))
    diffs = 0.9 * latent + 0.44 * rng.standard_normal((600, 4))

    shared = permutation.sign_flip_replicates(diffs, 4000, rng)
    off_shared = np.corrcoef(shared.T)[~np.eye(4, dtype=bool)]
    assert off_shared.mean() > 0.5

    separate = np.empty_like(shared)
    for m in range(4):
        separate[:, [m]] = permutation.sign_flip_replicates(diffs[:, [m]], 4000, rng)
    off_separate = np.corrcoef(separate.T)[~np.eye(4, dtype=bool)]
    assert abs(off_separate.mean()) < 0.1


def test_block_sign_flip_reduces_to_sign_flip_at_block_one():
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    diffs = np.random.default_rng(10).standard_normal((64, 3))
    a = permutation.sign_flip_replicates(diffs, 20, rng1)
    b = permutation.block_sign_flip_replicates(diffs, 20, 1, rng2)
    assert np.allclose(a, b)


def test_studentise_equalises_scales():
    rng = np.random.default_rng(11)
    rep = rng.standard_normal((1000, 3)) * np.array([1.0, 5.0, 20.0])
    obs = np.array([1.0, 5.0, 20.0])
    rep_s, obs_s = permutation.studentise(rep, obs)
    assert np.allclose(rep_s.std(axis=0, ddof=1), 1.0, atol=0.05)
    assert np.allclose(obs_s, 1.0, atol=0.1)


def test_stationary_bootstrap_rejects_bad_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        stationary.stationary_bootstrap_indices(1, 10, 5.0, rng)
    with pytest.raises(ValueError):
        stationary.stationary_bootstrap_indices(10, 10, 0.5, rng)


def test_bootstrap_metric_matrix_batches_without_changing_the_answer():
    """The batching exists to keep a 95 TB intermediate off the heap on the
    large graph; it must not change a single number."""
    rng = np.random.default_rng(20)
    scores = rng.random((200, 7)).astype(np.float32)
    idx = instance.joint_instance_indices(200, 50, rng)
    whole = instance.bootstrap_metric_matrix(scores, idx, max_bytes=10 ** 12)
    batched = instance.bootstrap_metric_matrix(scores, idx, max_bytes=6000)
    assert np.allclose(whole, batched)

    idx3 = instance.independent_instance_indices(200, 50, 7, rng)
    whole3 = instance.bootstrap_metric_matrix(scores, idx3, max_bytes=10 ** 12)
    batched3 = instance.bootstrap_metric_matrix(scores, idx3, max_bytes=1600)
    assert np.allclose(whole3, batched3)


def test_bootstrap_metric_matrix_memory_is_bounded():
    """A gather that would be 60 GB unbatched must complete in a small budget."""
    rng = np.random.default_rng(21)
    scores = (rng.random((50_000, 40)) > 0.3).astype(np.float32)
    idx = instance.joint_instance_indices(50_000, 20, rng)
    out = instance.bootstrap_metric_matrix(scores, idx, max_bytes=50_000_000)
    assert out.shape == (20, 40)
    assert np.all((out > 0.5) & (out < 0.9))


def test_independent_bootstrap_without_materialising_all_indices():
    """The streaming version must match the array version in distribution and
    must not need the memory the array version does."""
    rng_a = np.random.default_rng(30)
    rng_b = np.random.default_rng(30)
    scores = _correlated_scores(np.random.default_rng(31), n_inst=800, n_models=5)
    via_array = instance.bootstrap_metric_matrix(
        scores, instance.independent_instance_indices(800, 3000, 5, rng_a)
    )
    streamed = instance.bootstrap_metric_matrix_independent(scores, 3000, rng_b)
    # Different draws, so compare the distributions rather than the values.
    assert np.allclose(via_array.mean(axis=0), streamed.mean(axis=0), atol=0.002)
    assert np.allclose(
        via_array.std(axis=0, ddof=1), streamed.std(axis=0, ddof=1), rtol=0.15
    )
    # And it destroys the cross-model correlation, which is its whole purpose.
    off = np.corrcoef(streamed.T)[~np.eye(5, dtype=bool)]
    assert abs(off.mean()) < 0.1
