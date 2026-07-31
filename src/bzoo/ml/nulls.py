"""Constructing a null population for a machine learning benchmark.

In the finance testbed the null population is handed to us: a strategy built
from ticker letters cannot work.  Here there is no analogue, so the population
has to be built, and the finance testbed's job is to establish that this style
of construction recovers the truth when the truth is known.

Three constructions, from the same sweep, reported side by side because they
answer different questions.

``seed``
    Fix architecture and configuration, vary only the seed.  This is the
    irreducible noise floor: the spread of ``Delta`` here is what you get for
    doing nothing at all.
``config``
    Fix architecture, vary the configuration.  This is search without
    innovation.  It is the relevant null for collective search, because a field
    that tries a thousand configurations of the same idea and reports the best
    one is drawing from exactly this distribution.
``ablation``
    Architectures with one claimed component removed, at matched parameter
    count and budget.  This is the null for "the component is what did it".

For each we need ``sigma_Delta``, the standard deviation of
``Delta = theta(M) - theta(B*)`` where ``B*`` is the best tuned baseline chosen
on validation accuracy.  Two estimates are produced and both are reported:

``across-run``
    The standard deviation of ``Delta`` across the runs in the null population.
    Includes genuine differences in model quality as well as test-set noise.
``bootstrap``
    The standard deviation of ``Delta`` for a single fixed pair of models under
    the joint instance bootstrap.  This is test-set noise only.

The across-run number is the one to deflate with, because a field selecting the
best of N trials is selecting over both sources.  The bootstrap number is
reported alongside because it is a lower bound and because the gap between the
two says how much of the spread is real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..resample.instance import (
    bootstrap_metric_matrix,
    bootstrap_metric_matrix_independent,
    joint_instance_indices,
)

BASELINE_ARCHS = ("gcn", "sage", "mlp")
ABLATION_ARCHS = ("gcn_noprop", "gcn_unnorm", "sage_noneigh")


@dataclass
class NullPopulation:
    name: str
    delta: np.ndarray  # test-accuracy improvement over the tuned baseline
    run_index: np.ndarray  # rows of the correctness matrix these came from
    baseline_run: int
    baseline_test_accuracy: float

    def summary(self) -> Dict[str, float]:
        d = self.delta[np.isfinite(self.delta)]
        nan = float("nan")
        if d.size == 0:
            return {
                "population": self.name,
                "n": 0,
                "mean_delta": nan,
                "sd_delta": nan,
                "max_delta": nan,
                "q95_delta": nan,
                "share_positive": nan,
                "baseline_test_accuracy": self.baseline_test_accuracy,
            }
        return {
            "population": self.name,
            "n": int(d.size),
            "mean_delta": float(d.mean()),
            "sd_delta": float(d.std(ddof=1)) if d.size > 1 else nan,
            "max_delta": float(d.max()),
            "q95_delta": float(np.quantile(d, 0.95)),
            "share_positive": float(np.mean(d > 0)),
            "baseline_test_accuracy": self.baseline_test_accuracy,
        }


def select_baseline(runs: pd.DataFrame, arch: str = "gcn") -> "tuple[int, float]":
    """Pick the tuned baseline by validation accuracy, never by test accuracy.

    Returns the row index into ``runs`` and its test accuracy.  Selecting on
    validation is the whole discipline: a baseline chosen on test accuracy would
    already have absorbed part of the multiplicity we are trying to measure, and
    every ``Delta`` below would be biased downwards.
    """
    sub = runs.loc[runs["arch"] == arch]
    if sub.empty:
        raise ValueError(f"no runs for architecture {arch!r}")
    row = sub.loc[sub["valid_accuracy"].idxmax()]
    return int(row.name), float(row["test_accuracy"])


def build_null_populations(
    runs: pd.DataFrame,
    baseline_arch: str = "gcn",
) -> Dict[str, NullPopulation]:
    """The three null populations, all measured against one tuned baseline."""
    base_idx, base_test = select_baseline(runs, baseline_arch)
    base_config = runs.loc[base_idx, "config_id"]
    out: Dict[str, NullPopulation] = {}

    # 1. seed only: same architecture and configuration as the baseline
    seed_rows = runs.index[
        (runs["arch"] == baseline_arch) & (runs["config_id"] == base_config)
    ]
    seed_rows = np.array([i for i in seed_rows if i != base_idx], dtype=int)
    out["seed"] = NullPopulation(
        "seed variation",
        runs.loc[seed_rows, "test_accuracy"].to_numpy() - base_test,
        seed_rows,
        base_idx,
        base_test,
    )

    # 2. random configurations of the baseline architecture
    cfg_rows = runs.index[
        (runs["arch"] == baseline_arch) & (runs["config_id"] != base_config)
    ].to_numpy()
    out["config"] = NullPopulation(
        "random configurations",
        runs.loc[cfg_rows, "test_accuracy"].to_numpy() - base_test,
        cfg_rows,
        base_idx,
        base_test,
    )

    # 3. ablated architectures
    abl_rows = runs.index[runs["arch"].isin(ABLATION_ARCHS)].to_numpy()
    out["ablation"] = NullPopulation(
        "ablated architectures",
        runs.loc[abl_rows, "test_accuracy"].to_numpy() - base_test,
        abl_rows,
        base_idx,
        base_test,
    )

    # 4. pooled: configurations and ablations together, the population a
    #    collective search over "plausible things to try" would draw from
    pooled = np.concatenate([cfg_rows, abl_rows])
    out["pooled"] = NullPopulation(
        "configurations and ablations",
        runs.loc[pooled, "test_accuracy"].to_numpy() - base_test,
        pooled,
        base_idx,
        base_test,
    )
    return out


def screened_population(
    runs: pd.DataFrame,
    correct: np.ndarray,
    pop: NullPopulation,
    n_boot: int = 2000,
    seed: int = 0,
    min_keep: int = 10,
    min_pass: int = 5,
) -> "tuple[NullPopulation, dict]":
    """Apply :func:`screen_not_worse` to a null population.

    The per-trial standard deviation of ``Delta`` comes from the joint instance
    bootstrap, so the screening rule is scaled to the noise of the metric on this
    particular test set rather than to a constant.

    Two separate things are reported and they must not be confused.

    ``relaxed``
        Fewer than ``min_keep`` trials passed the rule, so the ``min_keep``
        least-negative were kept instead.  Harmless when the rule was close to
        keeping that many anyway.
    ``estimable``
        At least ``min_pass`` trials passed the rule on their own merits.  If
        not, the pool contains no candidate that is plausibly as good as the
        baseline, and there is no null population in it to measure.  Relaxing
        the rule then does not rescue anything: it keeps the least-bad
        candidates, which are still genuinely worse, and a spread computed over
        them is a spread of real differences, not of noise.

    This is a requirement on the sweep, not a defect of the rule. To calibrate a
    threshold you need a search space that contains near-baseline candidates, and
    a sweep too small to produce one cannot calibrate anything. The caller should
    check ``estimable`` and decline to report a threshold when it is false.
    """
    scores = correct[[pop.baseline_run] + list(pop.run_index)].T.astype(float)
    rng = np.random.default_rng(seed)
    idx = joint_instance_indices(scores.shape[0], n_boot, rng)
    boot = bootstrap_metric_matrix(scores, idx)
    per_trial_sd = (boot[:, 1:] - boot[:, [0]]).std(axis=0, ddof=1)

    keep = screen_not_worse(pop.delta, per_trial_sd, len(pop.delta))
    n_rule = int(keep.sum())
    relaxed = False
    if n_rule < min_keep:
        relaxed = True
        take = min(min_keep, pop.delta.size)
        order = np.argsort(-pop.delta)[:take]
        keep = np.zeros_like(keep)
        keep[order] = True

    kept = NullPopulation(
        pop.name + (", screened (relaxed)" if relaxed else ", screened"),
        pop.delta[keep],
        pop.run_index[keep],
        pop.baseline_run,
        pop.baseline_test_accuracy,
    )
    # How far the best non-baseline candidate is from the baseline, in units of
    # the metric's own noise.  This is the number that says whether the sweep
    # produced a null population at all.
    best_delta = float(pop.delta.max())
    noise = float(np.median(per_trial_sd))
    info = {
        "n_before": int(pop.delta.size),
        "n_after": int(keep.sum()),
        "n_passing_rule": n_rule,
        "relaxed": bool(relaxed),
        "min_keep": int(min_keep),
        "min_pass": int(min_pass),
        "estimable": bool(n_rule >= min_pass),
        "share_kept": float(keep.mean()),
        "median_per_trial_sd": noise,
        "screen_rate": float(np.sqrt(2.0 * np.log(np.log(max(3, pop.delta.size))))),
        "worst_delta_kept": float(pop.delta[keep].min()),
        "best_delta_in_pool": best_delta,
        "baseline_lead_in_noise_units": (
            float(-best_delta / noise) if noise > 0 else float("inf")
        ),
    }
    return kept, info


def bootstrap_sigma_delta(
    correct: np.ndarray,
    baseline_run: int,
    candidate_runs: Sequence[int],
    n_boot: int = 2000,
    seed: int = 0,
    joint: bool = True,
) -> Dict[str, float]:
    """Test-set-noise standard deviation of ``Delta``, by instance bootstrap.

    ``correct`` is the ``(n_runs, n_instances)`` boolean matrix.  The same
    resampled instance set is used for the baseline and for every candidate
    within a replicate, which is what preserves the correlation between two
    models that mostly agree.  ``joint=False`` gives the incorrect
    per-model version, for the robustness comparison.
    """
    scores = correct[[baseline_run] + list(candidate_runs)].T.astype(np.float32)
    rng = np.random.default_rng(seed)
    n_inst, n_models = scores.shape
    if joint:
        idx = joint_instance_indices(n_inst, n_boot, rng)
        boot = bootstrap_metric_matrix(scores, idx)
    else:
        # Per-model draws, generated one model at a time: the full index array
        # would be 95 GB on the large graph.
        boot = bootstrap_metric_matrix_independent(scores, n_boot, rng)
    delta = boot[:, 1:] - boot[:, [0]]
    per_candidate_sd = delta.std(axis=0, ddof=1)
    return {
        "n_candidates": int(len(candidate_runs)),
        "n_instances": int(n_inst),
        "n_replicates": int(n_boot),
        "median_sd_delta": float(np.median(per_candidate_sd)),
        "mean_sd_delta": float(per_candidate_sd.mean()),
        "sd_delta_of_best": float(
            per_candidate_sd[int(np.argmax(scores[:, 1:].mean(axis=0)))]
        ),
        "mean_pairwise_correlation": float(
            _mean_offdiag_corr(boot) if n_models > 2 else np.nan
        ),
        "joint": bool(joint),
    }


def _mean_offdiag_corr(boot: np.ndarray) -> float:
    c = np.corrcoef(boot.T)
    k = c.shape[0]
    return float(c[~np.eye(k, dtype=bool)].mean())


def seed_noise_floor(runs: pd.DataFrame) -> Dict[str, float]:
    """Pooled within-configuration standard deviation of test accuracy.

    The irreducible noise floor: how much the metric moves when nothing changes
    but the random seed.  Pooling over every architecture-configuration cell uses
    all the seed replicates in the sweep, which is far more information than the
    two or three replicates of the baseline cell alone, and it is a scalar rather
    than a population because it is not a distribution of improvements.
    """
    g = runs.groupby(["arch", "config_id"])["test_accuracy"]
    within = g.transform("mean")
    resid = runs["test_accuracy"] - within
    n_cells = int(g.ngroups)
    dof = int(len(runs) - n_cells)
    if dof <= 0:
        return {"n_cells": n_cells, "dof": 0, "sd": float("nan")}
    sd = float(np.sqrt((resid ** 2).sum() / dof))
    per_cell = g.std(ddof=1).dropna()
    return {
        "n_cells": n_cells,
        "dof": dof,
        "sd": sd,
        "median_cell_sd": float(per_cell.median()) if len(per_cell) else float("nan"),
        "max_cell_sd": float(per_cell.max()) if len(per_cell) else float("nan"),
    }


def screen_not_worse(
    delta: np.ndarray,
    sigma_per_trial: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """Keep the trials that are not detectably worse than the baseline.

    Why this is needed, and why the plan did not anticipate it.  In finance the
    candidate pool is a set of long-short strategies whose expected returns are
    all small; a mined strategy is rarely *much* worse than zero.  In machine
    learning the pool contains models that are genuinely far worse - an ablated
    graph network on Cora loses twenty-five accuracy points, not one - and those
    trials have true ``Delta`` around ``-0.25``, not ``0``.  Including them in a
    null population inflates ``sigma_Delta`` by an order of magnitude while
    contributing nothing at all to its upper tail, so the deflated threshold
    comes out at fifty accuracy points and the correction is vacuous.

    This is the same failure White's (2000) Reality Check has when obviously
    poor models are added to the candidate set, and the fix is the same one
    Hansen (2005) uses: drop a candidate from the null distribution once its
    shortfall exceeds the law-of-the-iterated-logarithm rate,

        ``Delta_k < - sigma_k sqrt(2 log log n)``,

    where ``sigma_k`` is the standard deviation of ``Delta_k`` itself.  The
    difference between the two domains is one of degree, and it is large enough
    that in finance the recentring is a refinement and in machine learning it is
    the difference between a usable threshold and a meaningless one.
    """
    delta = np.asarray(delta, dtype=float)
    sigma_per_trial = np.asarray(sigma_per_trial, dtype=float)
    n = max(int(n_trials), 3)
    rate = np.sqrt(2.0 * np.log(np.log(n)))
    return delta > -rate * sigma_per_trial


def selection_null(
    runs: pd.DataFrame,
    baseline_run: int,
    candidate_rows: np.ndarray,
    n_search: Sequence[int] = (1, 3, 10, 30, 100),
    n_draws: int = 20000,
    seed: int = 0,
) -> Dict[str, object]:
    """The distribution of the improvement a researcher would report.

    Rather than model the maximum of ``N`` draws, simulate the selection: draw
    ``m`` candidates from the null pool, keep the one with the best
    *validation* accuracy, and record its *test* improvement over the baseline.
    Repeating that gives the distribution of "what someone who tried ``m``
    things and reported the best would see", with no distributional assumption
    and with the validation/test split respected.

    This is the non-parametric counterpart of the closed-form
    ``E[max Delta]``, and it handles the bad-candidate problem automatically:
    a model twenty-five points worse never wins the validation comparison, so
    it never enters the reported distribution, and no screening rule is needed.
    It is the version we prefer, and the closed form is reported next to it so
    that the reader can see where the two agree.
    """
    rng = np.random.default_rng(seed)
    val = runs.loc[candidate_rows, "valid_accuracy"].to_numpy()
    test = runs.loc[candidate_rows, "test_accuracy"].to_numpy()
    base_test = float(runs.loc[baseline_run, "test_accuracy"])
    pool = val.size
    out = []
    for m in n_search:
        if m > pool:
            m_eff = pool
        else:
            m_eff = m
        idx = rng.integers(0, pool, size=(n_draws, m_eff))
        winners = idx[np.arange(n_draws), np.argmax(val[idx], axis=1)]
        d = test[winners] - base_test
        out.append(
            {
                "n_search": int(m),
                "n_search_effective": int(m_eff),
                "mean_delta": float(d.mean()),
                "sd_delta": float(d.std(ddof=1)),
                "q95_delta": float(np.quantile(d, 0.95)),
                "q99_delta": float(np.quantile(d, 0.99)),
                "max_delta": float(d.max()),
                "share_positive": float(np.mean(d > 0)),
            }
        )
    return {"pool_size": int(pool), "baseline_test_accuracy": base_test, "grid": out}


def headroom(
    best_accuracy: float,
    ceiling: float = 1.0,
) -> float:
    """Remaining distance to the ceiling, the saturation measure.

    ``ceiling`` is 1.0 for accuracy by default.  A real benchmark has a lower
    effective ceiling than 1.0 because of label noise and irreducible ambiguity,
    which is unknown; using 1.0 makes the headroom an upper bound and therefore
    makes the saturation claim conservative.
    """
    return float(ceiling - best_accuracy)
