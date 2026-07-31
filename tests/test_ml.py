"""Tests for the machine learning testbed.

The parts that do not need a trained model are tested directly.  The parts that
do are tested on a synthetic run table, so the suite runs in continuous
integration without torch and without downloading a graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bzoo.ml import nulls, tuning


# ----------------------------------------------------------------------
# tuning.py: budget parity


def _records(archs=("gcn", "sage"), configs=("a", "b"), seeds=(0, 1)):
    return [
        {"arch": a, "config_id": c, "seed": s}
        for a in archs
        for c in configs
        for s in seeds
    ]


def test_parity_accepts_a_balanced_sweep():
    info = tuning.check_budget_parity(_records())
    assert info["n_architectures"] == 2
    assert info["trials_per_architecture"] == 4
    assert info["n_configs"] == 2
    assert info["n_seeds"] == 2
    assert info["parity_enforced"] is True


def test_parity_rejects_an_extra_trial_for_one_architecture():
    recs = _records() + [{"arch": "sage", "config_id": "c", "seed": 0}]
    with pytest.raises(tuning.BudgetParityError, match="unequal trial counts"):
        tuning.check_budget_parity(recs)


def test_parity_rejects_a_different_seed_set():
    recs = _records(archs=("gcn",)) + [
        {"arch": "sage", "config_id": c, "seed": s}
        for c in ("a", "b")
        for s in (0, 7)
    ]
    with pytest.raises(tuning.BudgetParityError, match="unequal seed sets"):
        tuning.check_budget_parity(recs)


def test_parity_rejects_a_different_configuration_set():
    recs = _records(archs=("gcn",)) + [
        {"arch": "sage", "config_id": c, "seed": s}
        for c in ("a", "z")
        for s in (0, 1)
    ]
    with pytest.raises(tuning.BudgetParityError, match="same configurations"):
        tuning.check_budget_parity(recs)


def test_parity_rejects_duplicate_runs():
    recs = _records()
    recs[0] = dict(recs[1])
    with pytest.raises(tuning.BudgetParityError, match="duplicate runs"):
        tuning.check_budget_parity(recs)


def test_parity_rejects_an_empty_sweep():
    with pytest.raises(tuning.BudgetParityError):
        tuning.check_budget_parity([])


def test_sampled_configs_are_deterministic_and_complete():
    a = tuning.sample_configs(20, seed=1)
    b = tuning.sample_configs(20, seed=1)
    c = tuning.sample_configs(20, seed=2)
    assert a == b
    assert a != c
    assert len(a) == 20
    assert all(set(cfg) == set(tuning.SEARCH_SPACE) for cfg in a)
    for cfg in a:
        for k, v in cfg.items():
            assert v in tuning.SEARCH_SPACE[k]


def test_sampled_configs_are_distinct():
    cfgs = tuning.sample_configs(40, seed=3)
    ids = {tuning.config_id(c) for c in cfgs}
    assert len(ids) == 40


# ----------------------------------------------------------------------
# nulls.py


def _fake_sweep(rng, n_configs=10, seeds=(0, 1, 2)):
    """A run table with a strong baseline family and a much weaker ablation."""
    rows = []
    quality = {
        "gcn": 0.80,
        "sage": 0.79,
        "mlp": 0.55,
        "gcn_noprop": 0.55,
        "gcn_unnorm": 0.76,
        "sage_noneigh": 0.56,
    }
    for arch, base in quality.items():
        for c in range(n_configs):
            penalty = 0.004 * c  # some configurations are mildly worse
            for s in seeds:
                val = base - penalty + rng.normal(0, 0.005)
                rows.append(
                    {
                        "arch": arch,
                        "config_id": f"cfg{c}",
                        "seed": s,
                        "valid_accuracy": val,
                        "test_accuracy": val + rng.normal(0, 0.005),
                    }
                )
    return pd.DataFrame(rows)


def _fake_correct(runs, rng, n_inst=1000, rho=0.8):
    """Per-instance correctness with each run's accuracy and a shared difficulty.

    A purely nested construction (every model correct on a prefix of the same
    ordering) makes every paired difference noiseless, which is not how two
    trained models behave and which makes the screening rule reject everything.
    Here a shared latent difficulty carries correlation ``rho`` and the rest is
    independent, so the paired difference has a realistic variance.
    """
    from scipy import stats

    latent = rng.standard_normal(n_inst)
    out = np.empty((len(runs), n_inst), dtype=bool)
    for i, acc in enumerate(runs["test_accuracy"].to_numpy()):
        z = np.sqrt(rho) * latent + np.sqrt(1 - rho) * rng.standard_normal(n_inst)
        out[i] = z < stats.norm.ppf(float(np.clip(acc, 0.01, 0.99)))
    return out


def test_baseline_is_selected_on_validation_not_test():
    runs = pd.DataFrame(
        {
            "arch": ["gcn"] * 3,
            "config_id": ["a", "b", "c"],
            "seed": [0, 0, 0],
            "valid_accuracy": [0.80, 0.70, 0.60],
            "test_accuracy": [0.70, 0.90, 0.80],  # test disagrees with validation
        }
    )
    idx, test = nulls.select_baseline(runs, "gcn")
    assert idx == 0
    assert test == pytest.approx(0.70)


def test_select_baseline_raises_for_a_missing_architecture():
    runs = pd.DataFrame(
        {"arch": ["gcn"], "config_id": ["a"], "seed": [0],
         "valid_accuracy": [0.8], "test_accuracy": [0.8]}
    )
    with pytest.raises(ValueError, match="no runs"):
        nulls.select_baseline(runs, "sage")


def test_null_populations_partition_the_runs():
    rng = np.random.default_rng(0)
    runs = _fake_sweep(rng)
    pops = nulls.build_null_populations(runs)
    assert set(pops) == {"seed", "config", "ablation", "pooled"}
    # pooled is exactly config plus ablation, and never contains the baseline
    assert set(pops["pooled"].run_index) == set(pops["config"].run_index) | set(
        pops["ablation"].run_index
    )
    for pop in pops.values():
        assert pop.baseline_run not in set(pop.run_index)
    # seed variation shares the baseline's configuration
    base_cfg = runs.loc[pops["seed"].baseline_run, "config_id"]
    assert (runs.loc[pops["seed"].run_index, "config_id"] == base_cfg).all()


def test_screening_removes_the_much_worse_models():
    rng = np.random.default_rng(1)
    runs = _fake_sweep(rng)
    correct = _fake_correct(runs, rng)
    pops = nulls.build_null_populations(runs)
    kept, info = nulls.screened_population(runs, correct, pops["pooled"], n_boot=400)
    assert info["n_after"] < info["n_before"]
    assert not info["relaxed"]
    # The unscreened spread is dominated by models tens of points worse.
    assert pops["pooled"].summary()["sd_delta"] > 5 * kept.summary()["sd_delta"]
    # Nothing surviving the screen is far below the baseline.
    assert kept.delta.min() > -0.10


def test_screening_relaxes_rather_than_returning_nothing():
    """A baseline so strong that nothing passes the rule must still yield a
    usable population, with the relaxation recorded."""
    rng = np.random.default_rng(11)
    runs = _fake_sweep(rng, n_configs=6, seeds=(0, 1))
    # Make the baseline configuration far better than everything else.
    best = runs["valid_accuracy"].idxmax()
    runs.loc[best, ["valid_accuracy", "test_accuracy"]] = [0.99, 0.99]
    correct = _fake_correct(runs, rng)
    pops = nulls.build_null_populations(runs, baseline_arch="gcn")
    kept, info = nulls.screened_population(
        runs, correct, pops["pooled"], n_boot=400, min_keep=10
    )
    assert info["relaxed"] is True
    assert info["n_after"] == 10
    assert kept.delta.size == 10
    # It keeps the closest candidates, not an arbitrary ten.
    assert kept.delta.min() == pytest.approx(np.sort(pops["pooled"].delta)[-10])


def test_empty_population_summarises_without_crashing():
    pop = nulls.NullPopulation("empty", np.array([]), np.array([], dtype=int), 0, 0.8)
    s = pop.summary()
    assert s["n"] == 0
    assert np.isnan(s["sd_delta"]) and np.isnan(s["max_delta"])


def test_screen_rule_is_scaled_by_the_per_trial_noise():
    delta = np.array([-0.001, -0.05, 0.002])
    sigma = np.array([0.01, 0.01, 0.01])
    keep = nulls.screen_not_worse(delta, sigma, n_trials=100)
    assert keep.tolist() == [True, False, True]
    # A noisier metric keeps more, because less is detectable.
    keep_loose = nulls.screen_not_worse(delta, sigma * 10, n_trials=100)
    assert keep_loose.all()


def test_selection_null_grows_with_the_search_size():
    rng = np.random.default_rng(2)
    runs = _fake_sweep(rng)
    pops = nulls.build_null_populations(runs)
    out = nulls.selection_null(
        runs,
        pops["pooled"].baseline_run,
        pops["pooled"].run_index,
        n_search=(1, 5, 20),
        n_draws=4000,
        seed=0,
    )
    means = [r["mean_delta"] for r in out["grid"]]
    assert means[0] < means[1] < means[2]
    assert out["pool_size"] == len(pops["pooled"].run_index)


def test_selection_null_caps_the_search_at_the_pool_size():
    rng = np.random.default_rng(3)
    runs = _fake_sweep(rng, n_configs=2, seeds=(0,))
    pops = nulls.build_null_populations(runs)
    out = nulls.selection_null(
        runs,
        pops["pooled"].baseline_run,
        pops["pooled"].run_index,
        n_search=(1, 10_000),
        n_draws=200,
        seed=0,
    )
    assert out["grid"][1]["n_search_effective"] == out["pool_size"]


def test_bootstrap_sigma_delta_is_smaller_under_joint_resampling():
    """Two models that mostly agree have a small paired difference, and joint
    resampling is what preserves that."""
    rng = np.random.default_rng(4)
    runs = _fake_sweep(rng)
    correct = _fake_correct(runs, rng)
    pops = nulls.build_null_populations(runs)
    pop = pops["config"]
    j = nulls.bootstrap_sigma_delta(
        correct, pop.baseline_run, pop.run_index, n_boot=1500, seed=0, joint=True
    )
    i = nulls.bootstrap_sigma_delta(
        correct, pop.baseline_run, pop.run_index, n_boot=1500, seed=0, joint=False
    )
    assert j["median_sd_delta"] < i["median_sd_delta"]
    assert j["mean_pairwise_correlation"] > i["mean_pairwise_correlation"]


def test_headroom_is_the_distance_to_the_ceiling():
    assert nulls.headroom(0.82) == pytest.approx(0.18)
    assert nulls.headroom(0.82, ceiling=0.95) == pytest.approx(0.13)


# ----------------------------------------------------------------------
# models.py and loaders.py, only what does not need a download


def test_architecture_registry_declares_its_ablations():
    from bzoo.ml import models

    for name, spec in models.ARCHITECTURES.items():
        assert spec["prop"] in ("sym", "row", "none")
        target = spec["ablation_of"]
        assert target is None or target in models.ARCHITECTURES
    ablations = [n for n, s in models.ARCHITECTURES.items() if s["ablation_of"]]
    assert set(ablations) == set(nulls.ABLATION_ARCHS)


def test_ablation_keeps_the_parameter_count_of_its_target():
    from bzoo.ml import models

    cfg = dict(n_hidden=64, n_layers=2, dropout=0.5, lr=0.01,
               weight_decay=0.0, batch_norm=False)
    n = {}
    for arch in ("gcn", "gcn_noprop", "gcn_unnorm", "sage", "sage_noneigh"):
        n[arch] = models.n_parameters(models.build_model(arch, 100, 7, cfg))
    assert n["gcn_noprop"] == n["gcn"]
    assert n["gcn_unnorm"] == n["gcn"]
    assert n["sage_noneigh"] == n["sage"]


def test_normalised_adjacency_row_sums():
    import scipy.sparse as sp

    from bzoo.ml import loaders

    a = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    row = loaders.normalise_adjacency(a, mode="row")
    assert np.allclose(np.asarray(row.sum(axis=1)).ravel(), 1.0)
    sym = loaders.normalise_adjacency(a, mode="sym")
    assert np.allclose(sym.toarray(), sym.toarray().T)
    none = loaders.normalise_adjacency(a, mode="none")
    assert np.allclose(np.diag(none.toarray()), 1.0)  # self loops added
    with pytest.raises(ValueError):
        loaders.normalise_adjacency(a, mode="nonsense")


def test_screening_reports_when_no_null_population_exists():
    """A sweep whose baseline beats everything else by many noise units contains
    no null population, and the screen must say so rather than returning the
    least-bad candidates as if they were one.  This is what happened on
    ogbn-arxiv with a five-configuration sweep."""
    rng = np.random.default_rng(40)
    runs = _fake_sweep(rng, n_configs=5, seeds=(0, 1))
    # The baseline architecture's best run is the one that has to run away from
    # the pool; picking the global best could land on another architecture and
    # leave the gcn pool intact.
    gcn = runs.loc[runs["arch"] == "gcn"]
    best = gcn["valid_accuracy"].idxmax()
    runs.loc[best, ["valid_accuracy", "test_accuracy"]] = [0.99, 0.99]
    correct = _fake_correct(runs, rng)
    pops = nulls.build_null_populations(runs, baseline_arch="gcn")
    _, info = nulls.screened_population(
        runs, correct, pops["pooled"], n_boot=400, min_keep=10, min_pass=5
    )
    assert info["n_passing_rule"] == 0
    assert info["estimable"] is False
    assert info["relaxed"] is True
    assert info["baseline_lead_in_noise_units"] > 3.0


def test_screening_is_estimable_when_the_pool_has_near_baseline_candidates():
    rng = np.random.default_rng(41)
    runs = _fake_sweep(rng, n_configs=12, seeds=(0, 1, 2))
    correct = _fake_correct(runs, rng)
    pops = nulls.build_null_populations(runs, baseline_arch="gcn")
    _, info = nulls.screened_population(
        runs, correct, pops["pooled"], n_boot=400, min_pass=5
    )
    assert info["estimable"] is True
    assert info["n_passing_rule"] >= 5
    assert info["baseline_lead_in_noise_units"] < 3.0
