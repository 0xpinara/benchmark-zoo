"""Random search with tuning-budget parity enforced in code.

The plan's rule is that a proposed model must not get a larger tuning budget
than its baselines, and that the rule has to be enforced rather than intended.
:func:`check_budget_parity` raises if any architecture in a sweep received a
different number of trials, or a different set of seeds, from any other.  It is
called by the sweep script before any result is written, so a sweep that
violates parity produces no output at all.

The search space is deliberately modest and identical across architectures.
Every architecture sees exactly the same list of configurations, drawn once
from a fixed seed, so a difference in outcome between two architectures cannot
come from one of them having been handed easier hyperparameters.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

SEARCH_SPACE = {
    "n_hidden": [32, 64, 128, 256],
    "n_layers": [2, 3],
    "dropout": [0.0, 0.2, 0.5, 0.7],
    "lr": [0.001, 0.003, 0.01, 0.03],
    "weight_decay": [0.0, 1e-5, 5e-4, 5e-3],
    "batch_norm": [False, True],
}


def sample_configs(n_configs: int, seed: int) -> List[Dict[str, object]]:
    """Draw ``n_configs`` configurations uniformly from the grid, without
    replacement where possible.

    Uniform random draws are the point: they represent search without
    innovation, which is the null a collective-search correction needs.  A
    Bayesian or bandit search would be a better way to find a good model and a
    worse model of the null, because it concentrates on the good region and so
    understates how often plain search stumbles onto a large improvement.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(SEARCH_SPACE)
    grid_size = int(np.prod([len(SEARCH_SPACE[k]) for k in keys]))
    seen = set()
    out: List[Dict[str, object]] = []
    while len(out) < n_configs:
        cfg = {k: SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))] for k in keys}
        key = tuple(cfg[k] for k in keys)
        if key in seen and len(seen) < grid_size:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def config_id(config: Dict[str, object]) -> str:
    return "|".join(f"{k}={config[k]}" for k in sorted(config))


class BudgetParityError(RuntimeError):
    pass


def check_budget_parity(
    records: Sequence[Dict[str, object]],
    arch_key: str = "arch",
    config_key: str = "config_id",
    seed_key: str = "seed",
) -> Dict[str, object]:
    """Raise unless every architecture got the same trials and the same seeds.

    Parameters
    ----------
    records:
        One dict per completed run, with at least an architecture name, a
        configuration id and a seed.

    Returns
    -------
    A small summary that the caller writes into the results file, so that the
    parity claim in the paper is backed by something in the artifact rather
    than by a sentence.
    """
    if not records:
        raise BudgetParityError("no runs to check")
    by_arch: Dict[str, List[Dict[str, object]]] = {}
    for r in records:
        by_arch.setdefault(str(r[arch_key]), []).append(r)

    counts = {a: len(v) for a, v in by_arch.items()}
    if len(set(counts.values())) != 1:
        raise BudgetParityError(f"unequal trial counts across architectures: {counts}")

    seed_sets = {a: tuple(sorted({r[seed_key] for r in v})) for a, v in by_arch.items()}
    if len(set(seed_sets.values())) != 1:
        raise BudgetParityError(f"unequal seed sets across architectures: {seed_sets}")

    config_sets = {
        a: tuple(sorted({str(r[config_key]) for r in v})) for a, v in by_arch.items()
    }
    if len(set(config_sets.values())) != 1:
        raise BudgetParityError(
            "architectures did not see the same configurations; "
            f"sizes {[len(v) for v in config_sets.values()]}"
        )

    per_cell = Counter((str(r[arch_key]), str(r[config_key]), r[seed_key]) for r in records)
    duplicates = {k: v for k, v in per_cell.items() if v > 1}
    if duplicates:
        raise BudgetParityError(f"duplicate runs: {list(duplicates)[:5]}")

    one = next(iter(by_arch))
    return {
        "n_architectures": len(by_arch),
        "trials_per_architecture": counts[one],
        "n_configs": len(config_sets[one]),
        "n_seeds": len(seed_sets[one]),
        "seeds": list(seed_sets[one]),
        "parity_enforced": True,
    }
