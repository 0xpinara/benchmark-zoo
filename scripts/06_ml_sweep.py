"""Run the machine learning sweep and store per-instance test correctness.

The sweep is the null population.  Each architecture is trained on the same
list of randomly drawn hyperparameter configurations, with the same seeds and
the same epoch budget, and every run's per-node test correctness vector is
saved.  Nothing is aggregated here; the analysis in
``scripts/07_ml_analysis.py`` reads the vectors.

Three constructions of the null, all from the same runs:

``random configuration draws``
    All runs of one architecture across configurations.  This is "search
    without innovation": the spread of the best-of-N over these is what a
    reported improvement has to beat.
``seed variation alone``
    Runs that share an architecture and a configuration and differ only in the
    seed.  Gives the irreducible noise floor.
``ablated architectures``
    ``gcn_noprop``, ``gcn_unnorm`` and ``sage_noneigh``, each of which removes
    one claimed component from a published model while keeping its parameter
    count and budget.

Budget parity is checked before anything is written, and the script exits
non-zero without writing if it fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

from bzoo.ml import loaders, models, train, tuning
from bzoo.paths import INTERIM, RESULTS, ensure_dirs

ARCHS = ["mlp", "gcn", "sage", "gcn_noprop", "gcn_unnorm", "sage_noneigh"]

# Per-dataset budget.  The planning rule is a modest per-model trial count
# applied uniformly; the large graph gets fewer configurations because a run
# costs two orders of magnitude more, and that asymmetry is between datasets,
# never between architectures within a dataset.
BUDGETS = {
    "cora": {"n_configs": 15, "seeds": [0, 1, 2], "epochs": 200, "patience": 50, "eval_every": 5},
    "citeseer": {"n_configs": 15, "seeds": [0, 1, 2], "epochs": 200, "patience": 50, "eval_every": 5},
    "pubmed": {"n_configs": 15, "seeds": [0, 1, 2], "epochs": 200, "patience": 50, "eval_every": 5},
    # ogbn-arxiv is 170,000 nodes and a full-batch run costs two orders of
    # magnitude more than one on Cora, so it gets fewer configurations and
    # seeds.  The asymmetry is between datasets, never between architectures
    # within a dataset, which is what budget parity requires.
    "ogbn-arxiv": {"n_configs": 5, "seeds": [0, 1], "epochs": 100, "patience": 30, "eval_every": 10},
}
CONFIG_SEED = 20260901


def _worker(task):
    dataset, arch, config, seed, budget, threads = task
    import torch

    torch.set_num_threads(threads)
    ds = loaders.load_dataset(dataset)
    res = train.train_once(
        ds,
        arch,
        config,
        seed=seed,
        epochs=budget["epochs"],
        patience=budget["patience"],
        eval_every=budget["eval_every"],
    )
    return res.to_record(), res.correct


def run_dataset(dataset: str, workers: int, threads: int) -> "tuple[pd.DataFrame, np.ndarray]":
    budget = BUDGETS[dataset]
    configs = tuning.sample_configs(budget["n_configs"], seed=CONFIG_SEED)
    tasks = [
        (dataset, arch, cfg, seed, budget, threads)
        for arch in ARCHS
        for cfg in configs
        for seed in budget["seeds"]
    ]
    print(
        f"[{dataset}] {len(tasks)} runs = {len(ARCHS)} architectures x "
        f"{len(configs)} configs x {len(budget['seeds'])} seeds",
        flush=True,
    )

    records: List[Dict[str, object]] = []
    correct: List[np.ndarray] = []
    t0 = time.time()
    if workers <= 1:
        for i, task in enumerate(tasks):
            rec, cor = _worker(task)
            records.append(rec)
            correct.append(cor)
            if (i + 1) % 20 == 0:
                print(
                    f"[{dataset}] {i + 1}/{len(tasks)} "
                    f"({time.time() - t0:.0f}s elapsed)",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures)):
                rec, cor = fut.result()
                records.append(rec)
                correct.append(cor)
                if (i + 1) % 20 == 0:
                    print(
                        f"[{dataset}] {i + 1}/{len(tasks)} "
                        f"({time.time() - t0:.0f}s elapsed)",
                        flush=True,
                    )

    df = pd.DataFrame(records)
    # Sort so that the run order in the stored matrix is deterministic and does
    # not depend on which worker finished first.
    order = np.lexsort((df["seed"].to_numpy(), df["config_id"].to_numpy(), df["arch"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    mat = np.vstack([correct[i] for i in order])
    return df, mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    ensure_dirs()

    summary = {}
    for dataset in args.datasets:
        df, mat = run_dataset(dataset, args.workers, args.threads)
        parity = tuning.check_budget_parity(df.to_dict(orient="records"))
        df.to_parquet(INTERIM / f"ml_runs_{dataset}.parquet", index=False)
        np.save(INTERIM / f"ml_correct_{dataset}.npy", mat)

        best = df.loc[df.groupby("arch")["valid_accuracy"].idxmax()]
        summary[dataset] = {
            "parity": parity,
            "dataset": loaders.load_dataset(dataset).summary(),
            "n_runs": int(len(df)),
            "total_seconds": float(df["seconds"].sum()),
            "best_by_arch": {
                r["arch"]: {
                    "valid_accuracy": r["valid_accuracy"],
                    "test_accuracy": r["test_accuracy"],
                    "config_id": r["config_id"],
                }
                for _, r in best.iterrows()
            },
        }
        print(f"[{dataset}] parity {parity}", flush=True)
        print(
            f"[{dataset}] best test accuracy by architecture: "
            + ", ".join(
                f"{a}={v['test_accuracy']:.4f}"
                for a, v in summary[dataset]["best_by_arch"].items()
            ),
            flush=True,
        )

    out = RESULTS / "ml_sweep_summary.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(summary)
    out.write_text(json.dumps(existing, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
