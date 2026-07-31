"""Apply the calibrated corrections to the machine learning benchmark.

Reads the sweep output and answers four questions.

1. What is ``sigma_Delta``, the standard deviation of the improvement over a
   tuned baseline under each of the three null constructions?  Reported as an
   across-run number and as a bootstrap number, so that the reader can see how
   much of the spread is real difference and how much is test-set noise.
2. What improvement does a reported result need in order to survive, as a
   function of the number of trials the field is credited with?  Reported over
   ``N`` from 10 to 100,000, never at a single ``N``.
3. How many entries on the ogbn-arxiv leaderboard beat the entry below them by
   more than the deflated threshold?  This is the question the paper lives on.
4. Does the correction bind harder on a saturated benchmark?  Compared across
   the four datasets, which sit at different distances from their ceiling.

Everything is also computed with the closed-form deflation replaced by a
permutation alternative, and with joint resampling replaced by the incorrect
per-model version, so both robustness claims are measured rather than asserted.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from bzoo.corrections import bootstrap_tests, deflation, fdr, fwer
from bzoo.ml import nulls
from bzoo.paths import INTERIM, RAW, RESULTS, ensure_dirs
from bzoo.resample import instance as inst
from bzoo.resample.permutation import sign_flip_replicates, studentise

SEED = 20260901
ALPHA = 0.05
N_GRID = (10, 100, 1_000, 10_000, 100_000)
N_BOOT = 4000
BASELINE_ARCH = "gcn"
DATASETS = ("cora", "citeseer", "pubmed", "ogbn-arxiv")


def load_runs(dataset: str):
    runs_path = INTERIM / f"ml_runs_{dataset}.parquet"
    corr_path = INTERIM / f"ml_correct_{dataset}.npy"
    if not runs_path.exists():
        return None, None
    runs = pd.read_parquet(runs_path)
    correct = np.load(corr_path)
    if len(runs) != correct.shape[0]:
        raise ValueError(f"{dataset}: run table and matrix disagree in length")
    return runs, correct


def analyse_dataset(dataset: str) -> dict:
    runs, correct = load_runs(dataset)
    if runs is None:
        return {"available": False}

    # Replicate count scaled to the test set.  A standard deviation over 48,603
    # instances is already precise at 1,000 replicates, and each replicate of the
    # per-model scheme costs one gather per model, so 4,000 would quadruple the
    # cost of the whole script for no gain in precision.
    n_boot = N_BOOT if correct.shape[1] <= 5_000 else 1_000

    pops = nulls.build_null_populations(runs, baseline_arch=BASELINE_ARCH)
    base_idx = pops["pooled"].baseline_run
    base_test = pops["pooled"].baseline_test_accuracy

    out: dict = {
        "available": True,
        "n_runs": int(len(runs)),
        "n_test_instances": int(correct.shape[1]),
        "n_bootstrap": int(n_boot),
        "baseline_arch": BASELINE_ARCH,
        "baseline_test_accuracy": base_test,
        "baseline_valid_accuracy": float(runs.loc[base_idx, "valid_accuracy"]),
        "best_test_accuracy_any_arch": float(runs["test_accuracy"].max()),
        "best_valid_selected_test_accuracy": float(
            runs.loc[runs["valid_accuracy"].idxmax(), "test_accuracy"]
        ),
        "headroom_to_one": nulls.headroom(base_test),
        "populations": {},
    }

    for name, pop in pops.items():
        s = pop.summary()
        boot = nulls.bootstrap_sigma_delta(
            correct, pop.baseline_run, pop.run_index, n_boot=n_boot, seed=SEED
        )
        boot_indep = nulls.bootstrap_sigma_delta(
            correct,
            pop.baseline_run,
            pop.run_index,
            n_boot=n_boot,
            seed=SEED,
            joint=False,
        )
        s["bootstrap_sigma_delta_joint"] = boot["median_sd_delta"]
        s["bootstrap_sigma_delta_independent"] = boot_indep["median_sd_delta"]
        s["bootstrap_pairwise_correlation_joint"] = boot["mean_pairwise_correlation"]
        s["bootstrap_pairwise_correlation_independent"] = boot_indep[
            "mean_pairwise_correlation"
        ]
        s["across_run_over_bootstrap"] = (
            s["sd_delta"] / boot["median_sd_delta"]
            if boot["median_sd_delta"] > 0
            else float("nan")
        )
        out["populations"][name] = s

    # Screen the pooled population: drop trials that are detectably worse than
    # the baseline, because their true improvement is not zero and including
    # them inflates sigma_Delta without touching its upper tail.
    screened, screen_info = nulls.screened_population(
        runs, correct, pops["pooled"], n_boot=n_boot, seed=SEED
    )
    s_scr = screened.summary()
    boot_scr = nulls.bootstrap_sigma_delta(
        correct, screened.baseline_run, screened.run_index, n_boot=n_boot, seed=SEED
    )
    s_scr["bootstrap_sigma_delta_joint"] = boot_scr["median_sd_delta"]
    s_scr["bootstrap_sigma_delta_independent"] = float("nan")
    s_scr["bootstrap_pairwise_correlation_joint"] = boot_scr[
        "mean_pairwise_correlation"
    ]
    s_scr["bootstrap_pairwise_correlation_independent"] = float("nan")
    s_scr["across_run_over_bootstrap"] = (
        s_scr["sd_delta"] / boot_scr["median_sd_delta"]
        if boot_scr["median_sd_delta"] > 0
        else float("nan")
    )
    out["populations"]["screened"] = s_scr
    out["screening"] = screen_info
    out["seed_noise_floor"] = nulls.seed_noise_floor(runs)

    # The selection null: simulate what a researcher who tried m things and
    # reported the best on validation would see on test.
    # Two pools.  The full pool answers "search blindly over everything",
    # which on a benchmark where the baseline is strong is dominated by
    # candidates nobody would report.  The screened pool answers the question a
    # reader cares about: search over m plausible candidates that are not
    # detectably worse, report the best on validation, and see what improvement
    # you announce.
    out["selection_null"] = nulls.selection_null(
        runs, pops["pooled"].baseline_run, pops["pooled"].run_index, seed=SEED
    )
    out["selection_null_screened"] = nulls.selection_null(
        runs, screened.baseline_run, screened.run_index, seed=SEED
    )

    # Deflation thresholds from the screened null, over the trial grid -- but
    # only when the screening rule actually found a null population.  When it
    # did not, no threshold is reported at all; see the note below.
    sigma = s_scr["sd_delta"]
    sigma_unscreened = out["populations"]["pooled"]["sd_delta"]
    sigma_seed = out["populations"]["seed"]["sd_delta"]
    estimable = bool(screen_info["estimable"])
    out["sigma_delta_estimable"] = estimable
    if estimable:
        grid = []
        for n in N_GRID:
            d = deflation.deflated_improvement(0.0, sigma, n, alpha=ALPHA)
            grid.append(
                {
                    "n_trials": n,
                    "expected_max_delta": d["expected_max"],
                    "threshold_delta": d["threshold"],
                    "threshold_accuracy_points": 100 * d["threshold"],
                }
            )
        out["deflation_grid_pooled"] = grid
        out["sigma_delta_pooled"] = sigma
    else:
        out["deflation_grid_pooled"] = []
        out["sigma_delta_pooled"] = None
        out["not_estimable_reason"] = (
            f"{screen_info['n_passing_rule']} of {screen_info['n_before']} null "
            f"trials pass the screening rule, below the minimum of "
            f"{screen_info['min_pass']}. The tuned baseline leads the best other "
            f"candidate in the pool by "
            f"{screen_info['baseline_lead_in_noise_units']:.1f} times the "
            f"per-trial noise, so the sweep contains no candidate that is "
            f"plausibly as good as the baseline and therefore no null "
            f"population. A spread computed over the least-bad candidates would "
            f"be a spread of real differences, not of noise, and the threshold "
            f"it produced would be meaningless. A larger configuration sweep is "
            f"needed."
        )
    out["sigma_delta_screened_if_forced"] = sigma
    out["sigma_delta_unscreened"] = sigma_unscreened
    out["sigma_delta_seed"] = sigma_seed
    # The median cell is the representative figure; the pooled number is
    # dominated by the handful of configurations where training is unstable, and
    # both are reported because the difference between them is itself the point.
    out["sigma_delta_noise_floor"] = out["seed_noise_floor"]["median_cell_sd"]
    out["sigma_delta_noise_floor_pooled"] = out["seed_noise_floor"]["sd"]

    # Permutation alternative to the closed form: sign-flip the per-instance
    # differences between the baseline and each null candidate, then take the
    # maximum over candidates.  Makes no normality assumption.
    cand = screened.run_index
    scores = correct[[base_idx] + list(cand)].T.astype(float)
    diffs = inst.paired_differences(scores, baseline_col=0)
    rng = np.random.default_rng(SEED)
    perm = sign_flip_replicates(diffs, n_boot, rng)
    perm_max = perm.max(axis=1)
    out["permutation_alternative"] = {
        "n_candidates": int(diffs.shape[1]),
        "threshold_q95": float(np.quantile(perm_max, 1.0 - ALPHA)),
        "expected_max": float(perm_max.mean()),
        "closed_form_threshold_at_matched_n": deflation.deflated_threshold(
            sigma, max(2, diffs.shape[1]), ALPHA
        ),
        "closed_form_expected_max_at_matched_n": deflation.expected_max_normal(
            sigma, max(2, diffs.shape[1])
        ),
    }

    # Metric-distribution check: is the maximum of Delta approximately normal?
    d_pool = screened.delta
    out["normality_of_delta"] = {
        "skew": float(stats.skew(d_pool)),
        "kurtosis": float(stats.kurtosis(d_pool, fisher=False)),
        "shapiro_p": float(stats.shapiro(d_pool[:500]).pvalue)
        if d_pool.size >= 20
        else float("nan"),
        "jarque_bera_p": float(stats.jarque_bera(d_pool).pvalue),
    }

    # Reality Check and SPA on the per-instance differentials: does the best of
    # the null population beat the baseline once the search is accounted for?
    rc = bootstrap_tests.white_reality_check(
        diffs, n_boot=n_boot, scheme="iid", alpha=ALPHA, seed=SEED
    )
    spa = bootstrap_tests.hansen_spa(
        diffs, n_boot=n_boot, scheme="iid", alpha=ALPHA, seed=SEED
    )
    obs, cent, _ = bootstrap_tests.bootstrap_centred_matrix(
        diffs, n_boot=n_boot, scheme="iid", seed=SEED, studentised=True
    )
    rw = fwer.romano_wolf(obs, cent, alpha=ALPHA)
    out["resampling_tests_on_null_population"] = {
        "reality_check_p": rc.extra["p_value"],
        "spa_p": spa.extra["p_value"],
        "romano_wolf_n_reject": rw.n_reject,
        "n_candidates": int(diffs.shape[1]),
        "comment": (
            "These are run on the constructed null population, so a small "
            "p-value here would mean the population is not null. It is a "
            "check on the construction, not a result about published models."
        ),
    }
    return out


# Entries whose method name says they use the paper text or an external
# corpus, and therefore sit outside the scope of our sweep.  The rule is on the
# name as submitted, applied before any result is computed, and it is listed
# here rather than described so that it can be checked.
TEXT_FEATURE_MARKERS = (
    "GIANT", "TAPE", "SimTeG", "GLEM", "GraDBERT", "SciBERT", "BiGTex",
    "E2EG", "use raw text", "use MAG data",
)


def _uses_text_features(method: str) -> bool:
    m = str(method)
    return any(k.lower() in m.lower() for k in TEXT_FEATURE_MARKERS)


def leaderboard_analysis(sigma_by_dataset: dict) -> dict:
    """Deflate the ogbn-arxiv leaderboard."""
    path = RAW / "leaderboards" / "ogbn_arxiv_leaderboard.csv"
    lb = pd.read_csv(path, parse_dates=["date"]).sort_values(
        "test_acc", ascending=False
    )
    lb = lb.reset_index(drop=True)
    lb["uses_text"] = lb["method"].map(_uses_text_features)

    gaps = -lb["test_acc"].diff().dropna()
    out = {
        "n_entries": int(len(lb)),
        "n_ranks": int(lb["rank"].nunique()),
        "date_min": str(lb["date"].min().date()),
        "date_max": str(lb["date"].max().date()),
        "best_test_acc": float(lb["test_acc"].max()),
        "worst_test_acc": float(lb["test_acc"].min()),
        "median_reported_std": float(lb["test_std"].median()),
        "median_adjacent_gap": float(gaps.median()),
        "share_adjacent_gaps_below_median_std": float(
            (gaps < lb["test_std"].median()).mean()
        ),
        "gap_over_reported_std": float(gaps.median() / lb["test_std"].median()),
    }

    sigma = sigma_by_dataset.get("ogbn-arxiv")
    if sigma is None:
        out["deflation"] = {
            "available": False,
            "reason": "no ogbn-arxiv sweep, so no sigma_Delta for this benchmark",
        }
        return out

    # Improvement of each entry over the best entry that predates it, which is
    # the comparison a paper actually makes.
    def _advances(frame: pd.DataFrame) -> pd.DataFrame:
        t = frame.sort_values("date").reset_index(drop=True)
        prev = t["test_acc"].cummax().shift(1)
        claims = pd.DataFrame(
            {
                "method": t["method"],
                "date": t["date"],
                "test_acc": t["test_acc"],
                "previous_best": prev,
                "delta": t["test_acc"] - prev,
                "uses_text": t["uses_text"],
            }
        ).dropna(subset=["delta"])
        return claims.loc[claims["delta"] > 0].copy()

    out["deflation"] = {"available": True, "sigma_delta": sigma}
    for scope, frame in (
        ("all", lb),
        ("graph_features_only", lb.loc[~lb["uses_text"]]),
    ):
        adv = _advances(frame)
        rows = []
        for n in N_GRID:
            thr = deflation.deflated_threshold(sigma, n, ALPHA)
            rows.append(
                {
                    "n_trials": n,
                    "threshold_delta": thr,
                    "threshold_accuracy_points": 100 * thr,
                    "n_advances": int(len(adv)),
                    "n_survive": int((adv["delta"] > thr).sum()),
                    "share_survive": float((adv["delta"] > thr).mean()),
                }
            )
        out["deflation"][scope] = {
            "n_entries": int(len(frame)),
            "n_advances": int(len(adv)),
            "median_advance": float(adv["delta"].median()),
            "largest_advance": float(adv["delta"].max()),
            "grid": rows,
        }
        if scope == "graph_features_only":
            out["advances"] = adv.assign(date=adv["date"].astype(str)).to_dict(
                orient="records"
            )
    # The primary scope is the one our null population covers.  Entries that use
    # the paper text or an external corpus are outside the architecture family we
    # swept, so a sigma estimated from graph-only configurations is not the right
    # null for them; both are reported and the choice is stated.
    out["deflation"]["grid"] = out["deflation"]["graph_features_only"]["grid"]
    out["deflation"]["primary_scope"] = "graph_features_only"
    out["deflation"]["n_text_feature_entries"] = int(lb["uses_text"].sum())
    out["deflation"]["leaderboard_lower_bound_on_n"] = int(len(lb))
    return out


def saturation_analysis(per_dataset: dict) -> dict:
    """Does the correction bind harder as a benchmark saturates?

    Two quantities per dataset: how much headroom is left above the tuned
    baseline, and how large ``sigma_Delta`` is.  If ``sigma_Delta`` shrinks
    more slowly than the headroom, then the share of the remaining headroom
    that a null search can produce by chance grows, and the correction binds
    harder on the more saturated benchmark.
    """
    rows = []
    excluded = []
    for name, blk in per_dataset.items():
        if not blk.get("available"):
            continue
        if not blk.get("sigma_delta_estimable"):
            excluded.append(name)
            continue
        sigma = blk["sigma_delta_pooled"]
        head = blk["headroom_to_one"]
        rows.append(
            {
                "dataset": name,
                "baseline_accuracy": blk["baseline_test_accuracy"],
                "headroom": head,
                "sigma_delta": sigma,
                "sigma_over_headroom": sigma / head if head > 0 else float("nan"),
                "n_test_instances": blk["n_test_instances"],
            }
        )
    df = pd.DataFrame(rows)
    out = {"table": df.to_dict(orient="records"), "excluded": excluded}
    if excluded:
        out["excluded_reason"] = (
            "no null population could be estimated on these benchmarks, so they "
            "have no sigma_Delta to plot"
        )
    if len(df) >= 3:
        # Rank correlation, because four points cannot support anything else.
        rho, p = stats.spearmanr(df["baseline_accuracy"], df["sigma_over_headroom"])
        out["spearman_accuracy_vs_sigma_over_headroom"] = {
            "rho": float(rho),
            "p_value": float(p),
            "n": int(len(df)),
        }
        if df["n_test_instances"].nunique() > 1:
            rho2, p2 = stats.spearmanr(df["n_test_instances"], df["sigma_delta"])
            out["spearman_test_size_vs_sigma"] = {
                "rho": float(rho2),
                "p_value": float(p2),
                "n": int(len(df)),
            }
        else:
            out["spearman_test_size_vs_sigma"] = {
                "skipped": "test set size is the same for every dataset present"
            }
    return out


def main() -> int:
    ensure_dirs()
    results = {"config": {"seed": SEED, "alpha": ALPHA, "n_boot": N_BOOT}}
    per_dataset = {}
    sigma_by_dataset = {}

    for ds in DATASETS:
        blk = analyse_dataset(ds)
        per_dataset[ds] = blk
        if blk.get("available"):
            if blk.get("sigma_delta_estimable"):
                sigma_by_dataset[ds] = blk["sigma_delta_pooled"]
            print(f"\n=== {ds} ===", flush=True)
            print(
                f"  {blk['n_runs']} runs, {blk['n_test_instances']} test nodes; "
                f"tuned {BASELINE_ARCH} baseline test accuracy "
                f"{blk['baseline_test_accuracy']:.4f}",
                flush=True,
            )
            print(
                f"  seed noise floor: pooled sd "
                f"{blk['seed_noise_floor']['sd']:.5f}, median cell sd "
                f"{blk['seed_noise_floor']['median_cell_sd']:.5f}, max cell sd "
                f"{blk['seed_noise_floor']['max_cell_sd']:.5f} "
                f"({blk['seed_noise_floor']['n_cells']} cells)",
                flush=True,
            )
            si = blk["screening"]
            print(
                f"  screening: {si['n_passing_rule']} of {si['n_before']} pass "
                f"the rule, {si['n_after']} kept"
                + (" (relaxed)" if si["relaxed"] else "")
                + f"; baseline leads the pool by "
                f"{si['baseline_lead_in_noise_units']:.1f} noise units; "
                f"estimable: {si['estimable']}",
                flush=True,
            )
            if not blk["sigma_delta_estimable"]:
                print(f"  NOT ESTIMABLE: {blk['not_estimable_reason']}", flush=True)
            for name, s in blk["populations"].items():
                print(
                    f"  {name:28s} n={s['n']:4d} "
                    f"sd(Delta)={s['sd_delta']:.5f} "
                    f"boot sd(Delta)={s['bootstrap_sigma_delta_joint']:.5f} "
                    f"(indep {s['bootstrap_sigma_delta_independent']:.5f}) "
                    f"ratio={s['across_run_over_bootstrap']:.2f} "
                    f"max Delta={s['max_delta']:+.4f}",
                    flush=True,
                )
            g = blk["deflation_grid_pooled"]
            if g:
                print(
                    "  deflated threshold (accuracy points): "
                    + ", ".join(
                        f"N={r['n_trials']}:{r['threshold_accuracy_points']:.2f}"
                        for r in g
                    ),
                    flush=True,
                )
            for tag, key in (
                ("full pool", "selection_null"),
                ("screened pool", "selection_null_screened"),
            ):
                sel = blk[key]
                print(
                    f"  selection null ({tag}, size {sel['pool_size']}), "
                    "mean / 95th pct reported Delta in points: "
                    + ", ".join(
                        f"m={r['n_search']}:{100 * r['mean_delta']:+.2f}/"
                        f"{100 * r['q95_delta']:+.2f}"
                        for r in sel["grid"]
                    ),
                    flush=True,
                )
            pa = blk["permutation_alternative"]
            print(
                f"  permutation threshold {pa['threshold_q95']:.5f} vs closed form "
                f"{pa['closed_form_threshold_at_matched_n']:.5f} "
                f"(matched N={pa['n_candidates']})",
                flush=True,
            )
            nd = blk["normality_of_delta"]
            print(
                f"  Delta skew={nd['skew']:+.2f} kurtosis={nd['kurtosis']:.2f} "
                f"Jarque-Bera p={nd['jarque_bera_p']:.3g}",
                flush=True,
            )
        else:
            print(f"\n=== {ds}: no sweep output, skipped ===", flush=True)

    results["datasets"] = per_dataset
    print("\n=== ogbn-arxiv leaderboard ===", flush=True)
    results["leaderboard"] = leaderboard_analysis(sigma_by_dataset)
    lb = results["leaderboard"]
    print(
        f"  {lb['n_entries']} entries over {lb['n_ranks']} ranks, "
        f"{lb['date_min']} to {lb['date_max']}",
        flush=True,
    )
    print(
        f"  median adjacent gap {lb['median_adjacent_gap']:.5f}, "
        f"median reported std {lb['median_reported_std']:.5f}, "
        f"ratio {lb['gap_over_reported_std']:.2f}; "
        f"{100 * lb['share_adjacent_gaps_below_median_std']:.0f}% of gaps are "
        "smaller than one reported standard deviation",
        flush=True,
    )
    if lb["deflation"]["available"]:
        for r in lb["deflation"]["grid"]:
            print(
                f"  N={r['n_trials']:>6}: threshold "
                f"{r['threshold_accuracy_points']:.2f} points, "
                f"{r['n_survive']}/{r['n_advances']} advances survive",
                flush=True,
            )

    print("\n=== saturation ===", flush=True)
    results["saturation"] = saturation_analysis(per_dataset)
    for r in results["saturation"]["table"]:
        print(
            f"  {r['dataset']:12s} baseline={r['baseline_accuracy']:.4f} "
            f"headroom={r['headroom']:.4f} sigma={r['sigma_delta']:.5f} "
            f"sigma/headroom={r['sigma_over_headroom']:.4f}",
            flush=True,
        )
    if "spearman_accuracy_vs_sigma_over_headroom" in results["saturation"]:
        s = results["saturation"]["spearman_accuracy_vs_sigma_over_headroom"]
        print(
            f"  Spearman(accuracy, sigma/headroom) = {s['rho']:.3f} "
            f"(p={s['p_value']:.3f}, n={s['n']})",
            flush=True,
        )

    out = RESULTS / "ml_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
