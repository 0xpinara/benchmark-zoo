"""How many trials, actually? Three bounds, none of them the answer.

The finance literature treats the total number of tests $M$ as a free parameter
and estimates it structurally, because nothing about it is observable. Machine
learning benchmarks do better, and this script collects what is observable and
is careful about what each number is.

``leaderboard``
    Submissions to the ogbn-arxiv leaderboard. A **lower bound** on the number
    of trials behind the current best number, and a loose one: it counts
    attempts that were made public and succeeded well enough to submit.

``citations``
    Works citing the benchmark paper, from OpenAlex. Not a trial count in either
    direction. It bounds the number of research efforts that could have touched
    the benchmark, and most citing papers report no number on it while some
    report dozens. Different indexes disagree substantially, because preprint and
    proceedings versions are separate records, so the figure is reported with its
    source and date and is not used as an input to any threshold.

``published_predictors``
    On the finance side, the 212 documented predictors, which is the only $M$
    there that is observed rather than assumed, together with the Google Scholar
    citation counts that ``SignalDoc.csv`` records for each of them.

The point of collecting these is not to pick one. It is that the grid of $N$ in
every table has to span them, and it does: from 10 to 100,000.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from bzoo.finance import loaders
from bzoo.paths import RAW, RESULTS, ensure_dirs

OPENALEX = "https://api.openalex.org/works/doi:{doi}"
MAILTO = "aksoy.p@northeastern.edu"  # OpenAlex asks for a contact address

BENCHMARK_PAPERS = {
    "ogb": {
        "doi": "10.48550/arXiv.2005.00687",
        "label": "Hu et al. (2020), Open Graph Benchmark",
    },
    "gcn": {
        "doi": "10.48550/arXiv.1609.02907",
        "label": "Kipf and Welling (2017), GCN",
    },
    "planetoid": {
        "doi": "10.48550/arXiv.1603.08861",
        "label": "Yang et al. (2016), Planetoid splits",
    },
    "sage": {
        "doi": "10.48550/arXiv.1706.02216",
        "label": "Hamilton et al. (2017), GraphSAGE",
    },
}


def fetch_citations() -> Dict[str, dict]:
    out = {}
    for key, spec in BENCHMARK_PAPERS.items():
        url = OPENALEX.format(doi=spec["doi"]) + f"?mailto={MAILTO}"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            d = resp.json()
            out[key] = {
                "label": spec["label"],
                "doi": spec["doi"],
                "title": d.get("title"),
                "year": d.get("publication_year"),
                "cited_by_count": d.get("cited_by_count"),
                "source": "OpenAlex",
            }
        except Exception as exc:  # noqa: BLE001 - a missing count is reported
            out[key] = {"label": spec["label"], "doi": spec["doi"], "error": str(exc)}
        print(f"  {key}: {out[key].get('cited_by_count', out[key].get('error'))}",
              flush=True)
        time.sleep(1.0)  # be polite to a free API
    return out


def leaderboard_counts() -> dict:
    lb = pd.read_csv(
        RAW / "leaderboards" / "ogbn_arxiv_leaderboard.csv", parse_dates=["date"]
    )
    by_year = lb.groupby(lb["date"].dt.year).size()
    by_contact = lb["contact"].value_counts()
    return {
        "n_submissions": int(len(lb)),
        "n_displayed_ranks": int(lb["rank"].nunique()),
        "n_distinct_submitters": int(lb["contact"].nunique()),
        "submissions_per_year": {int(k): int(v) for k, v in by_year.items()},
        "max_submissions_by_one_submitter": int(by_contact.max()),
        "date_min": str(lb["date"].min().date()),
        "date_max": str(lb["date"].max().date()),
        "interpretation": (
            "A lower bound on the number of trials behind the current best "
            "number. It counts public submissions, not attempts, and a single "
            "submission is itself the outcome of an unreported search."
        ),
    }


def finance_counts() -> dict:
    doc = loaders.load_osap_signal_doc()
    pred = doc.loc[doc["Cat.Signal"] == "Predictor"].copy()
    cites = pd.to_numeric(pred.get("GScholarCites202509"), errors="coerce").dropna()
    years = pd.to_numeric(pred["Year"], errors="coerce").dropna()
    return {
        "n_documented_predictors": int(len(pred)),
        "n_documented_placebos": int((doc["Cat.Signal"] == "Placebo").sum()),
        "publication_year_min": int(years.min()),
        "publication_year_max": int(years.max()),
        "citation_count_source": "Google Scholar via SignalDoc.csv, 2025-09",
        "citations_total": int(cites.sum()) if len(cites) else None,
        "citations_median": float(cites.median()) if len(cites) else None,
        "citations_max": int(cites.max()) if len(cites) else None,
        "interpretation": (
            "212 is the only observed M in the finance literature: the number of "
            "predictors someone has documented from a published paper. The "
            "unpublished attempts behind them are invisible, which is why "
            "Harvey, Liu and Zhu estimate the total structurally."
        ),
    }


def main() -> int:
    ensure_dirs()
    results = {}
    print("=== leaderboard ===", flush=True)
    results["leaderboard"] = leaderboard_counts()
    print(json.dumps(results["leaderboard"], indent=2), flush=True)

    print("\n=== citations (OpenAlex) ===", flush=True)
    results["citations"] = fetch_citations()

    print("\n=== finance ===", flush=True)
    results["finance"] = finance_counts()
    print(json.dumps(results["finance"], indent=2), flush=True)

    lb = results["leaderboard"]["n_submissions"]
    cites = [
        v.get("cited_by_count")
        for v in results["citations"].values()
        if v.get("cited_by_count")
    ]
    results["summary"] = {
        "observed_lower_bound_ml": lb,
        "citation_range_ml": [int(min(cites)), int(max(cites))] if cites else None,
        "observed_lower_bound_finance": results["finance"]["n_documented_predictors"],
        "grid_used_in_paper": [10, 100, 1000, 10000, 100000],
        "note": (
            "The grid spans every bound above. No threshold in the paper is "
            "reported at a single N, and the observed lower bounds are marked "
            "on the tables rather than substituted for N."
        ),
    }
    out = RESULTS / "observable_n.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
