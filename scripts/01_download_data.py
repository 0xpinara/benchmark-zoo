"""Fetch and cache every data source, then report what was obtained.

Run this first.  Everything downstream reads from the parquet cache it
writes, so no other script touches the network.

Sources
-------
1. Chen and Dim mined long-short strategies (ticker, past-return).  These
   are large and are downloaded by ``make data`` with curl, because the
   Dropbox folder is served as a single 1.8 GB zip; this script only
   converts them to parquet.
2. Open Source Asset Pricing published-predictor portfolios and the signal
   documentation, through the ``openassetpricing`` package.
3. Fama-French five factors plus momentum, from Ken French's data library.

Licences, as of the run date recorded in DECISIONS.md:
  - Chen and Zimmermann OSAP: MIT (code) and freely redistributable data.
  - Chen and Dim mined strategy returns: distributed publicly by the
    authors; we redistribute only derived statistics, not the returns.
  - Ken French data library: free for academic use, not redistributed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from bzoo.finance import loaders
from bzoo.paths import CACHE, RAW, ensure_dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild every cache")
    ap.add_argument(
        "--skip-mined", action="store_true", help="only fetch OSAP and factors"
    )
    args = ap.parse_args()
    ensure_dirs()
    report = {}

    if not args.skip_mined:
        for pop in ("ticker", "pastret"):
            t0 = time.time()
            df = loaders.load_mined_returns(pop, force=args.force)
            names = loaders.load_mined_names(pop)
            report[f"mined_{pop}"] = {
                "rows": int(len(df)),
                "signals": int(df["signalid"].nunique()),
                "names": int(len(names)),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
                "seconds": round(time.time() - t0, 1),
            }
            print(f"[mined:{pop}] {report[f'mined_{pop}']}", flush=True)

    t0 = time.time()
    ports, doc = loaders.download_osap(force=args.force)
    report["osap"] = {
        "portfolio_rows": int(len(ports)),
        "predictors": int(ports["signalname"].nunique()),
        "doc_rows": int(len(doc)),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[osap] {report['osap']}", flush=True)

    t0 = time.time()
    fac = loaders.download_factors(force=args.force)
    report["factors"] = {
        "rows": int(len(fac)),
        "columns": list(fac.columns),
        "date_min": str(fac.index.min().date()),
        "date_max": str(fac.index.max().date()),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[factors] {report['factors']}", flush=True)

    out = CACHE / "download_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
