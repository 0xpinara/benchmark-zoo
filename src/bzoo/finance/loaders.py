"""Loading the finance data, with a parquet cache.

Three sources, all public:

``osap_portfolios``
    Long-short portfolio returns for the 212 published predictors of Chen
    and Zimmermann (2022), Critical Finance Review 11, 207-264.  Downloaded
    through the ``openassetpricing`` package.

``osap_signal_doc``
    The accompanying documentation table: sample period, original journal,
    original t-statistic, and the economic category of each predictor.

``mined_returns``
    The data-mined long-short strategies of Chen and Dim, High-Throughput
    Asset Pricing.  Two populations are used here:
    ``ticker`` (19,380 signals built from letters of the ticker symbol) and
    ``pastret`` (19,402 signals built from lagged own returns).

The raw mined files are about 1.8 GB of CSV.  We convert them once to
parquet, which loads them in seconds instead of minutes, and every later
step reads the parquet.  ``force=True`` rebuilds the cache.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from ..paths import CACHE, RAW, ensure_dirs

MINED_FILES = {
    "ticker": ("TickerSignalsLongShort.csv.gzip", "TickerSignalNames.csv.gzip"),
    "pastret": ("PastReturnSignalsLongShort.csv.gzip", "PastReturnSignalNames.csv.gzip"),
}


def to_month_start(index: pd.Index) -> pd.DatetimeIndex:
    """Snap a monthly datetime index to the first day of its month.

    The sources disagree: the ticker strategy file dates months by their first
    day, the past-return strategy file and the OSAP portfolios by their last, and
    the Fama-French files by their first.  Any merge between two of them on the
    raw dates silently produces an empty intersection, which is exactly what
    happened the first time we regressed the past-return strategies on the
    factors: 210 complete series and 0 usable regressions, with no error.  Every
    monthly index in this package is snapped here, so a merge cannot fail that
    way again.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    return idx.to_period("M").to_timestamp(how="start")


def _read_maybe_gzip(path, **kwargs) -> pd.DataFrame:
    """Read a CSV that may or may not actually be gzip-compressed.

    The upstream files are named ``.csv.gzip`` but were written by pandas
    with an extension it does not recognise as a compression hint, so they
    are plain text.  We sniff the magic bytes rather than trusting the name,
    because a future release could start compressing them.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, **kwargs)
    return pd.read_csv(path, **kwargs)


def load_mined_names(population: str) -> pd.DataFrame:
    """Signal id to signal name map for a mined population."""
    if population not in MINED_FILES:
        raise ValueError(f"population must be one of {sorted(MINED_FILES)}")
    _, names_file = MINED_FILES[population]
    path = RAW / "htap" / names_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/01_download_data.py first"
        )
    df = _read_maybe_gzip(path)
    df["signalid"] = df["signalid"].astype(np.int32)
    return df


def load_mined_returns(
    population: str,
    force: bool = False,
    weighting: str = "both",
) -> pd.DataFrame:
    """Monthly long-short returns for a mined population.

    Returns a frame with columns ``signalid``, ``date`` (month start),
    ``ret_ew``, ``ret_vw``, ``nlong``, ``nshort``.  Returns are in percent,
    as distributed.

    ``weighting`` selects which return columns to keep; ``"both"`` keeps
    both, which is what the main results use, because equal- and
    value-weighted versions of the same signal are two separate strategies
    in the counts reported by Chen and Dim.
    """
    ensure_dirs()
    if population not in MINED_FILES:
        raise ValueError(f"population must be one of {sorted(MINED_FILES)}")
    cache = CACHE / f"mined_{population}.parquet"

    if force or not cache.exists():
        returns_file, _ = MINED_FILES[population]
        path = RAW / "htap" / returns_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run scripts/01_download_data.py first"
            )
        df = _read_maybe_gzip(
            path,
            dtype={
                "signalid": np.int32,
                "ret_ew": np.float32,
                "ret_vw": np.float32,
                "nlong": np.int32,
                "nshort": np.int32,
            },
            parse_dates=["date"],
        )
        # The two source files list nlong/nshort in a different column order;
        # select by name so the order in the file cannot matter.
        df = df[["signalid", "date", "ret_ew", "ret_vw", "nlong", "nshort"]]
        df = df.sort_values(["signalid", "date"], kind="mergesort").reset_index(drop=True)
        df.to_parquet(cache, index=False)

    df = pd.read_parquet(cache)
    if weighting == "ew":
        df = df.drop(columns=["ret_vw"])
    elif weighting == "vw":
        df = df.drop(columns=["ret_ew"])
    elif weighting != "both":
        raise ValueError("weighting must be 'ew', 'vw' or 'both'")
    return df


def mined_return_panel(population: str, weighting: str) -> pd.DataFrame:
    """Wide panel: rows are months, columns are signal ids.

    This is the shape every downstream calculation wants (cross-sectional
    moments of t-statistics, correlation spectra, block bootstrap over
    months).  For the ticker population it is about 740 x 19,380 float32,
    roughly 55 MB, which fits comfortably in memory.
    """
    if weighting not in ("ew", "vw"):
        raise ValueError("weighting must be 'ew' or 'vw'")
    df = load_mined_returns(population, weighting=weighting)
    col = f"ret_{weighting}"
    panel = df.pivot(index="date", columns="signalid", values=col)
    panel.columns.name = "signalid"
    panel.index = to_month_start(panel.index)
    return panel.sort_index()


# ----------------------------------------------------------------------
# Open Source Asset Pricing


def download_osap(force: bool = False) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Fetch the published-predictor portfolios and the signal documentation.

    Kept separate from :func:`load_osap_portfolios` so that the network call
    happens exactly once, in ``scripts/01_download_data.py``.
    """
    ensure_dirs()
    port_cache = CACHE / "osap_portfolios.parquet"
    doc_cache = CACHE / "osap_signal_doc.parquet"
    if not force and port_cache.exists() and doc_cache.exists():
        return pd.read_parquet(port_cache), pd.read_parquet(doc_cache)

    import openassetpricing as oap  # imported lazily: only needed to download

    conn = oap.OpenAP()
    ports = conn.dl_port("op", "pandas")
    doc = conn.dl_signal_doc("pandas")
    ports.to_parquet(port_cache, index=False)
    doc.to_parquet(doc_cache, index=False)
    return ports, doc


def load_osap_portfolios() -> pd.DataFrame:
    path = CACHE / "osap_portfolios.parquet"
    if not path.exists():
        raise FileNotFoundError("run scripts/01_download_data.py first")
    return pd.read_parquet(path)


def load_osap_signal_doc() -> pd.DataFrame:
    path = CACHE / "osap_signal_doc.parquet"
    if not path.exists():
        raise FileNotFoundError("run scripts/01_download_data.py first")
    return pd.read_parquet(path)


def osap_longshort_panel(
    sample: str = "original",
    doc: Optional[pd.DataFrame] = None,
    category: str = "Predictor",
) -> pd.DataFrame:
    """Wide panel of published-predictor long-short returns.

    ``PredictorPortsFull.csv`` stores one row per signal, portfolio and month.
    The long-short leg is the row with ``port == 'LS'``.

    ``sample`` controls the window:

    ``"original"``
        each predictor's own in-sample window, as recorded in
        ``SignalDoc.csv`` (``SampleStartYear`` to ``SampleEndYear``).  This
        is the window whose t-statistics the original papers reported, so it
        is the one to use when re-evaluating published claims.
    ``"full"``
        everything available, which mixes in-sample and post-publication
        data.

    ``category`` selects rows of ``SignalDoc.csv`` by ``Cat.Signal``.  The
    file documents 212 ``Predictor`` signals, 114 ``Placebo`` signals -
    characteristics that the original papers did *not* claim predict returns -
    and 5 dropped ones.  ``"Predictor"`` is the published population of the
    paper; ``"Placebo"`` gives a second, independent null population that we
    use as a robustness check on the ticker calibration.  Pass ``"all"`` to
    keep everything.
    """
    ports = load_osap_portfolios()
    ls = ports.loc[ports["port"] == "LS", ["signalname", "date", "ret"]].copy()
    ls["date"] = pd.to_datetime(ls["date"])

    doc = load_osap_signal_doc() if doc is None else doc
    if category != "all":
        keep_names = set(doc.loc[doc["Cat.Signal"] == category, "Acronym"])
        if not keep_names:
            raise ValueError(f"no signals with Cat.Signal == {category!r}")
        ls = ls.loc[ls["signalname"].isin(keep_names)]

    if sample == "full":
        pass
    elif sample == "original":
        win = doc[["Acronym", "SampleStartYear", "SampleEndYear"]].dropna()
        win = win.rename(columns={"Acronym": "signalname"})
        ls = ls.merge(win, on="signalname", how="inner")
        year = ls["date"].dt.year
        keep = (year >= ls["SampleStartYear"]) & (year <= ls["SampleEndYear"])
        ls = ls.loc[keep, ["signalname", "date", "ret"]]
    else:
        raise ValueError("sample must be 'original' or 'full'")

    panel = ls.pivot(index="date", columns="signalname", values="ret")
    panel.index = to_month_start(panel.index)
    return panel.sort_index()


# ----------------------------------------------------------------------
# Factor returns


FF_URLS = {
    "ff5": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    ),
    "mom": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_CSV.zip"
    ),
}


def download_factors(force: bool = False) -> pd.DataFrame:
    """Monthly Fama-French five factors plus momentum, from Ken French's site.

    Returns a frame indexed by month start with columns ``mktrf``, ``smb``,
    ``hml``, ``rmw``, ``cma``, ``umd``, ``rf``, in percent, matching the
    units of the mined and published return series.
    """
    import requests

    ensure_dirs()
    cache = CACHE / "ff_factors.parquet"
    if not force and cache.exists():
        return pd.read_parquet(cache)

    frames = {}
    for name, url in FF_URLS.items():
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("latin-1")
        frames[name] = _parse_french_csv(raw)

    ff5 = frames["ff5"].rename(
        columns={
            "Mkt-RF": "mktrf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "rf",
        }
    )
    mom = frames["mom"]
    mom.columns = [c.strip() for c in mom.columns]
    mom = mom.rename(columns={mom.columns[0]: "umd"})
    out = ff5.join(mom[["umd"]], how="left")
    out.index = to_month_start(out.index)
    out = out.sort_index()
    out.to_parquet(cache)
    return out


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Pull the monthly block out of a Ken French CSV.

    These files start with a licence header, then the monthly table keyed by
    ``YYYYMM``, then an annual table keyed by ``YYYY``.  We keep rows whose
    first field is exactly six digits, which selects the monthly block and
    drops both the header and the annual block without needing to count
    lines.
    """
    rows = []
    header = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        key = parts[0]
        if header is None and key == "" and any(p for p in parts[1:]):
            header = parts[1:]
            continue
        if len(key) == 6 and key.isdigit():
            try:
                vals = [float(p) for p in parts[1 : len(header) + 1]]
            except ValueError:
                continue
            rows.append([key] + vals)
    if header is None or not rows:
        raise ValueError("could not parse Ken French CSV")
    df = pd.DataFrame(rows, columns=["yyyymm"] + header)
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m")
    return df.drop(columns=["yyyymm"]).set_index("date").sort_index()
