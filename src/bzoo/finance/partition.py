"""Splitting the mined strategies into a known-null set and a known-signal set.

Why this file exists
--------------------
The whole finance testbed rests on one claim: the ticker-symbol strategies
contain no real predictability, so their cross-sectional distribution of
t-statistics *is* the null distribution.  If a past-return strategy leaks
into that set, the estimated null inflates, every calibrated threshold gets
stricter, and the paper's headline number is wrong in the direction that
flatters it.  So the classification is written out explicitly, every rule is
logged, and the rules are unit-tested against hand-labelled names.

What the upstream data gives us
-------------------------------
Chen and Dim distribute the two families in separate files
(``TickerSignalsLongShort.csv.gzip`` and
``PastReturnSignalsLongShort.csv.gzip``), each with its own signal-id space
starting at zero.  That is a much better starting point than a single mixed
file, but it also means the ids collide across files, so anything that
pools the two populations must namespace them.  :func:`tagged_names` does
that.

The rules
---------
Ticker names follow the grammar produced by ``get_ticker_signals.py`` in the
authors' replication code::

    L<p>_lng_<a>_<b>_sht_<c>_<d>      p in 1..4,  a,b,c,d in 1..20 distinct

``p`` is the position of the letter of the ticker used for sorting, and the
four numbers are group indices out of twenty.  Past-return names follow::

    <root>_<q1>[_<q2>...]            root in {ret, std, skew, kurt}
                                      q in 1..20 (see below)

We accept a name into the known-null set only if it matches the ticker
grammar exactly, and into the known-signal set only if it matches the
past-return grammar exactly.  Anything else is put in ``"unclassified"`` and
raises, rather than being silently assigned.  A grammar change upstream
should break the pipeline loudly.

What the past-return indices mean
---------------------------------
This is easy to get backwards and getting it backwards inverts every
economic interpretation, so it is written out.  In the authors'
``get_past_return_signals.py`` the five years of monthly returns before the
prediction month are laid out in chronological order and labelled
``[1,1,1,2,2,2,...,20,20,20]``.  So the index is a **non-overlapping
quarter**, and it is ordered **oldest first**:

* ``q = 20`` is the most recent quarter, months ``t-3`` to ``t-1``;
* ``q = 17..20`` is the most recent twelve months;
* ``q = 1`` is the quarter five years back.

The strategy is long the top decile of the signal and short the bottom
decile, so ``ret_17_18_19_20`` is twelve-month momentum with no gap.

Known-signal subsets
--------------------
Three families with established effects and known signs, used as the power
check:

``momentum``      built only from ``q in 17..20``: twelve-month momentum,
                  Jegadeesh and Titman (1993).  Expected sign positive.
``recent``        ``ret_20`` alone: the most recent quarter, which at
                  monthly frequency is dominated by short-term reversal,
                  Jegadeesh (1990).  Expected sign negative.
``longrun``       built only from ``q in 1..8``: returns from three to five
                  years back, long-run reversal, De Bondt and Thaler (1985).
                  Expected sign negative.

A calibration that finds no signal in these has a power problem, and that is
the third sanity check in the plan.  Note that the check is on the *signed*
statistic against the expected sign, not on its magnitude: a family that
came out significant with the wrong sign would mean the index convention had
been misread, which is exactly the error this section guards against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd

TICKER_PATTERN = re.compile(r"^L([1-4])_lng_(\d+)_(\d+)_sht_(\d+)_(\d+)$")
PASTRET_PATTERN = re.compile(r"^(ret|std|skew|kurt)((?:_\d+)+)$")

N_TICKER_GROUPS = 20
N_PASTRET_QUARTERS = 20

# Known-signal families: quarter sets and the sign the literature predicts.
KNOWN_SIGNAL_FAMILIES = {
    "momentum": {"quarters": frozenset(range(17, 21)), "expected_sign": +1},
    "longrun": {"quarters": frozenset(range(1, 9)), "expected_sign": -1},
}
RECENT_QUARTER_SIGNAL = "ret_20"
RECENT_EXPECTED_SIGN = -1

# Every rule below is logged verbatim into DECISIONS.md by
# scripts/02_partition_report.py.  Do not edit one without editing that.
RULES: List[str] = [
    "R1  A name is ticker-based if and only if it matches "
    r"^L([1-4])_lng_(\d+)_(\d+)_sht_(\d+)_(\d+)$.",
    "R2  Its four group indices must be distinct and lie in 1..20; the "
    "generating code draws them as a 4-subset of 20 groups.",
    "R3  A name is past-return-based if and only if it matches "
    r"^(ret|std|skew|kurt)((?:_\d+)+)$.",
    "R4  Its quarter indices must be distinct and lie in 1..20.  The index "
    "is a non-overlapping quarter of the five years before the prediction "
    "month, ordered oldest first, so 20 is the most recent quarter.",
    "R5  Any name matching neither grammar is 'unclassified'; the loader "
    "raises rather than assigning it.",
    "R6  A name matching both grammars would also raise.  The two grammars "
    "are disjoint because rule R1 requires a leading 'L' and rule R3 "
    "requires a leading root token from a closed set, so this cannot "
    "happen; the check is kept so that a future grammar change is caught.",
    "R7  The known-null set is exactly the ticker-based set.  Past-return "
    "strategies are never included in it, in either weighting.",
    "R8  Equal-weighted and value-weighted versions of one signal are two "
    "strategies, following Chen and Dim's own counts, and are kept in "
    "separate panels so that no strategy appears twice within a panel.",
    "R9  Within the past-return set: any 'ret_*' whose quarters all lie in "
    "17..20 is labelled momentum (expected sign positive); 'ret_20' alone "
    "is labelled recent (expected sign negative, short-term reversal); any "
    "'ret_*' whose quarters all lie in 1..8 is labelled longrun (expected "
    "sign negative). These labels are used only for the power check, and "
    "the check is on the signed statistic against the expected sign.",
    "R10 Signal ids restart at 0 in each source file, so pooled analyses "
    "use the namespaced key '<population>:<signalid>'.",
]


@dataclass(frozen=True)
class TickerName:
    letter_position: int
    long_groups: "tuple[int, int]"
    short_groups: "tuple[int, int]"


@dataclass(frozen=True)
class PastReturnName:
    root: str
    quarters: "tuple[int, ...]"


def parse_ticker_name(name: str) -> "TickerName | None":
    """Parse a ticker strategy name, or return ``None`` if it is not one."""
    m = TICKER_PATTERN.match(name)
    if m is None:
        return None
    pos = int(m.group(1))
    groups = tuple(int(m.group(i)) for i in (2, 3, 4, 5))
    if len(set(groups)) != 4:  # R2
        return None
    if any(g < 1 or g > N_TICKER_GROUPS for g in groups):  # R2
        return None
    return TickerName(pos, (groups[0], groups[1]), (groups[2], groups[3]))


def parse_pastret_name(name: str) -> "PastReturnName | None":
    """Parse a past-return strategy name, or return ``None`` if it is not one."""
    m = PASTRET_PATTERN.match(name)
    if m is None:
        return None
    quarters = tuple(int(x) for x in m.group(2).lstrip("_").split("_"))
    if len(set(quarters)) != len(quarters):  # R4
        return None
    if any(q < 1 or q > N_PASTRET_QUARTERS for q in quarters):  # R4
        return None
    return PastReturnName(m.group(1), quarters)


def classify_name(name: str) -> str:
    """Return ``"ticker"``, ``"pastret"`` or ``"unclassified"``."""
    is_ticker = parse_ticker_name(name) is not None
    is_pastret = parse_pastret_name(name) is not None
    if is_ticker and is_pastret:  # R6
        raise ValueError(f"name {name!r} matches both grammars")
    if is_ticker:
        return "ticker"
    if is_pastret:
        return "pastret"
    return "unclassified"  # R5


def classify_names(names: Sequence[str]) -> pd.Series:
    return pd.Series([classify_name(n) for n in names], index=list(names), name="family")


def pastret_label(name: str) -> str:
    """Label a past-return strategy for the power check (rule R9)."""
    parsed = parse_pastret_name(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a past-return strategy name")
    if parsed.root != "ret":
        return "other"
    if name == RECENT_QUARTER_SIGNAL:
        return "recent"
    qs = set(parsed.quarters)
    for label, spec in KNOWN_SIGNAL_FAMILIES.items():
        if qs <= spec["quarters"]:
            return label
    return "other"


def expected_sign(label: str) -> int:
    """Sign the literature predicts for a known-signal family (rule R9)."""
    if label == "recent":
        return RECENT_EXPECTED_SIGN
    if label in KNOWN_SIGNAL_FAMILIES:
        return int(KNOWN_SIGNAL_FAMILIES[label]["expected_sign"])
    raise ValueError(f"no expected sign for label {label!r}")


def partition(names_df: pd.DataFrame, population: str) -> pd.DataFrame:
    """Classify a signal-name table and check it against its source file.

    Parameters
    ----------
    names_df:
        Frame with columns ``signalid`` and ``signalname``, as distributed.
    population:
        ``"ticker"`` or ``"pastret"``: which file the table came from.

    Raises
    ------
    ValueError
        If any name is unclassified, or if any name's family disagrees with
        the file it was distributed in.  Both are leakage, which rule R7
        forbids.
    """
    if population not in ("ticker", "pastret"):
        raise ValueError("population must be 'ticker' or 'pastret'")
    out = names_df.copy()
    out["family"] = [classify_name(n) for n in out["signalname"]]

    bad = out.loc[out["family"] == "unclassified", "signalname"]
    if len(bad):
        raise ValueError(
            f"{len(bad)} names in the {population} file match neither grammar, "
            f"first few: {list(bad.head(5))}"
        )
    wrong = out.loc[out["family"] != population, "signalname"]
    if len(wrong):
        raise ValueError(
            f"{len(wrong)} names in the {population} file were classified as "
            f"the other family, first few: {list(wrong.head(5))}"
        )

    if population == "ticker":
        parsed = [parse_ticker_name(n) for n in out["signalname"]]
        out["letter_position"] = [p.letter_position for p in parsed]
        out["long_groups"] = [p.long_groups for p in parsed]
        out["short_groups"] = [p.short_groups for p in parsed]
        out["role"] = "known_null"
    else:
        parsed = [parse_pastret_name(n) for n in out["signalname"]]
        out["root"] = [p.root for p in parsed]
        out["quarters"] = [p.quarters for p in parsed]
        out["label"] = [pastret_label(n) for n in out["signalname"]]
        out["role"] = "known_signal"

    out["key"] = population + ":" + out["signalid"].astype(str)  # R10
    return out


def partition_summary(parts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per population, for the data section of the paper."""
    rows = []
    for pop, df in parts.items():
        # Spaces rather than underscores: these go straight into a LaTeX table.
        row = {
            "population": pop,
            "n_signals": len(df),
            "role": str(df["role"].iloc[0]).replace("_", " "),
        }
        if pop == "ticker":
            counts = df["letter_position"].value_counts().sort_index()
            row["detail"] = ", ".join(f"L{k}: {v}" for k, v in counts.items())
        else:
            counts = df["label"].value_counts()
            row["detail"] = ", ".join(f"{k}: {v}" for k, v in counts.items())
        rows.append(row)
    return pd.DataFrame(rows)
