"""Hand-labelled tests for the known-null / known-signal split.

Every example below was written down by reading the two signal-name files
and the authors' generating code, before the classifier was run on them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bzoo.finance import partition as P

TICKER_EXAMPLES = [
    "L1_lng_1_2_sht_3_4",
    "L2_lng_1_2_sht_3_5",
    "L3_lng_17_18_sht_19_20",
    "L4_lng_1_20_sht_2_19",
]

PASTRET_EXAMPLES = [
    "ret_1",
    "ret_20",
    "ret_1_2",
    "ret_1_2_3",
    "ret_16_18_19_20",
    "std_1_2_3_4",
    "skew_1_2_3_5",
    "kurt_17_18_19_20",
]

NEITHER = [
    "",
    "L5_lng_1_2_sht_3_4",  # only four letter positions exist
    "L0_lng_1_2_sht_3_4",
    "L1_lng_1_2_sht_3",  # too few groups
    "L1_lng_1_2_sht_3_4_5",  # too many
    "L1_lng_1_1_sht_2_3",  # repeated group index
    "L1_lng_1_2_sht_2_3",  # group both long and short
    "L1_lng_1_2_sht_3_21",  # group index out of range
    "ret",  # no lags
    "ret_0",  # quarter zero does not exist
    "ret_21",  # quarter out of range
    "ret_1_1",  # repeated quarter
    "vol_1_2",  # root not in the closed set
    "Ret_1",  # roots are lower case in the file
    "L1_LNG_1_2_SHT_3_4",
]


@pytest.mark.parametrize("name", TICKER_EXAMPLES)
def test_ticker_names_are_ticker(name):
    assert P.classify_name(name) == "ticker"


@pytest.mark.parametrize("name", PASTRET_EXAMPLES)
def test_pastret_names_are_pastret(name):
    assert P.classify_name(name) == "pastret"


@pytest.mark.parametrize("name", NEITHER)
def test_unclassified_names(name):
    assert P.classify_name(name) == "unclassified"


def test_ticker_parse_fields():
    p = P.parse_ticker_name("L3_lng_4_7_sht_11_20")
    assert p.letter_position == 3
    assert p.long_groups == (4, 7)
    assert p.short_groups == (11, 20)


def test_pastret_parse_fields():
    p = P.parse_pastret_name("skew_2_5_9")
    assert p.root == "skew"
    assert p.quarters == (2, 5, 9)


def test_power_labels():
    """Indices are quarters ordered oldest first, so 17-20 is the most recent
    twelve months and 1-8 is three to five years back."""
    assert P.pastret_label("ret_17_18_19_20") == "momentum"
    assert P.pastret_label("ret_18") == "momentum"
    assert P.pastret_label("ret_20") == "recent"  # most recent quarter
    assert P.pastret_label("ret_1_2_3_4") == "longrun"
    assert P.pastret_label("ret_5_6_7_8") == "longrun"
    # spanning both ends is neither family
    assert P.pastret_label("ret_1_2_19_20") == "other"
    assert P.pastret_label("ret_9_10_11_12") == "other"
    # only the 'ret' root is a return strategy
    assert P.pastret_label("std_17_18_19_20") == "other"


def test_expected_signs_follow_the_literature():
    assert P.expected_sign("momentum") == +1
    assert P.expected_sign("recent") == -1
    assert P.expected_sign("longrun") == -1
    with pytest.raises(ValueError):
        P.expected_sign("other")


def test_recent_label_beats_the_momentum_label():
    """ret_20 lies inside the momentum window but is labelled 'recent', because
    a one-quarter signal at monthly frequency is a reversal strategy."""
    assert P.pastret_label("ret_20") == "recent"
    assert P.pastret_label("ret_19") == "momentum"


def test_partition_accepts_a_clean_ticker_table():
    df = pd.DataFrame({"signalid": range(len(TICKER_EXAMPLES)), "signalname": TICKER_EXAMPLES})
    out = P.partition(df, "ticker")
    assert (out["role"] == "known_null").all()
    assert out["letter_position"].tolist() == [1, 2, 3, 4]
    assert out["key"].tolist() == [f"ticker:{i}" for i in range(4)]


def test_partition_rejects_leakage_into_the_null_set():
    """A past-return name in the ticker file must raise, not be assigned."""
    names = TICKER_EXAMPLES + ["ret_2_3_4_5"]
    df = pd.DataFrame({"signalid": range(len(names)), "signalname": names})
    with pytest.raises(ValueError, match="classified as the other family"):
        P.partition(df, "ticker")


def test_partition_rejects_unparseable_names():
    names = TICKER_EXAMPLES + ["L9_lng_1_2_sht_3_4"]
    df = pd.DataFrame({"signalid": range(len(names)), "signalname": names})
    with pytest.raises(ValueError, match="match neither grammar"):
        P.partition(df, "ticker")


def test_grammars_are_disjoint_on_every_example():
    for name in TICKER_EXAMPLES + PASTRET_EXAMPLES + NEITHER:
        n_match = sum(
            x is not None
            for x in (P.parse_ticker_name(name), P.parse_pastret_name(name))
        )
        assert n_match <= 1, name


def test_rules_are_documented():
    """Every rule id referenced in the code must exist in RULES."""
    text = " ".join(P.RULES)
    for rid in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"):
        assert rid in text
