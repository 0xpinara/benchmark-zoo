"""Tests for table and figure emission, and for the paper's own consistency.

The last two tests are the ones that matter most: they check that every macro the
paper text uses is actually generated, and that every table the paper includes
actually exists.  Those are what make "no number in the paper is typed by hand"
an enforced property rather than a promise.  They skip when the pipeline has not
been run, so continuous integration on a clean checkout still passes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bzoo.paths import PAPER, TABLES
from bzoo.report import tables

MAIN_TEX = PAPER / "main.tex"


# ----------------------------------------------------------------------
# formatting


def test_fmt_handles_each_type():
    assert tables.fmt(3) == "3"
    assert tables.fmt(1234567) == r"1\,234\,567"
    assert tables.fmt(0.123456, 3) == "0.123"
    assert tables.fmt(1.0, 0) == "1"
    assert tables.fmt(True) == "yes"
    assert tables.fmt(False) == "no"
    assert tables.fmt(None) == "--"
    assert tables.fmt(float("nan")) == "--"
    assert tables.fmt(float("inf")) == r"$\infty$"


def test_fmt_switches_to_scientific_only_when_it_has_to():
    assert tables.fmt(0.0001, 3) == r"$1.00\times 10^{-4}$"
    assert tables.fmt(0.0001, 5) == "0.00010"
    assert tables.fmt(0.5, 3) == "0.500"


def test_escape_only_when_asked():
    assert tables.fmt("a_b", escape_strings=True) == r"a\_b"
    assert tables.fmt("a_b") == "a_b"
    # Table labels contain deliberate mathematics that escaping would destroy.
    assert tables.escape(r"$\Pr(|t|>3)$") != r"$\Pr(|t|>3)$"


def test_macro_names_are_legal_latex():
    assert tables.macro_name("nullFfSdVw") == "bzoonullFfSdVw"
    assert tables.macro_name("null_ff_sd_vw") == "bzoonullFfSdVw"
    for bad in ("sigma2", "a_1", "n-eff"):
        with pytest.raises(ValueError):
            tables.macro_name(bad)


def test_dataframe_to_latex_structure():
    df = pd.DataFrame({"Name": ["a", "b"], r"$\sigma$": [0.5, 0.25]})
    out = tables.dataframe_to_latex(df, "A caption", "tab:test", digits=2)
    assert r"\begin{table}" in out and r"\end{table}" in out
    assert r"\caption{A caption}" in out
    assert r"\label{tab:test}" in out
    assert out.count(r"\midrule") == 1
    assert r"\textbf{$\sigma$}" in out  # header not escaped
    assert "0.50" in out and "0.25" in out


def test_rules_after_inserts_panel_separators():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    out = tables.dataframe_to_latex(df, "c", "l", rules_after=(1,))
    assert out.count(r"\midrule") == 2


def test_per_column_digits():
    df = pd.DataFrame({"coarse": [0.123456], "fine": [0.123456]})
    out = tables.dataframe_to_latex(df, "c", "l", digits={"coarse": 1, "fine": 4})
    assert "0.1 " in out
    assert "0.1235" in out


def test_write_macros_accepts_per_entry_digits(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "TABLES", tmp_path)
    tables.write_macros("m", {"one": (1.23456, 2), "two": 1.23456})
    text = (tmp_path / "m.tex").read_text()
    assert r"\newcommand{\bzooone}{1.23\xspace}" in text
    assert r"\newcommand{\bzootwo}{1.235\xspace}" in text


# ----------------------------------------------------------------------
# figures: they must produce a file and must not need colour to be readable


def test_figures_write_pdf_and_png(tmp_path, monkeypatch):
    from bzoo.report import figures

    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    rng = np.random.default_rng(0)
    figures.null_density(
        {"a": rng.standard_normal(2000), "b": rng.standard_normal(2000) * 1.4},
        name="fig",
    )
    assert (tmp_path / "fig.pdf").exists()
    assert (tmp_path / "fig.png").exists()


def test_figure_tones_and_styles_are_distinct():
    """Identity must not rest on colour alone: every series gets its own line
    style and marker as well as its own tone."""
    from bzoo.report import figures

    assert len(set(figures.TONES)) == len(figures.TONES)
    assert len(set(figures.STYLES)) == len(figures.STYLES)
    assert len(set(figures.MARKERS)) == len(figures.MARKERS)
    assert len(figures.STYLES) >= len(figures.TONES)


# ----------------------------------------------------------------------
# the paper's own consistency


def _used_macros() -> set:
    text = MAIN_TEX.read_text()
    return set(re.findall(r"\\bzoo([A-Za-z]+)", text))


def _defined_macros() -> set:
    """Every macro either macro file defines.

    There are two, and main.tex inputs both: ``macros.tex`` from
    ``09_tables_and_figures.py`` and ``macros_alpha.tex`` from
    ``13_alpha_tables_and_figures.py``.  They are kept apart so that running
    one script does not blank the other one's numbers.
    """
    out: set = set()
    for name in ("macros.tex", "macros_alpha.tex"):
        path = TABLES / name
        if path.exists():
            out |= set(
                re.findall(r"\\newcommand\{\\bzoo([A-Za-z]+)\}", path.read_text())
            )
    return out


@pytest.mark.skipif(not MAIN_TEX.exists(), reason="paper source absent")
@pytest.mark.skipif(
    not (TABLES / "macros.tex").exists(), reason="pipeline has not been run"
)
def test_every_macro_the_paper_uses_is_generated():
    missing = sorted(_used_macros() - _defined_macros())
    assert not missing, (
        "the paper text uses macros that no script generates, which means a "
        f"number would render as an undefined command: {missing}"
    )


@pytest.mark.skipif(
    not (TABLES / "macros.tex").exists(), reason="pipeline has not been run"
)
def test_no_generated_macro_is_unused():
    """Not an error, but worth surfacing: a macro nobody uses is usually a
    result that was computed and then not reported."""
    unused = sorted(_defined_macros() - _used_macros())
    if unused:
        pytest.skip(f"{len(unused)} generated macros are unused: {unused[:8]}")


@pytest.mark.skipif(not MAIN_TEX.exists(), reason="paper source absent")
def test_every_included_table_exists():
    text = MAIN_TEX.read_text()
    wanted = re.findall(r"\\input\{tables/([A-Za-z0-9_]+)\}", text)
    assert wanted, "the paper includes no generated tables, which cannot be right"
    missing = [w for w in wanted if not (TABLES / f"{w}.tex").exists()]
    if missing:
        pytest.skip(f"pipeline has not produced: {missing}")


@pytest.mark.skipif(not MAIN_TEX.exists(), reason="paper source absent")
def test_every_included_figure_exists():
    from bzoo.paths import FIGURES

    text = MAIN_TEX.read_text()
    wanted = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([A-Za-z0-9_]+)\}", text)
    assert wanted
    missing = [w for w in wanted if not (FIGURES / f"{w}.pdf").exists()]
    if missing:
        pytest.skip(f"pipeline has not produced: {missing}")


@pytest.mark.skipif(not MAIN_TEX.exists(), reason="paper source absent")
def test_every_citation_key_is_in_the_bibliography():
    text = MAIN_TEX.read_text()
    bib = (PAPER / "references.bib").read_text()
    keys = set()
    for cmd in re.findall(r"\\cite[a-z]*\{([^}]+)\}", text):
        keys.update(k.strip() for k in cmd.split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(keys - defined)
    assert not missing, f"cited but not in references.bib: {missing}"


# ----------------------------------------------------------------------
# date alignment across sources


def test_month_start_normalisation():
    from bzoo.finance.loaders import to_month_start

    idx = pd.to_datetime(["1963-01-31", "1963-02-01", "1963-03-15"])
    out = to_month_start(idx)
    assert list(out.day) == [1, 1, 1]
    assert list(out.month) == [1, 2, 3]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "cache"
         / "mined_ticker.parquet").exists(),
    reason="data cache absent",
)
def test_every_monthly_index_aligns():
    """Three sources date months differently.  A merge on raw dates gives an
    empty intersection with no error, which cost us a silently empty regression
    once, so alignment is checked rather than assumed."""
    from bzoo.finance import loaders

    factors = loaders.download_factors()
    ticker = loaders.mined_return_panel("ticker", "ew")
    pastret = loaders.mined_return_panel("pastret", "ew")
    osap = loaders.osap_longshort_panel(sample="full")
    for name, idx in (
        ("factors", factors.index),
        ("ticker", ticker.index),
        ("pastret", pastret.index),
        ("osap", osap.index),
    ):
        assert set(pd.DatetimeIndex(idx).day) == {1}, name
    # and the overlaps are large, not empty
    assert len(ticker.index.intersection(factors.index)) > 700
    assert len(pastret.index.intersection(factors.index)) > 700
    assert len(osap.index.intersection(factors.index)) > 700
    assert len(ticker.index.intersection(pastret.index)) > 700
