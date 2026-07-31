"""Assemble a self-contained folder to upload to Overleaf, and check it.

Overleaf has no access to this repository, so the paper has to travel with
everything it needs: the style file, the bibliography, every generated table and
every generated figure. This script copies them into ``overleaf/``, validates the
result statically, and zips it.

The validation is the point. There is no LaTeX installation in the environment
where the pipeline runs, so we cannot compile the paper to find out whether it
works. What we can do is check the things that actually break a compile:

* every ``\\input`` resolves to a file that exists;
* every ``\\includegraphics`` resolves to a PDF that exists;
* every ``\\bzoo`` macro the text uses is defined, and none is still the ``??``
  placeholder that ``scripts/09_tables_and_figures.py`` inserts for a number the
  pipeline has not produced;
* every citation key is in ``references.bib``;
* every ``\\ref`` points at a label that exists somewhere in the paper or in a
  generated table;
* the environments are balanced.

That catches every failure we have actually hit. It does not catch a genuine
LaTeX syntax error inside a macro body, and it is not a substitute for compiling
once before submitting.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List

from bzoo.paths import FIGURES, PAPER, ROOT, TABLES

OUT = ROOT / "overleaf"
ZIP = ROOT / "benchmark-zoo-paper.zip"


def assemble() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tables").mkdir(parents=True)
    (OUT / "figures").mkdir(parents=True)

    for name in ("main.tex", "references.bib"):
        shutil.copy2(PAPER / name, OUT / name)
    style = list(PAPER.glob("neurips*.sty"))
    if not style:
        raise FileNotFoundError("no neurips*.sty in paper/")
    for st in style:
        shutil.copy2(st, OUT / st.name)
    for t in sorted(TABLES.glob("*.tex")):
        shutil.copy2(t, OUT / "tables" / t.name)
    for f in sorted(FIGURES.glob("*.pdf")):
        shutil.copy2(f, OUT / "figures" / f.name)
    readme = PAPER.parent / "overleaf_README.md"
    if readme.exists():
        shutil.copy2(readme, OUT / "README.md")


def validate() -> List[str]:
    main = (OUT / "main.tex").read_text()
    bib = (OUT / "references.bib").read_text()
    # Two macro files, both inputted by main.tex: macros.tex from script 09
    # and macros_alpha.tex from script 13.  Concatenated here so the undefined
    # and placeholder checks below see all of them.
    macros = ""
    for name in ("macros.tex", "macros_alpha.tex"):
        path = OUT / "tables" / name
        if path.exists():
            macros += path.read_text() + "\n"
    problems: List[str] = []

    for f in re.findall(r"\\input\{([^}]+)\}", main):
        if not (OUT / (f + ".tex")).exists() and not (OUT / f).exists():
            problems.append(f"missing \\input: {f}")
    for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", main):
        if not (OUT / (f + ".pdf")).exists():
            problems.append(f"missing figure: {f}")

    used = set(re.findall(r"\\bzoo([A-Za-z]+)", main))
    defined = set(re.findall(r"\\newcommand\{\\bzoo([A-Za-z]+)\}", macros))
    problems += [f"undefined macro: \\bzoo{m}" for m in sorted(used - defined)]
    problems += [
        f"macro is still the ?? placeholder: \\bzoo{m}"
        for m in sorted(used)
        if re.search(r"\\newcommand\{\\bzoo" + re.escape(m) + r"\}\{\?\?", macros)
    ]

    keys = set()
    for c in re.findall(r"\\cite[a-z]*\{([^}]+)\}", main):
        keys.update(k.strip() for k in c.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    problems += [f"citation not in bib: {k}" for k in sorted(keys - bib_keys)]

    labels = set(re.findall(r"\\label\{([^}]+)\}", main))
    for t in (OUT / "tables").glob("*.tex"):
        labels.update(re.findall(r"\\label\{([^}]+)\}", t.read_text()))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", main))
    problems += [f"\\ref to a missing label: {r}" for r in sorted(refs - labels)]

    for env in ("table", "figure", "tabular", "enumerate", "itemize", "abstract"):
        n_open = len(re.findall(r"\\begin\{" + env + r"\}", main))
        n_close = len(re.findall(r"\\end\{" + env + r"\}", main))
        if n_open != n_close:
            problems.append(f"unbalanced {env}: {n_open} begin, {n_close} end")

    if "PLACEHOLDER" in "".join(
        p.read_text() for p in (OUT / "tables").glob("*.tex")
    ):
        problems.append("at least one table is still a placeholder")

    print(f"macros: {len(used)} used, {len(defined)} defined")
    print(f"citations: {len(keys)} cited, {len(bib_keys)} in bibliography")
    print(f"cross-references: {len(refs)} used, {len(labels)} labels available")
    return problems


def make_zip() -> None:
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                zf.write(path, path.relative_to(ROOT))
    print(f"wrote {ZIP} ({ZIP.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-problems",
        action="store_true",
        help="assemble and zip even if validation fails",
    )
    args = ap.parse_args()

    assemble()
    problems = validate()
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        if not args.allow_problems:
            print("\nnot zipping; fix these or pass --allow-problems")
            return 1
    else:
        print("\nall checks pass")
    make_zip()
    return 0


if __name__ == "__main__":
    sys.exit(main())
