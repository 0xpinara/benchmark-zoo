# Overleaf upload folder

Upload this whole folder (or `benchmark-zoo-paper.zip`) to Overleaf and set
**`main.tex`** as the main document. Compiler: **pdfLaTeX**. Nothing else to
set up.

```
main.tex               the paper
neurips_2023.sty       official NeurIPS style file, unmodified
references.bib         bibliography, 69 entries
tables/                16 generated .tex tables, plus macros.tex and
                       macros_alpha.tex
figures/               6 generated .pdf figures, 5 of them used
```

Some of the tables and figures are not `\input` by the current draft. They are
left-overs from the machine learning half, which is parked on a branch. They
are shipped because the pipeline writes them, not because the paper wants them.

## Two things to know before you read it

**Every number is generated, not typed.** `tables/macros.tex` and
`tables/macros_alpha.tex` hold each number that appears in the prose as a
`\newcommand`, and the text uses the macro. So a sentence cannot disagree with
a table. If you see `??` on the page, that is a number the pipeline has not
produced yet, and it is a bug rather than a typo. There are none right now.

**The draft is finance only.** The earlier version had a machine learning half
as well. It is on a separate branch and nothing was deleted from the
repository, only from this document.

## The five figures the paper uses

| file | what it shows |
| --- | --- |
| `null_density.pdf` | the signature figure: measured null against the theoretical one, tail on a log scale |
| `across_models.pdf` | the same population under CAPM, three-factor and five-factor adjustment |
| `exposure_dose_response.pdf` | Pr(\|t\| > 3) against exposure decile |
| `thresholds.pdf` | five thresholds for the same family of tests, on one scale |
| `survival_vs_n.pdf` | how many published predictors survive as the assumed trial count moves |

All of them are readable in greyscale at half column width. Identity is carried
by line style and marker shape rather than colour.

## If a table looks wrong

Do not edit `tables/*.tex`. They are overwritten by
`scripts/09_tables_and_figures.py` and `scripts/13_alpha_tables_and_figures.py`
in the code repository. Fix the script.

## Style file

`neurips_2023.sty` is the official file from
<https://media.neurips.cc/Conferences/NeurIPS2023/Styles/>. Replace it with the
current year's before submitting, and check the anonymisation policy, because
the Datasets and Benchmarks track has changed it before. `main.tex` loads it
with `[preprint]`, which shows the author block and turns line numbers off. For
a double-blind submission switch to plain `\usepackage{neurips_2023}` and
replace the repository URL in the abstract at the same time, or the author is
recoverable from the link.
