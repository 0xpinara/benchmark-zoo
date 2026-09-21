# Overleaf upload folder

Upload this whole folder (or `benchmark-zoo-paper.zip`) to Overleaf and set
**`main.tex`** as the main document. Compiler: **pdfLaTeX**. No other setup.

```
main.tex               the paper
neurips_2023.sty       official NeurIPS style file, unmodified
references.bib         bibliography, 50 entries
tables/                13 generated .tex tables, plus macros.tex
figures/               4 generated .pdf figures
```

## Two things to know before you read it

**Every number is generated, not typed.** `tables/macros.tex` holds each number
that appears in the prose as a `\newcommand`, and the text uses the macro. So a
sentence cannot disagree with a table. If you see `??` on the page, that is a
number the pipeline has not produced yet and it is a bug, not a typo. There are
none at the moment.

**The draft is incomplete and says so.** Section "Status of this draft, and what
is still missing" lists what is finished, what is running, and what is planned.
The short version: the finance half is complete; the machine learning half is
complete on three of four benchmarks, and the ogbn-arxiv sweep is still running,
so the leaderboard is reported descriptively rather than deflated.

## The four figures

| file | what it shows |
| --- | --- |
| `null_density.pdf` | the signature figure: measured null against the theoretical one, with the tail on a log scale |
| `thresholds.pdf` | five thresholds for the same family of tests, on one scale |
| `survival_vs_n.pdf` | how many published predictors survive as the assumed trial count moves |
| `saturation.pdf` | null spread against remaining headroom, one point per benchmark |

All four are readable in greyscale at half column width; identity is carried by
line style and marker shape rather than colour.

## If a table looks wrong

Do not edit `tables/*.tex` — they are overwritten by
`scripts/09_tables_and_figures.py` in the code repository. Fix the script.

## Style file

`neurips_2023.sty` is the official file from
<https://media.neurips.cc/Conferences/NeurIPS2023/Styles/>. Replace it with the
current year's before submitting, and check the anonymisation policy: the
Datasets and Benchmarks track has changed it before. `main.tex` currently loads
it with `[preprint]`, which shows the author names and turns line numbers off.
Switch to plain `\usepackage{neurips_2023}` for a camera-ready or anonymous
submission.
