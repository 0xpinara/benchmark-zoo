# bzoo

A multiplicity correction needs two inputs: how many trials were run, and the
null distribution of the statistic you screened on. The second one is almost
always assumed rather than measured. `bzoo` measures it, on a population where
the answer can be checked against ground truth, and then runs the corrections
with the measured null instead of the textbook one.

The population is 19,380 long-short equity strategies built from the letters of
ticker symbols, from Chen and Dim's high-throughput data. A strategy that buys
firms whose ticker starts in one letter range and sells firms in another has no
economic content by construction, so every rejection on it is false and you can
just count them.

## The finding

The raw mean-return t-statistics on that population are close to standard
normal (sd 1.03). The five-factor alpha t-statistics are not (sd 1.40), and
5.2 percent of them cross |t| > 3 where a normal says 0.27 percent.

This is not a multiplicity problem, which is why it matters. OLS gives
`alpha = rbar - beta'fbar` exactly, so a strategy that bears risk and earns
nothing has a true alpha of `-beta'mu_f`, which is not zero. Raising the
threshold for the number of trials does not remove those alphas, because they
are real. They are just not interesting. The fix is the one data mining already
found when it replaced confidence with lift: re-centre the statistic, or
calibrate it on the population you are actually searching.

## Install

```bash
git clone https://github.com/0xpinara/benchmark-zoo && cd benchmark-zoo
make env      # venv + pip install -e ".[dev,ml]"
make test     # the test suite, no data needed
```

Python 3.9 or newer. The statistics need numpy, pandas, scipy and statsmodels.
torch and ogb are only for the benchmark sweep and live in the `ml` extra.

## A corrected threshold in four lines

```python
from bzoo.corrections import deflated_improvement

deflated_improvement(delta_obs=0.006, sigma_delta=0.0021, n_trials=1000)
# {'expected_max': 0.0068, 'threshold': 0.0082, 'p_value': 0.8823,
#  'survives': False, ...}
```

A reported gain of 0.6 accuracy points does not survive once you credit the
field with a thousand attempts and the null spread of the gain is 0.21 points.
`sigma_delta` has no default and is never guessed. It has to come from a null
population in the domain you are working in, which is the whole point.

## The resampling design

This is the first thing anyone who knows the area will check, so it goes here
rather than in an appendix.

* Finance metrics are averages over time, so the bootstrap resamples time
  blocks: Politis and Romano (1994), block length by the rule in Politis and
  White (2004).
* Benchmark metrics are averages over instances, so the bootstrap resamples
  instances.
* Every model shares the same resampled instance set inside a replicate.
  Resampling each model separately makes the columns independent by
  construction, which inflates the null maximum and makes every correction too
  conservative. `bzoo.resample.instance` still exposes the wrong version as
  `independent_instance_indices`, so the robustness section can measure how big
  the mistake is instead of asserting it. Nothing in the main results calls it.
* Permutation nulls flip the sign of whole blocks, one sign vector shared
  across the cross-section, so both the dependence and the autocorrelation
  survive.

## Corrections

Each one has a test against a published worked example or a published property
of the procedure, and a docstring naming the equation it follows.

| method | controls | dependence | source |
| --- | --- | --- | --- |
| `bonferroni` | FWER | none | Bonferroni (1936) |
| `sidak` | FWER | none (exact under independence) | Šidák (1967) |
| `holm` | FWER, step-down | arbitrary | Holm (1979) |
| `benjamini_hochberg` | FDR, step-up | independence or PRDS | Benjamini and Hochberg (1995) |
| `benjamini_yekutieli` | FDR, step-up | arbitrary | Benjamini and Yekutieli (2001) |
| `storey_qvalues` | pFDR | weak | Storey (2002) |
| `white_reality_check` | FWER, bootstrap | learned from resampling | White (2000) |
| `hansen_spa` | FWER, studentised and recentred | learned from resampling | Hansen (2005) |
| `romano_wolf` | FWER, step-down bootstrap | learned from resampling | Romano and Wolf (2005) |
| `westfall_young_maxt`, `westfall_young_minp` | FWER, permutation | native | Westfall and Young (1993) |
| `harvey_liu_zhu`, `haircuts` | haircuts on a reported Sharpe | partial | Harvey, Liu and Zhu (2016) |
| `deflated_improvement`, `deflated_sharpe_ratio` | single trial, via `N` and `sigma` | through the effective `N` | Bailey and López de Prado (2014) |

Cases the tests pin down, as examples: the fifteen p-values from Benjamini and
Hochberg's 1995 worked example, where the family-wise procedures reject three
and the step-up procedure rejects four; the ordering
`p_lower <= p_consistent <= p_upper` for Hansen's three SPA p-values; and
Hansen's point that adding bad candidates hurts the Reality Check more than it
hurts SPA.

## Reproducing the paper

```bash
make fetch    # 1.8 GB of mined strategy returns, once
make all      # every table and figure, from the raw sources
```

Stages run on their own too (`make sanity`, `make calibrate`, `make sweep`,
`make mlanalysis`, `make robust`, `make nulltest`, `make tables`), and each one
only reads what an earlier one wrote. `make help` lists them. The sweep is the
slow part: the three small graphs take under an hour on eight cores, ogbn-arxiv
takes a few hours.

The three checks in `make sanity` have to pass before anything downstream runs.
The published t-statistics reproduce from the distributed returns, the
known-null population's aggregate mean is indistinguishable from zero, and the
known-signal families come out with the sign the literature predicts.

## Layout

```
src/bzoo/
  finance/     OSAP and mined-strategy loading, the known-null partition,
               performance statistics, factor models
  ml/          node classification datasets, models and ablations,
               budget parity, null construction
  null/        empirical null estimation, GPD tail fits, dependence
  resample/    stationary bootstrap, joint instance bootstrap, permutation
  corrections/ the twelve methods above, behind one interface
  report/      LaTeX table and figure emission
scripts/       01 to 14, in order; every number in the paper comes from these
tests/         pytest, one file per module group
paper/         LaTeX source; tables/ and figures/ are generated, never edited
DECISIONS.md   dated log of every non-obvious choice, the wrong ones included
DATASHEET.md   provenance and limitations of each released artifact
```

Three rules the code enforces rather than asks for. `ml/tuning.py` raises if any
architecture got a different trial count, seed set or configuration set from any
other. Every stochastic procedure takes an explicit seed. And `report/tables.py`
writes the `.tex` the paper `\input`s, so no number is ever transcribed by hand.

## Two things I got wrong first

Both are in `DECISIONS.md` with dates, and both are worth knowing if you build
on this.

1. The indices in Chen and Dim's past-return signal names are quarters ordered
   oldest first, not month lags. Read the wrong way, twelve-month momentum comes
   out at t = -2.42. Read the right way it is +5.60, and every other known
   effect lands with the sign the literature predicts.
2. A population that is null for one statistic need not be null for a
   transformation of it. The ticker strategies have zero expected return by
   construction but non-zero factor loadings, and OLS gives
   `alpha = rbar - beta'fbar` exactly, so their alphas are not zero. Between 29
   and 41 percent of the cross-sectional variance of those alphas comes from the
   exposures. So the alpha null has to come from a permutation that imposes
   `alpha = 0` given the loadings, not from the marginal spread.

## Licence and data

Code is MIT. Only derived statistics are redistributed here. The mined strategy
returns and the OSAP portfolios are published by their own authors and pulled by
`make fetch`; Planetoid and OGB are openly licensed and pulled by the sweep; the
Fama-French factors come from the Ken French library and are not redistributed.
The one file kept in the repository is
`data/raw/leaderboards/ogbn_arxiv_leaderboard.csv`, a dated transcription of a
public web page, so the analysis still reproduces after the page changes. See
`DATASHEET.md`.

## Citation

```bibtex
@misc{aksoy2026screening,
  title  = {Screening on the Wrong Null: When a No-Content Hypothesis
            Does Not Imply a Zero Statistic},
  author = {Pinar Aksoy},
  year   = {2026},
  note   = {https://github.com/0xpinara/benchmark-zoo}
}
```
