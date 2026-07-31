# bzoo — multiplicity corrections with a null you measure instead of assume

Every correction for multiple testing needs two inputs: how many trials were
run, and what the null distribution of the statistic is. In practice both are
supplied by assumption. `bzoo` measures the second one, on a population where
the answer can be checked against ground truth, and then applies the corrections
with the measured null in place of the textbook one.

The package has two halves that share one statistical core.

* **Finance is the validation domain.** The Open Source Asset Pricing project and
  Chen and Dim's high-throughput data include roughly 19,000 long–short
  strategies built from the letters of ticker symbols. A strategy that buys firms
  whose ticker starts in one letter range and sells firms in another has no
  economic content *by construction*. Its statistics are therefore a sample from
  the null, so a calibration can be estimated and then verified.
* **Machine learning is the application domain.** Here the null population has to
  be built — random hyperparameter draws, seed variation, ablated architectures —
  and cannot be verified as cleanly. But leaderboards make the number of trials
  partly *observable*, which the finance literature cannot do at all.

## Install

```bash
git clone https://github.com/ANONYMISED/benchmark-zoo && cd benchmark-zoo
make env            # python3 -m venv .venv, then pip install -e ".[dev,ml]"
make test           # the test suite, no data needed
```

Python 3.9 or newer. The statistical core needs only numpy, pandas, scipy and
statsmodels; torch and ogb are needed only for the benchmark sweep and are in the
`ml` extra.

## Five lines that produce a corrected threshold

```python
import numpy as np
from bzoo.corrections import deflated_improvement
from bzoo.null import variance_inflation

null_t = np.load("data/interim/maxt_perm_ticker_vw.npy")   # a measured null
print(variance_inflation(null_t))                          # scale of that null
print(deflated_improvement(delta_obs=0.006, sigma_delta=0.0021, n_trials=1000))
# {'threshold': 0.0071, 'p_value': 0.121, 'survives': False, ...}
```

A reported gain of 0.6 accuracy points does not survive once the field is
credited with a thousand trials and the null spread of the gain is 0.21 points.
`sigma_delta` has no default and is never guessed: it is estimated from a null
population in the domain at hand, which is the whole point.

## The resampling design, stated up front

This is the first thing a knowledgeable reader checks, so it is here rather than
buried.

* **Finance metrics are averages over time**, so the bootstrap resamples time
  blocks — the stationary bootstrap of Politis and Romano (1994), with the block
  length chosen by the automatic rule of Politis and White (2004).
* **Benchmark metrics are averages over instances**, so the bootstrap resamples
  instances.
* **All models share the same resampled instance set within a replicate.** This
  is not a detail. Independent per-model resampling makes the model columns
  independent by construction, so the null maximum comes out too large and every
  correction too conservative. `bzoo.resample.instance` exposes the wrong version
  explicitly, as `independent_instance_indices`, so that the robustness section
  can *measure* the size of the mistake rather than assert it. Nothing in the
  main results calls it.
* **Permutation nulls flip the sign of whole blocks**, with one sign vector
  shared across the cross-section, which keeps both the dependence and the
  autocorrelation.

## Corrections implemented

Each has a unit test against a published worked example or a published property
of the procedure, and a docstring that names the equation it follows.

| method | controls | dependence | source |
| --- | --- | --- | --- |
| `bonferroni` | FWER | none | Bonferroni (1936) |
| `sidak` | FWER | none (exact under independence) | Šidák (1967) |
| `holm` | FWER, step-down | arbitrary | Holm (1979) |
| `benjamini_hochberg` | FDR, step-up | independence or PRDS | Benjamini and Hochberg (1995) |
| `benjamini_yekutieli` | FDR, step-up | arbitrary | Benjamini and Yekutieli (2001) |
| `storey_qvalues` | pFDR | weak | Storey (2002); Storey, Taylor and Siegmund (2004) |
| `white_reality_check` | FWER, bootstrap | learned from resampling | White (2000) |
| `hansen_spa` | FWER, studentised and recentred | learned from resampling | Hansen (2005) |
| `romano_wolf` | FWER, step-down bootstrap | learned from resampling | Romano and Wolf (2005) |
| `westfall_young_maxt`, `westfall_young_minp` | FWER, permutation | native | Westfall and Young (1993) |
| `harvey_liu_zhu`, `haircuts` | haircuts on a reported Sharpe ratio | partial | Harvey, Liu and Zhu (2016); Harvey and Liu (2015) |
| `deflated_improvement`, `deflated_sharpe_ratio` | single trial, via `N` and `sigma` | through the effective `N` | Bailey and López de Prado (2014) |

Reference cases the tests pin down, as examples: the fifteen p-values of
Benjamini and Hochberg's (1995) worked example, where the family-wise procedures
reject three and the step-up procedure rejects four; the ordering
`p_lower ≤ p_consistent ≤ p_upper` of Hansen's three SPA p-values; and Hansen's
central point that adding inferior candidates damages the Reality Check more
than it damages SPA.

## Reproducing the paper

```bash
make fetch          # 1.8 GB of mined strategy returns, once
make all            # every table and figure, from raw sources
```

Stages can be run individually — `make sanity`, `make calibrate`, `make sweep`,
`make mlanalysis`, `make robust`, `make tables` — and each reads only what an
earlier one wrote. `make help` lists them. The benchmark sweep is the expensive
stage: the three small graphs take under an hour on eight CPU cores, ogbn-arxiv
takes a few hours.

The three sanity checks in `make sanity` must pass before anything downstream
runs: the published t-statistics are reproduced from the distributed returns, the
known-null population's aggregate mean is indistinguishable from zero, and the
known-signal families appear with the sign the literature predicts.

## Repository layout

```
src/bzoo/
  finance/     OSAP and mined-strategy loading, the known-null partition,
               performance statistics, factor models
  ml/          node classification datasets, models and their ablations,
               budget-parity enforcement, null construction
  null/        empirical null estimation, GPD tail fits, dependence
  resample/    stationary bootstrap, joint instance bootstrap, permutation
  corrections/ the twelve methods above, behind one interface
  report/      LaTeX table and figure emission
scripts/       01 to 09, run in order; every number in the paper comes from these
tests/         pytest, one file per module group
paper/         LaTeX source; tables/ and figures/ are generated, never edited
DECISIONS.md   dated log of every non-obvious choice, including the wrong ones
DATASHEET.md   provenance, construction and limitations of each released artifact
```

Three rules the code enforces rather than asks for: `ml/tuning.py` raises if any
architecture received a different trial count, seed set or configuration set from
any other; every stochastic procedure takes an explicit seed; and
`report/tables.py` writes `.tex` that the paper `\input`s, so no number is
transcribed by hand.

## Two things we got wrong first

Both are in `DECISIONS.md` with dates, and both are worth knowing if you build on
this.

1. **The indices in Chen and Dim's past-return signal names are quarters ordered
   oldest first, not month lags.** Under the wrong reading, twelve-month momentum
   comes out at t = −2.42. Under the right one it is +5.60 and every other
   known effect lands with the sign the literature predicts.
2. **A population that is null with respect to one statistic need not be null with
   respect to a transformation of it.** The ticker strategies have zero expected
   return by construction, but non-zero factor loadings, and OLS gives
   `alpha = rbar - beta' fbar` exactly, so their factor-model alphas are not zero.
   Between 29 and 41 percent of the cross-sectional variance of those alphas is
   attributable to the exposures. The alpha null therefore comes from a
   permutation that imposes `alpha = 0` given the loadings, not from the marginal
   spread.

## Licence and data

Code is MIT. We redistribute derived statistics only. The mined strategy returns
and the OSAP portfolios are published by their authors and downloaded by
`make fetch`; Planetoid and OGB are openly licensed and downloaded by the sweep;
the Fama–French factors come from the Ken French data library and are not
redistributed. The one file we do keep is
`data/raw/leaderboards/ogbn_arxiv_leaderboard.csv`, a dated transcription of a
public web page, so that the analysis stays reproducible after the page changes.
See `DATASHEET.md`.

## Citation

```bibtex
@misc{aksoy2026benchmarkzoo,
  title  = {The Benchmark Zoo: Calibrating Multiplicity Corrections
            Against a Known-Null Population},
  author = {Aksoy, P\i nar and Toroslu, \.Ismail Hakk\i},
  year   = {2026}
}
```
