# Datasheet

One section per released artifact: what it is, how it was made, what it is for,
what it is not for, and what is known to be wrong with it. Following the
datasheet framing that the Datasets and Benchmarks track expects.

Nothing here redistributes an upstream dataset. Every artifact below is either a
derived statistic computed by this repository, or a dated transcription of a
public web page kept so that the analysis stays reproducible.

---

## 1. `calibration/empirical_null.json` — the measured null

**What.** The cross-sectional distribution of four test statistics over the
19,380 ticker-symbol long--short strategies of Chen and Dim, in equal-weighted
and value-weighted versions: mean-return $t$, Newey--West $t$, CAPM alpha $t$,
five-factor alpha $t$. For each: moments, quantiles, exceedance probabilities at
fixed cutoffs with Wilson intervals, generalised Pareto tail fits at five
thresholds, calibrated cutoffs at three levels, the eigenvalue spectrum of the
strategy correlation matrix, the permutation distribution of the maximum
statistic with and without cross-strategy dependence, and the whole set again
split by decade and by market volatility regime.

**Construction.** `scripts/03_empirical_null.py` from the upstream monthly
returns. Sample 1963-01 to 2022-12, 720 months. Strategies with fewer than 60
monthly observations are dropped. Factor models use the Fama--French five factors
from the Ken French data library. Seeds are fixed at 20260801 and recorded in the
file.

**Intended use.** As the null distribution a multiplicity correction consumes,
in place of a textbook one, for monthly long--short equity strategy statistics.
As a reference point for anyone building a comparable calibration in another
domain.

**Not for.** Transferring the numerical values to a different domain, a different
sampling frequency, or a different statistic. The whole argument of the
accompanying paper is that these values are domain-specific and that what
transfers is the method.

**Known limitations.**

* The population is null with respect to *raw returns* by construction. It is
  **not** null with respect to factor-model alphas, because a zero-mean strategy
  with non-zero factor loadings has a non-zero alpha. The marginal spread of the
  five-factor alpha $t$-statistic in this file therefore mixes a null
  distribution with real alpha; the object to use for an alpha threshold is the
  permutation maximum in artifact 2, not the marginal spread here. This is the
  single most important caveat in the datasheet.
* On roughly 4,845 dependent strategies the null standard deviation is pinned to
  about ±15 percent. Recalibrating on the four disjoint letter-position subsets
  gives 0.74 to 1.05 for the equal-weighted mean-return statistic.
* The permutation null uses block sign flips with the standard deviation held at
  its sample value, which is exact under sign symmetry and approximate otherwise.
* The generalised Pareto fits below about $5\times 10^{-5}$ are extrapolations;
  19,380 strategies cannot resolve a smaller tail probability directly.

---

## 2. `calibration/null_validation.json` — validation of the known-null claim

**What.** Four sets of tests on whether the population is really null:
eighteen alphabeticity subgroup tests with block-bootstrap intervals; the
decomposition of the five-factor alpha dispersion into factor exposure and
residual; a permutation null that imposes $\alpha=0$ given the loadings and
returns the distribution of the largest alpha $t$-statistic; and recalibration on
the four disjoint letter-position subsets.

**Construction.** `scripts/04_known_null_validation.py`. The alphabeticity tests
aggregate each subgroup into one monthly series *before* testing, because
strategies within a subgroup share months and are strongly dependent; a test
treating them as independent observations would reject constantly.

**Intended use.** As the evidence for or against the known-null assumption, and
as the source of the alpha threshold.

**Not for.** Concluding that alphabetical position has no effect on returns in
general. We test whether it produces a detectable mean in *these* long--short
portfolios, which is a narrower claim than the one Itzkowitz, Itzkowitz and
Rothbort (2016) make about turnover and valuation.

**Known biases.** The 114 documented placebo characteristics would have been a
second, independently constructed null, but the public OSAP release contains
portfolios only for the 212 predictors, so that check could not be run. The
substitute — disjoint subsets of the same population — shares months with itself
and is therefore weaker.

---

## 3. `benchmarks/ml_runs_<dataset>.parquet` and `ml_correct_<dataset>.npy`

**What.** One row per training run: architecture, configuration id, seed, epoch
budget, selected epoch, train, validation and test accuracy, parameter count and
wall-clock seconds. Alongside it, a boolean matrix of shape
(runs × test instances): was each test node classified correctly by each run.

**Construction.** `scripts/06_ml_sweep.py`. Four datasets, six architectures,
random configurations drawn once from a fixed grid with seed 20260901, and the
same configuration list, seed set and epoch budget for every architecture.
Full-batch training on CPU. Model selection on validation accuracy; the test set
is read once per run, at the validation-selected epoch.

**Intended use.** Re-deriving any of the paper's benchmark results without
retraining, and building alternative null constructions. The per-instance
correctness matrix is what an instance-level bootstrap needs, and it is why the
matrix rather than the scalar accuracy is released.

**Not for.** Reading off state-of-the-art numbers. The sweep is deliberately a
*modest, uniform* budget, not a serious attempt to maximise accuracy, and the
scope excludes any model requiring the fine-tuning of a pretrained language
model.

**Known limitations.**

* Per-instance *predictions* are not stored, only correctness. Balanced accuracy
  and per-class recall are recoverable from correctness plus true labels;
  macro-F1 is not.
* Configurations are drawn uniformly, not by a Bayesian search. That is
  deliberate — uniform draws model search without innovation, which is the
  relevant null — but it means the sweep understates the accuracy a competent
  practitioner would reach.
* ogbn-arxiv receives fewer configurations and seeds than the three small graphs,
  because a run costs two orders of magnitude more. The asymmetry is between
  datasets, never between architectures within a dataset.
* Runs on CPU, so wall-clock seconds are not comparable with GPU numbers and
  should not be read as an efficiency measurement.

---

## 4. `data/raw/leaderboards/ogbn_arxiv_leaderboard.csv`

**What.** A transcription of the ogbn-arxiv node property prediction leaderboard
of the Open Graph Benchmark: 81 submissions over 74 displayed ranks, dated
2020-05-01 to 2026-02-10, with reported test and validation accuracy, reported
standard deviations, submitter and date.

**Provenance.** <https://ogb.stanford.edu/docs/leader_nodeprop/>, accessed
2026-07-31. Kept in the repository because the page changes and the analysis has
to remain reproducible. The underlying graph is OGB's and is downloaded
separately under its own terms.

**Intended use.** As an *observable lower bound* on the number of trials behind
the current best number on this benchmark, and as the source of the
gap-versus-reported-noise comparison.

**Not for.** Treating 81 as the number of trials. It counts submissions that were
made public, not attempts, so it is a lower bound and a loose one. Every result
in the paper that uses a trial count is reported over a grid from 10 to 100,000
with this bound marked, never at this bound alone.

**Known issues.** Two entries report no validation accuracy and are kept with
blank validation fields rather than dropped. Some methods appear more than once
from different submitters with different numbers; those are separate submissions
and are counted as such. Names are as submitted and are not normalised, so the
same underlying method may appear under variant names.

---

## 5. `results/*.json` and `paper/tables/*.tex`

**What.** Every result the paper reports, and the LaTeX source of every table and
in-text number.

**Construction.** `scripts/02` through `scripts/09`. `make all` regenerates all of
it from raw sources in one command. `paper/tables/macros.tex` holds every single
number that appears in the prose, as `\newcommand` definitions, so the text
cannot disagree with the tables.

**Intended use.** Verification. A reader who doubts a number in the paper can
find the script that produced it and the JSON it came from.

---

## Maintenance, hosting and contact

The repository is the working copy; a versioned snapshot with a DOI is deposited
on Zenodo at release, which is the citable and persistent location. Issues and
corrections via the repository issue tracker. If a number here turns out to be
wrong we will say so in `DECISIONS.md` with a date, as we have already done twice
for our own mistakes, rather than silently editing it.

**Ethics.** No human subjects, no personal data. The work re-evaluates published
results; it is written as a measurement of a collective record and names no paper
as a false discovery. The foreseeable misuse is applying a threshold from this
work to a domain whose null has not been measured, and the package raises rather
than defaulting when `sigma` is not supplied.
