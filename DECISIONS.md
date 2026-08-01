# Decisions log

Every non-obvious choice, dated the day it was made, with the reason. Written
as it happened, not reconstructed. Where a decision was changed later the
original entry stays and a follow-up entry says what changed and why.

---

## 2026-07-30 — Kill test, before any code

**Kill condition 1: does Chen and Dim already estimate an empirical null from
the mined portfolios?** Partly. Their Figures 2 and 3 plot the distribution of
t-statistics by strategy family against a normal null, and Section 3.3 states
that for the ticker-based families "the null is a very good fit for the data".
Appendix A.3 describes the 38,000 ticker-based strategies. So the *qualitative*
claim is theirs and must be credited as theirs. What they do not do: quantify
the departure numerically, fit the tail, measure dependence or an effective
trial count, derive corrected thresholds, check whether the same holds for
risk-adjusted statistics, or examine stability over time. Decision: keep C1 but
narrow it to those, and cite Chen and Dim as establishing the premise.

**Kill condition 2: has anyone applied finance multiplicity corrections to ML
benchmark evaluation?** Searched "deflated" + benchmark, "family-wise error" +
leaderboard, "multiple testing" + benchmark overfitting, "reality check" +
machine learning evaluation. Found multiple-comparison work inside information
retrieval evaluation and conformal prediction, and the deflated Sharpe ratio
applied to ML-driven *trading* strategies, but nothing that transfers the
finance machinery to benchmark accuracy metrics with a calibrated null.
Decision: C2 stands.

**Kill condition 3: does the known-null assumption hold?** Deferred to
measurement; see 2026-08-05.

**Kill condition 4: does the adaptive-overfitting literature already answer
Q3?** No, and the distinction holds: Recht et al. and Roelofs et al. ask
whether accuracy on a *fresh* test set matches accuracy on the reused one, and
mostly find that it does. We ask a different question: given the reported
improvement over a baseline, and given the number of trials behind it, is the
improvement larger than the best of that many null draws. A benchmark can have
no adaptive-overfitting gap at all and still have a reported improvement that a
null search would produce routinely. Decision: go, and make the distinction in
the introduction rather than in a rebuttal.

**Go/pivot: go.**

---

## 2026-07-30 — Data sources

Chen and Dim's mined strategy returns come as a single 1.77 GB Dropbox zip.
Inside, ticker and past-return strategies are in **separate files**
(`TickerSignalsLongShort.csv.gzip`, `PastReturnSignalsLongShort.csv.gzip`), not
one mixed file as the plan assumed. That removes the largest data risk, but the
classifier is still written and unit-tested, because the signal-id space starts
at zero in *both* files, so anything that pools them must namespace the ids
(rule R10 in `src/bzoo/finance/partition.py`).

The files are named `.csv.gzip` but are uncompressed CSV: pandas does not treat
`.gzip` as a compression hint. The loader sniffs the magic bytes instead of
trusting the extension.

The public `PredictorPortsFull.csv` release contains portfolios for the 212
predictors only. `SignalDoc.csv` documents 114 placebo characteristics, but
their portfolio returns are not in the release, so the planned second null
population from placebos is unavailable. Substituted with recalibration on the
four disjoint ticker letter-position subsets; see 2026-08-05.

---

## 2026-07-31 — Past-return signal indices are quarters, ordered oldest first

The first version of rule R9 read the indices in past-return signal names as
month lags, and labelled `ret_1` as short-term reversal and anything inside
lags 2 to 12 as momentum. The power check then failed with momentum at a median
t-statistic of **-2.42**, which is the wrong sign.

Reading `get_past_return_signals.py`: `Nyrs = 5`, `nqtr = 20`, and
`qtr_ids = [1,1,1,2,2,2,...,20,20,20]` is assigned to
`df_crspm.loc[t_0:t-1]`, which is in chronological order. So the index is a
**non-overlapping quarter**, and **1 is the oldest quarter of the past five
years, 20 the most recent**.

Under the corrected reading the numbers line up with the literature without any
tuning: `ret_17_18_19_20` (the most recent twelve months) has t = +3.78
equal-weighted and +4.11 value-weighted; `ret_18` and `ret_19` are +5.46 and
+5.74, which is where intermediate-horizon momentum lives; `ret_20` alone (the
most recent quarter, contaminated by one-month reversal) is -1.65; and
`ret_1_2_3_4` (four to five years back) is -1.80, the long-run reversal sign.

Had this gone unnoticed the paper would have inverted every economic statement
about the power set. It is the single most consequential reading decision in the
project and is why the power check is on the *signed* statistic against a
pre-stated expected sign rather than on the magnitude.

---

## 2026-07-31 — Power-check pass criterion

Under the corrected labels the three known-signal families behave as follows.

| family | expected sign | EW median $t$ | VW median $t$ |
| --- | --- | --- | --- |
| momentum (q 17-20) | + | +5.60 | +4.18 |
| recent (q 20 alone) | - | -1.65 | +1.21 |
| long-run (q 1-8) | - | -1.92 | -0.45 |

Momentum is strong in both weightings. Short-term and long-run reversal are
right-signed and moderate equal-weighted, and weak or wrong-signed
value-weighted. That is what the literature says: both reversal effects are
concentrated in small stocks, so value weighting removes most of them, and it is
also Chen and Dim's own finding that predictability concentrates in small
stocks.

Criterion adopted, after seeing these numbers and stated as such: all three
families right-signed with median signed t above 1.5 equal-weighted, and
momentum above 2.0 in both weightings. The value-weighted reversal results are
reported, not required. Tightening a criterion after seeing the data is worth
flagging; the alternative was a criterion that fails for a reason the
literature predicts.

---

## 2026-07-31 — Which statistic the null is estimated for

The plan speaks of "the empirical null" as one object. It is not. On the same
known-null population, measured over 720 months and 19,380 strategies per
weighting:

| statistic | EW sd | VW sd |
| --- | --- | --- |
| mean-return $t$ | 0.889 | 1.026 |
| Newey-West $t$ | 0.886 | 1.054 |
| CAPM alpha $t$ | 0.883 | 1.070 |
| five-factor alpha $t$ | **1.150** | **1.405** |

So the calibration has to be reported per statistic. Decision: report all four
everywhere, and treat the five-factor alpha as the primary case because that is
the statistic published anomaly papers actually report.

---

## 2026-07-31 — Apple MPS is not used

`torch.sparse.mm` has no MPS kernel in PyTorch 2.8, and the models are sparse
matrix products. Falling back to a dense adjacency is not possible for
ogbn-arxiv. All runs are on CPU. Recorded because it fixes the compute budget
and therefore the sweep size.

`ogb` 1.3.6 caches its processed graph with `torch.save` of a plain dict, which
PyTorch 2.6+ refuses to load under the new `weights_only=True` default. The
loader relaxes the flag for the duration of the OGB call only.

---

## 2026-07-31 — Machine learning scope, fixed before results

Benchmark family: **transductive node classification, accuracy**. Four
datasets: Cora, CiteSeer, PubMed (Planetoid public splits) and ogbn-arxiv (OGB
official split). One family, one metric, four saturation levels.

Architectures, six, each with the same configuration list, seeds and epoch
budget: `mlp`, `gcn`, `sage`, and the ablations `gcn_noprop`, `gcn_unnorm`,
`sage_noneigh`.

Exclusion rule, fixed now and not revisited: **no model that requires
fine-tuning a pretrained language model.** That excludes the top of the
ogbn-arxiv leaderboard (GIANT-XRT, TAPE, SimTeG, GLEM and the rest), which is a
real limitation and is stated as one. It follows from the compute budget, not
from a judgement about those methods.

Budget parity is enforced by `bzoo.ml.tuning.check_budget_parity`, which raises
before anything is written if any architecture received a different number of
trials, a different seed set, or a different configuration set.

Configurations are drawn uniformly at random from a fixed grid, not by a
Bayesian search. A Bayesian search finds better models and is a worse model of
the null, because it concentrates on the good region and understates how often
undirected search stumbles onto a large improvement.

---

## 2026-08-05 — Known-null validation: what holds and what does not

**Alphabeticity.** Eighteen subgroup tests (four letter positions, three
alphabetical regions of the long leg, extreme versus interior sorts, in two
weightings). Every bootstrap confidence interval contains zero and no subgroup
survives Bonferroni. The Itzkowitz, Itzkowitz and Rothbort objection does not
bite on these portfolios.

**Where the assumption fails.** "No economic content" implies a null *raw
return*. It does **not** imply a null factor-model alpha, because OLS gives
`alpha_hat = rbar - beta_hat' fbar` exactly, so a strategy with zero expected
return and non-zero factor loadings has a non-zero alpha. Measured: 29 percent
(VW) to 41 percent (EW) of the cross-sectional variance of the five-factor
alphas is attributable to `beta' fbar`, and the correlation between the alpha
and `-beta' fbar` is 0.64 to 0.72. The Newey-West version of the alpha
t-statistic is *wider*, not narrower, than the OLS version, so this is not a
standard-error problem.

Consequence, and it is a limitation not a result: the ticker population
calibrates raw-return statistics against ground truth, and for the five-factor
alpha it calibrates the null distribution of the alpha *given* the exposures,
which requires the permutation construction in Part C of
`scripts/04_known_null_validation.py` rather than the marginal spread. Both are
reported, and the difference between them is stated in the paper.

**Reproducibility across subpopulations.** Recalibrating on the four disjoint
letter-position subsets gives sd of the mean-return t-statistic in the range
0.739 to 1.045 equal-weighted and 0.837 to 1.186 value-weighted; for the
five-factor alpha t-statistic, 0.869 to 1.416 and 1.103 to 1.590. So the
calibration is reproducible in sign and rough magnitude but not to two decimal
places on 4,845 dependent strategies. Reported as a range, not a point.

---

## 2026-08-05 — Effective number of tests is reported on the threshold scale

The plan asks for an effective trial count from an eigenvalue method and from a
permutation max-$t$ method, with a statement of which is preferred. The answer
is that neither count should be the headline.

Measured on 14,535 complete-history ticker strategies: Cheverud-Nyholt gives
14,088, Li-Ji gives 57 (EW) and 191 (VW), and the Sidak-implied count from the
permutation maximum gives about 3,000. The two eigenvalue methods differ by a
factor of 74 to 250 on the same matrix.

The reason the counts are unstable is that $N_{\text{eff}}$ is an exponentially
amplified rescaling of a threshold. Measured with the marginals held exactly
fixed - joint sign flips versus independent sign flips - dependence lowers the
95th percentile of $\max|t|$ from 4.73 to 4.39 for the mean-return statistic,
which is 7 percent. A 7 percent move in the threshold is a three- to five-fold
move in the implied count.

Decision: report the maximum-statistic quantile as the primary object and the
effective count as a derived quantity with its instability stated. This is a
substantive point about the literature's preferred summary and belongs in the
paper, not only here.

---

## 2026-08-06 — Baseline selection is on validation accuracy only

`bzoo.ml.nulls.select_baseline` picks the tuned baseline by validation accuracy
and never by test accuracy. A baseline chosen on test accuracy would already
have absorbed part of the multiplicity being measured, and every improvement
measured against it would be biased downwards. The function has no code path
that reads test accuracy during selection.

---

## 2026-08-06 — Leaderboard snapshot

The ogbn-arxiv leaderboard was transcribed on 2026-07-31 into
`data/raw/leaderboards/ogbn_arxiv_leaderboard.csv` and kept in the repository,
so that the analysis is reproducible after the page changes. 81 submissions over
74 displayed ranks, dated 2020-05-01 to 2026-02-10.

The count is a lower bound on the number of trials, and a loose one: it counts
public submissions, not attempts. Every result that uses a trial count is
reported over a grid from 10 to 100,000 with the leaderboard count marked, never
at the leaderboard count alone.

---

## 2026-08-07 — Three sources date months three different ways

The ticker strategy file dates each month by its first day, the past-return
strategy file and the OSAP portfolios by its last, and the Fama-French files by
its first. A merge between two of them on the raw dates produces an **empty
intersection and no error**.

This was found when the robustness check that regresses the past-return
higher-moment strategies on the five factors returned 210 complete return series
and zero usable regressions. Nothing raised; `factor_alphas` simply had no rows
to work with and returned an empty frame, whose standard deviation is `NaN`,
which appeared in the results file as a missing number rather than as a failure.

Every monthly index in the package is now snapped to the first of the month in
`bzoo.finance.loaders.to_month_start`, and `tests/test_report.py` asserts that
all four sources align and that every pairwise overlap exceeds 700 months. The
main results were unaffected, because the ticker panel and the factors already
agreed; only the robustness item was wrong, and it was wrong in a way that showed
up as a blank rather than as a wrong number.

Recorded because the general lesson is the one that matters: a silent empty merge
is worse than a crash, and the check that catches it belongs in the test suite
rather than in a reviewer's reading.

---

## 2026-08-07 — The direction of a haircut, pinned by a test

A wider null makes a given observed $t$-statistic less surprising, so **more** of
the reported Sharpe ratio is attributed to the search and the haircut is
**larger**; a narrower null gives a smaller haircut and more survivors. We wrote
the first version of the test with this backwards. Getting it wrong would invert
every survival count in the paper, so `tests/test_haircuts.py` now asserts the
direction explicitly in both weightings.

---

## 2026-08-08 — Two memory bugs the small benchmarks hid

The instance bootstrap worked fine on Cora, CiteSeer and PubMed, all of which
have 1,000 test nodes, and would have needed 95 TB on ogbn-arxiv, which has
48,603. Both bugs are the same mistake in two places: an intermediate array whose
size is the product of three quantities, two of which are small on a toy
benchmark.

**One.** `bootstrap_metric_matrix` was written as
`scores[indices].mean(axis=1)`, which materialises a
`(n_replicates, n_instances, n_models)` array. That is 5.7 GB for Cora with 4,000
replicates and 178 models, which the machine absorbed without complaint, and
`4000 x 48603 x 61 x 8` bytes for ogbn-arxiv. It now processes replicates in
batches sized to a byte ceiling, and a test checks that batching does not change
a single number.

**Two.** `independent_instance_indices` returns the whole
`(n_replicates, n_models, n_instances)` index array, which is 95 GB on the same
benchmark. Added `bootstrap_metric_matrix_independent`, which draws one model's
indices, uses them and discards them. The array version stays for the small cases
and for the tests, with the memory cost written into its docstring.

Also reduced the replicate count from 4,000 to 1,000 for test sets above 5,000
instances. A standard deviation over 48,603 instances is already precise at 1,000
replicates, and the per-model scheme costs one gather per model per batch, so the
extra 3,000 replicates buy nothing.

The general lesson, and the reason this is written down: the small benchmarks in
this project are not just faster, they are a different order of magnitude in every
dimension at once, so they do not exercise the memory behaviour at all. Anything
that runs on Cora should be re-read before it runs on ogbn-arxiv.

---

## 2026-07-31 — Reviewing an external critique, and what survived checking

An external critique of the draft proposed a rewrite around one claim: a
strategy with no predictive content has a null raw return and a non-null
factor-model alpha. Every checkable assertion in it was tested against the data
before anything was changed. Most held; four did not, and the four that did not
are the interesting part. Recorded here because the corrections are now in the
paper and the reasoning behind them is not obvious from the text.

**The `--` in `\bzooaltNullFfSd*` was not an unfilled placeholder.** It was a
NaN written by an earlier run of `08_robustness.py`. The current code computes
the quantity without error, so the NaN was stale, not a bug in the live path.
Re-running gives 0.89 equal-weighted and 1.01 value-weighted. Lesson: a `--`
that reaches a table is indistinguishable from a placeholder, and a NaN should
have failed the run instead of being formatted. The generated-macro discipline
protected the prose and not the number.

**The claim that the second no-content population "shows no widening at all"
was wrong, and wrong in our favour.** Its alpha t-statistic standard deviation
is 1.01 against the ticker population's 1.41, which reads as no inflation only
if one forgets that its *raw-return* null is 0.78 rather than 1.02. The
comparable quantity is the widening. Risk adjustment multiplies the dispersion
of that population's alpha numerator by 1.42 and the ticker population's by
1.39. Two unrelated sorting variables, the same widening. This replaced a
contrast we were reporting as a limitation with a replication, and it is now
Table `alt_population` and a falsification paragraph.

**The "slope of exactly 1" test does not follow from the identity.** Regressing
`alpha` on `-beta'fbar` has population slope `1 + Cov(rbar, -beta'fbar) /
Var(beta'fbar)`, so 1 requires the mean return to be cross-sectionally
orthogonal to the exposure. That holds equal-weighted (slope 0.992) and fails
value-weighted (1.297), and the formula reproduces both to machine precision.
Replaced with the exact variance decomposition `Var(alpha) = Var(rbar) +
Var(beta'fbar) - 2 Cov(rbar, beta'fbar)`, which needs no such condition.

**SD(t_alpha) is not monotone in the number of factors.** Value-weighted it
runs 1.02, 1.07, 1.36, 1.28, 1.41, 1.37 for K = 0, 1, 3, 4, 5, 6: adding
momentum *lowers* it. What drives the curve is whether the added factor carries
a mean return the sort loads on, not how many regressors there are. This is a
better result than monotonicity would have been, because a regressor count
cannot explain it and the exposures can.

**The sqrt(T) form of the divergence test does not fit** (RMSE 0.14 on
SD(t_alpha)), because the windows are nested and end in a decade when the
factor premia were near zero, so a short recent window has little
exposure-driven alpha to find. Replaced with a fit of `Var(alpha_T) = V_inf +
k/T`, which needs no assumption of stable premia: the persistent component is
11.2 bps per month, 80 percent of the variance at the full sample length,
against the 8.5 bps that pure estimation noise would leave. Same conclusion,
robust derivation.

**Two checks were run and discarded.** A residual *permutation* placebo returns
SD(t_alpha) = 0.36, which looks like a spectacular confirmation and is an
artefact: OLS residuals have exactly zero sample mean, so any permutation of
them does too, which forces alpha to `-beta'fbar` with beta near zero. Replaced
with a bootstrap with replacement, which gives 0.98. The wrong version is kept
as a failing-by-design test in `tests/test_alpha_mechanism.py` so nobody
reintroduces it. Separately, the first version of the analysis took `rbar` over
each strategy's own months and the exposure over the regression months; those
differ by the six months before the factor series starts, and the identity then
failed by about 3 percent of a cross-sectional standard deviation — small
enough to look like rounding. Everything is now computed on the regression
sample and the identity holds at 0.0.

**14,535 is not a subsetting bug.** It equals 3 x 4,845, which invited the
suspicion. The fourth ticker letter position only exists for tickers of four
characters or more, so those 4,845 strategies have 600 months rather than 720.
The threshold table caption now says so, along with the fact that M = 14,535 is
what the Bonferroni column actually uses, not the 19,380 the old caption
implied.

**210 versus 80 is a labelling collision, not an arithmetic error.** Table 1's
`longrun: 80` counts `ret`-root signals at that horizon; the second no-content
population is the disjoint set of 210 `std`/`skew`/`kurt` signals over the same
quarters. Both captions now say which is which.

**The OSAP placebo claim was too strong.** The release documents 114 placebo
characteristics and the `openassetpricing` client has no placebo portfolio
path, so "portfolios only for the predictors" is literally true. But the
placebo characteristics are distributed at the firm level and this package
already has the sorting code, so "unavailable" overstated it. Reworded to say
it is work we have not done. The release we used has 212 predictors, which is
recorded here because a later release may not.

**Bonferroni's two errors compound, they do not cancel.** Section 5 said the
nominal null is "slightly light relative to the true mixture". Table 2 shows
the measured mean-return null is narrower (0.889) *and* thinner-tailed
(kurtosis 2.81) than standard normal, so both the dependence effect and the
marginal effect push the correct threshold down and Bonferroni overshoots on
both counts. Corrected.

**Not done, and deliberately.** The critique also asked to cut the machine
learning half, restructure to a finance-only paper and retitle. That changes
the target venue and is a coauthor decision, so the ML sections are untouched.

## 2026-07-31 — ogbn-arxiv finished, and yields no threshold

The sweep completed while the above was being written. It produces no
`sigma_delta`: 0 of 38 null trials pass the screening rule, because the tuned
baseline leads the best other candidate by 5.0 times the per-trial noise. The
candidate pool contains nothing plausibly as good as the baseline, so it
contains no null population, and a spread over the least-bad candidates would
be a spread of real differences. This is the same failure the recentring step
exists to prevent, in a more extreme form than Cora shows. It is reported as a
result rather than as a pending item, and what it needs is a wider and
deliberately weaker configuration sweep, not more seeds.

`09_tables_and_figures.py` crashed on this case (`KeyError: 1000` on an empty
deflation grid) because the loop guarded on `available` and not on
`sigma_delta_estimable`. Fixed to emit the descriptive macros and skip the
deflation ones.

---

## 2026-08-01 — Testing the headline claim, and narrowing it

A second round of external critique made one observation that turned out to be
decisive: the paper asserted that no multiplicity correction is a remedy for
this channel and never applied a multiplicity correction to the null
population. We have twelve corrections and 19,380 strategies where every
rejection is false by construction. `scripts/14_corrections_on_the_null.py`
runs the test.

**The claim survives, but only in a narrower form.** Bonferroni at the full
family size of 19,380 with the textbook null rejects 0 strategies on the raw
mean return -- exactly the family-wise control it promises -- and 69 on the
five-factor alpha. Benjamini-Hochberg rejects 0 and 995. Same strategies, same
correction, same M; only the statistic changed. So the critique's expectation
that "Bonferroni mostly holds" was wrong, and the alarming version of our claim
is supported at the whole-population level and at every smaller family size we
checked: with fifty candidate signals Bonferroni still has an 85 percent chance
of at least one false positive against the 5 percent it promises.

**But the same script shows the correction is not what is broken.** Keep
Bonferroni, keep M = 19,380, and substitute the measured null for the standard
normal, and the 69 false rejections go to 0. So "no multiplicity correction is
a remedy" was too strong and is now "no correction applied with the *nominal*
null addresses this channel; one applied with a *measured* null does, and the
measurement is the hard part because it is a property of the candidate
population's exposures". The paper is more defensible for the change, and it
also removes a real internal contradiction: the remedy section already
recommended a measured cutoff on the same statistic, which made no sense if
the statistic itself were beyond repair.

**Why a fixed rescaling suffices.** Splitting
`Var(t_alpha) = 1 + c1 + c2*T`, where c1 is the sample-specific exposure term
and c2*T the persistent one, gives c1 = 0.73 and c2*T = 0.027 value-weighted:
96 percent of the excess is a fixed scale inflation, and the growing part would
take on the order of 1,600 years of data to matter as much. The critique
predicted this from the disjoint-block persistence result and it was right.

**Three things I got wrong and corrected.**

*The prose said "one clears Bonferroni at the full trial count."* That was meant
as "the largest one clears" and reads as "exactly one clears". It misled the
critic into concluding Bonferroni was working. The number is 69.

*The claim that in-sample beta estimation manufactures the negative
Cov(rbar, beta'fbar).* I wrote that estimating both on the same half turns the
covariance positive "which is the sign the mechanical coupling produces". Then
I simulated it: for OLS with an intercept the coupling term is proportional to
the sum of the demeaned regressor and is exactly zero, with independent
residuals and with a shared common shock alike. So there is no mechanical
coupling and that sentence was wrong.

*What the covariance actually is.* Working the algebra rather than guessing:
`Cov_i(rbar, beta'fbar) = (fbar - mu_f)' Sigma_beta fbar`, a realised quadratic
form in the sample factor means. Its expectation over samples is
`trace(Sigma_beta Var(fbar))`, which is positive, and simulation confirms it --
positive on average, negative in 44 percent of draws. So a negative value in
our sample is neither evidence the population fails to be null nor an artefact;
it is sample-specific, which is the same thing the c1/c2 split says. We now
build no claim on its sign.

**Also this round.** Added Novy-Marx (2014) and distinguished it explicitly:
his nonsense predictors are a data-mining result, which is what a correction is
designed to catch, and ours are not selected at all. Added Lo and MacKinlay
(1990), Ferson, Sarkissian and Simin (2003), Fama and French (2010) and Harvey
and Liu (2020). Made the recommended cutoff procedural rather than a number,
since the paper contains its own evidence that 3.04 does not transfer, and
noted that a ticker letter induces exposure accidentally while a real candidate
signal is related to size and value by construction -- so our figure is a lower
bound. Promoted "What to do instead" to a section. Connected the zero-sum
constraint to the survival result and credited Chen and Dim for reaching the
same conclusion first. Replaced five hand-typed numbers with macros, dropped a
duplicated decile panel, and consolidated the two macros that held the same
quantity.

---

## 2026-08-01 — Reframing from a finance paper to a data-mining paper

The result is about screening statistics in general and was written as a
finance paper. This round moves the framing without moving the evidence.

**The organising idea, which we did not have before.** The confidence of an
association rule between independent items is the marginal frequency of the
consequent, not zero, so screening on confidence harvests frequent consequents.
Data mining's answer was not a stricter threshold but a re-centred statistic:
lift. A factor-model alpha is confidence, not lift, and our remedy section is
the asset-pricing version of a move data mining made in 1997. Saying that makes
the paper general rather than a finance paper in a CS venue, and it costs
nothing because the analysis is unchanged.

Title is now `Screening on the Wrong Null: When a No-Content Hypothesis Does
Not Imply a Zero Statistic`. Abstract leads with the general failure mode and
gives finance as the population where it can be measured against ground truth.
Introduction restructured to four movements: search needs a null, the
association-rule precedent, why finance is the testbed rather than the subject,
the identity and the result. Related work reordered so pattern discovery leads
and the factor zoo becomes the application domain. Conclusion generalised.

**New Section~\ref{sec:instances}, three further instances, no computation.**
Accuracy against a uniform baseline when a majority-class predictor scores the
base rate; sampled-negative recommendation metrics (Krichene and Rendle 2020);
and improvement over an estimated baseline, which is the object Hansen's
recentring exists to fix. Each has the same shape: the no-content value of the
statistic is not the value the threshold is set against.

**`bzoo` promoted** from the fourth item in a contribution list to its own
section, with the two design decisions that follow from the paper's argument
written down: no default null anywhere in the API, and a test per correction
against a published worked example.

**Cut and compressed.** The ethics section was long, defensive and
finance-shaped; it is now one paragraph headed Broader impact. The survival
material was about a third of the body and is the demoted result --- one
paragraph in the body, the table and the dependence-aware procedures in an
appendix. The quarters reading decision went from twelve lines to four with a
pointer to the code. Robustness moved ahead of the remedy so the general
material closes the paper.

**Verified rather than acted on.** Two items in the critique were already
correct in the draft: the winner's raw-return $t$ is $-2.74$, so the "losing
22.7 bps" prose is right (the critic was reading an earlier draft where I had
stripped the signs), and the thresholds caption already states $M = 14{,}535$
and separates the marginal spread from the $\alpha = 0$ permutation. Also
softened "which the literature conflates" to a claim about reporting practice,
since we cite no instance of the conflation, and corrected "six benchmark
models" to five models plus the raw return, since $K = 0$ is not a model.

**Not done.** The document class is still `neurips_2023`, which is right for
arXiv and NeurIPS and wrong for DMKD or KAIS. That is a venue decision, and it
sets the length target, so it should be made before the next full pass rather
than guessed at here.
