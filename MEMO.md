# Week-1 memo: kill test, go or pivot

Hocam,

One page, as promised, before any code. What exists, what is still open, and the
decision.

## What exists

**On the finance side, the premise is already published, and narrower than we
assumed.** Chen and Dim (arXiv 2311.10685) plot the distribution of
$t$-statistics by strategy family against a normal null in their Figures 2 and
3, and Section 3.3 says that for the ticker-based families "the null is a very
good fit for the data". Appendix A.3 describes the 38,000 ticker-based
strategies. So the *qualitative* claim that ticker strategies behave like a
theoretical null is theirs and we must credit it as theirs.

What they do not do: put a number on the departure, fit the tail, measure
cross-strategy dependence or an effective trial count, derive corrected
thresholds from the measurement, check whether the same holds for the
*risk-adjusted* statistics that published papers actually report, or examine
stability across decades. C1 survives, narrowed to exactly those.

**On the machine learning side, nothing.** I searched "deflated" + benchmark,
"family-wise error" + leaderboard, "multiple testing" + benchmark overfitting,
and "reality check" + machine learning evaluation. There is multiple-comparison
work inside information retrieval evaluation, there is conformal-prediction
multiplicity work, and there is the deflated Sharpe ratio applied to
machine-learning-driven *trading* strategies. There is nothing that transfers the
finance machinery to benchmark accuracy metrics with a calibrated null. C2 is
open.

**The nearest neighbour is not the same question.** Recht et al. and Roelofs et
al. ask whether accuracy on a fresh test set matches accuracy on a reused one,
and mostly answer yes. That is a question about generalisation from a reused
holdout. Ours is a question about a threshold: given a reported improvement and
the number of attempts behind it, is the improvement larger than the best of that
many null draws? The two can come apart in both directions, and the distinction
belongs in the introduction, not in a rebuttal. Kill condition 4 does not fire.

## What is still open

The known-null assumption itself. Itzkowitz, Itzkowitz and Rothbort (Review of
Finance 2016) document that alphabetical position affects turnover and valuation,
so ticker-derived signals are not automatically null and a referee will attack
here first. This has to be measured, not argued: split the population by letter
position, by alphabetical region, and by extreme against interior sorts, and test
every subgroup. If a subgroup carries a real effect, exclude it and show the
downstream results do not move. That is week 2 work and it is the one place the
design could still collapse.

## Decision: go

The two things the paper needs are both present. The finance side has a
population that is null by construction, so a calibration can be verified rather
than assumed, which exists nowhere in machine learning. The machine learning side
has dated public leaderboards, so the trial count is partly observable, which
exists nowhere in finance. Each supplies what the other lacks, and that symmetry
is the paper rather than an ornament on it.

Your side of it is direct: this is the pattern explosion problem in its native
form. We are scanning a large hypothesis space and controlling false discoveries,
and the effective-number-of-tests question is the same one the permutation-based
significant pattern mining literature answers. I would like your reading on
whether the effective-$N$ estimator should be the eigenvalue kind or the
permutation max-$T$ kind; my expectation is that they will disagree and that the
disagreement is itself reportable.

Target: NeurIPS Datasets and Benchmarks. Data is entirely public, compute is a
laptop plus one small sweep. Detailed plan attached.

---

## Addendum, week 6: what the measurement actually showed

Recording this here rather than rewriting the memo, because the difference
between what we expected and what we found is the contribution.

1. **There is no such thing as "the" empirical null.** On the same known-null
   population the mean-return $t$-statistic has standard deviation 0.89
   equal-weighted and 1.03 value-weighted --- so the theoretical null is fine,
   confirming Chen and Dim numerically. The five-factor alpha $t$-statistic, which
   is what published anomaly papers report, has standard deviation 1.15 and 1.41.
   The calibration has to be reported per statistic.

2. **Part of that width is a real alpha, not a miscalibration, and that is a
   limitation we have to state.** OLS gives $\hat\alpha = \bar r - \hat\beta'\bar
   f$ exactly, so a strategy with zero expected return and non-zero loadings has
   a non-zero alpha. Between 29 and 41 percent of the cross-sectional variance of
   the alphas is attributable to the exposures. "No economic content" implies a
   null raw return and does not imply a null alpha. The alpha threshold therefore
   has to come from a permutation that imposes $\alpha = 0$ given the loadings.

3. **The correction goes the other way from what everyone expects, on the
   statistic where we can check it.** Calibrating the null makes *more* published
   predictors survive, not fewer: 102 instead of 83 under Bonferroni. Worth
   saying as plainly as the opposite result would have been.

4. **On your question about the effective number of tests: they disagree by two
   orders of magnitude, and the right response is to stop reporting the
   quantity.** On 14,535 strategies, Cheverud--Nyholt gives 14,088 and Li--Ji
   gives 57. Measured directly with the marginals held fixed, dependence moves
   the 95th percentile of $\max|t|$ from 4.73 to 4.39, which is 7 percent. A 7
   percent move in a threshold is a three- to five-fold move in the implied
   count, which is why the estimators cannot agree. We report the maximum-statistic
   quantile as the primary object and the count as derived, with its instability
   stated. This is the part of the paper that is most directly yours.

5. **The transfer needed a step the finance literature treats as optional.** In
   finance the candidate pool is strategies with small expected returns. In
   machine learning it contains models that are twenty-five accuracy points
   worse, whose true improvement is $-0.25$ rather than $0$. Including them makes
   $\sigma_\Delta$ twenty times too large and the threshold vacuous. Hansen's
   recentring is what fixes it, and here it is essential rather than a
   refinement.

6. **Alphabeticity does not bite.** Eighteen subgroup tests, every interval
   contains zero, nothing survives Bonferroni. The objection is answered
   empirically rather than argued away.
