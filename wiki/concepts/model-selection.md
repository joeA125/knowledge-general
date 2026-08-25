---
title: "Model Selection"
type: concept
tags: [model-selection, machine-learning, statistics, evaluation, regularization, identifiability, approximation]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 22%
  generated: 8%
  imported: 68%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Model Selection

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Choosing among candidate models — which architecture, how many components, which hyperparameters. The unifying problem is that **fit to observed data always improves with complexity**, so fit alone cannot decide.

## Two Families

^[imported]

| | Mechanism | Needs |
|---|---|---|
| **Penalised likelihood** | Add a complexity term to the fit | A likelihood, and an assumption about the penalty |
| **Held-out validation** | Estimate out-of-sample performance directly | Enough data to hold some back |

AIC penalises by $2k$, BIC by $k\ln n$ — so BIC penalises complexity more heavily as sample size grows and tends to select smaller models. They answer different questions: AIC targets predictive performance, BIC targets identifying a true model, and they disagree systematically rather than randomly.

Cross-validation makes fewer assumptions and costs more compute. Where a likelihood is unavailable — most of deep learning — it is the only option.

## The Asserted Parameter Problem

The failure mode worth naming, because it is pervasive and rarely visible.

Many parameters are neither fitted nor validated. They are **chosen and stated**: a window length, a decay rate, a threshold, a loss weight. The paper reports the value, gives a one-line justification, and never varies it.

> ### `asserted-parameters-are-untested-model-selection`
> **A parameter set by assertion is a model-selection decision made without any selection procedure. Its cost is invisible unless someone varies it, and the practice is self-reinforcing: a sensitivity analysis can only weaken a paper — showing robustness confirms what reviewers assume, and showing sensitivity invites the question of why that value was chosen.**
> ^[generated. rests-on: imported:model-selection-practice]

Not all asserted parameters are equally dangerous, and the distinction is worth carrying:

| Kind | Effect when wrong | Example |
|---|---|---|
| **Horizon** | Slightly wrong values | How far back to look |
| **Shape** | Systematically wrong weighting | A decay rate |
| **Gate** | **The wrong set of events entirely** | A detection threshold |
| **Prior strength** | **A wrong conclusion**, not a wrong number | An imitation-loss weight |

Horizons are often self-limiting — most of the signal falls near the event regardless. **Gates are the structural problem**: a slightly wrong threshold does not produce slightly wrong values, it produces a different dataset, and everything downstream is computed on it.

The fourth kind is the most consequential and the least recognised. Where a weight controls how strongly a model is pulled toward observed behaviour, it determines how much apparent suboptimality survives into the results — so the finding depends on the parameter. See [[imitation-learning]].

## The Test That Is Rarely Run

For any asserted parameter, recompute the *rankings* the model produces across a defensible range and report the rank correlation between the extremes.

| Result | Reading |
|---|---|
| $\rho > 0.95$ | Not load-bearing; the choice is free |
| $\rho \approx 0.7$–$0.9$ | Shifts at the margins — enough to change a shortlist |
| $\rho < 0.7$ | Every published ranking is one arbitrary choice from a different answer |

**Rank correlation rather than value correlation**, because the decisions such models inform are usually ordinal.

## Selection Is Itself Overfitting

Repeatedly selecting on a validation set fits the model to that set. The standard defence is a third split — train, validate, test — with the test set touched once.

In practice it is touched more than once, and the resulting optimism is unmeasured. This is the same structure as multiple comparisons: **the more configurations tried, the better the best one looks by chance alone.**

## See Also

- [[predictive-validity]] · [[split-half-reliability]] · [[probability-calibration]] · [[class-imbalance-evaluation]] · [[regularization]]
- [[identifiability]] · [[uncertainty-quantification]] · [[imitation-learning]] · [[selection-bias]] · [[interpretability]]
