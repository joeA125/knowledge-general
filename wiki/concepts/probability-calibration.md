---
title: "Probability Calibration"
type: concept
tags: [calibration, probabilistic-classification, uncertainty-quantification, evaluation, machine-learning, statistics, deep-learning]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 22%
  generated: 6%
  imported: 70%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Probability Calibration

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Whether predicted probabilities match observed frequencies. A model is calibrated if, among all cases it assigns probability 0.3, roughly 30% are positive.

This is a claim about the *numbers*, independent of whether the model ranks well. See [[probabilistic-classification]] for why the two come apart.

## Measuring It

| Method | What it shows |
|---|---|
| **Reliability curve** | Mean prediction against mean outcome, per bin. Perfect calibration is the diagonal |
| **Expected calibration error** | Weighted mean gap between the two, across bins |
| **Brier decomposition** | Splits the Brier score into calibration, refinement and uncertainty terms |

The binning is a free parameter and is rarely reported. Too few bins hides local miscalibration; too many produces bins dominated by noise. **A reliability curve is a statement about a binning choice as much as about a model.**^[generated]

## The Direction of Miscalibration Matters

Overconfidence and underconfidence fail differently, and conflating them loses the useful part.

- **Overconfident** — probabilities pushed toward 0 and 1. Downstream expected-value calculations become too extreme; rare risks are dismissed.
- **Underconfident** — probabilities compressed toward the base rate. The model is safe and uninformative.

Modern neural networks are systematically **overconfident**, and increasingly so with capacity — a large network fitted to convergence typically has worse calibration than a smaller one with the same accuracy.^[imported]

## Post-Hoc Correction

^[imported]

| Method | Fits | Note |
|---|---|---|
| **Platt scaling** | A logistic on the scores | Two parameters; assumes a sigmoid shape |
| **Temperature scaling** | A single $T$ dividing the logits | **One parameter, does not change ranking at all** |
| **Isotonic regression** | A monotone step function | Flexible; needs more data, can overfit |

Temperature scaling's property is the useful one: because a single positive scalar applied to all logits preserves their order, **it fixes calibration without touching discrimination.** That makes it nearly free to apply and hard to argue against.

All three require a held-out set. Calibrating on training data measures the fit, not the calibration.

## Why It Is Often Skipped, and Why That Is a Mistake

Calibration does not improve accuracy, AUC, or any ranking metric — so it never shows up in the headline number a model is selected on.

> ### `calibration-is-invisible-to-the-metric-that-selects-the-model`
> **Model selection almost always optimises discrimination, which is unaffected by calibration. A pipeline can therefore be systematically miscalibrated at every stage without any reported metric registering it — and the cost falls entirely on downstream decisions that treat the outputs as probabilities.**
> ^[generated. rests-on: imported:calibration-discrimination-independence]

The cost is concrete wherever probabilities are composed: an expected value computed from miscalibrated components inherits the error multiplicatively, and a decomposition into several miscalibrated submodels compounds it further.

## Limitations

- **Calibration is a population property.** A model can be calibrated overall and badly miscalibrated on an important subgroup — the same averaging problem as [[selection-bias]].
- **It says nothing about usefulness.** Predicting the base rate for every input is perfectly calibrated.
- **Recalibration does not fix a bad model**, it fixes the reporting of one.

## See Also

- [[probabilistic-classification]] · [[uncertainty-quantification]] · [[class-imbalance-evaluation]] · [[predictive-validity]] · [[split-half-reliability]]
- [[model-selection]] · [[bayesian-inference]] · [[selection-bias]] · [[interpretability]]
