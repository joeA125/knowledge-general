---
title: "Probabilistic Classification"
type: concept
tags: [probabilistic-classification, calibration, machine-learning, evaluation, class-imbalance, statistics, uncertainty-quantification]
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

# Probabilistic Classification

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Predicting a **probability** rather than a label. A classifier that outputs 0.7 is making a different claim from one that outputs "positive", and the difference matters wherever the output feeds a decision with asymmetric costs.

## Why the Probability Rather Than the Label

A hard label bakes in a decision threshold — usually 0.5 — and discards the information needed to choose a different one. Three consequences:

- **The threshold belongs to the decision, not the model.** Different costs of false positives and negatives imply different thresholds, and only a probability lets the same model serve both.
- **Probabilities compose.** Expected values, downstream products, and decompositions all require them; labels cannot be multiplied meaningfully.
- **Rare events need them.** At a low base rate, a calibrated model correctly outputs low probabilities for everything, and a 0.5 threshold predicts the negative class always. See [[class-imbalance-evaluation]].

That last point is where hard-label evaluation goes most badly wrong, and it is worth stating plainly: **a model reported as useless at a 0.5 threshold may be well-calibrated and genuinely informative.**

## Scoring Rules

^[imported]

A **proper** scoring rule is minimised by reporting one's true belief — so it cannot be gamed by hedging or overconfidence.

| Rule | Form | Note |
|---|---|---|
| **Brier score** | $\frac{1}{N}\sum (p_i - y_i)^2$ | Bounded, interpretable, less sensitive to extremes |
| **Log loss** | $-\frac{1}{N}\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$ | Unbounded — punishes confident errors severely |
| Accuracy | Fraction correct | **Not proper.** Ignores confidence entirely |

Log loss is the harsher of the two proper rules: a confident wrong prediction is penalised without limit, which is desirable when overconfidence is the failure mode of concern and destabilising when a single mislabelled example can dominate the metric.

## Discrimination and Calibration Are Different Things

The distinction that most often goes unmade.

| | Asks | Measured by |
|---|---|---|
| **Discrimination** | Does it rank positives above negatives? | AUC, ranking metrics |
| **[[probability-calibration\|Calibration]]** | Are the stated probabilities right? | Reliability curves, ECE |

**A model can rank perfectly and be badly calibrated** — outputting 0.9 for everything positive and 0.8 for everything negative gives an AUC of 1.0 and probabilities that are nonsense. The reverse also holds: a model outputting the base rate for every input is perfectly calibrated and useless for ranking.

Reporting only one is common and misleading. Which matters depends on the use: ranking for triage, calibration for expected-value computation.

## Limitations

- **Neural networks are typically overconfident**, and increasingly so with capacity. Calibration is not automatic and generally requires a post-hoc correction.^[imported]
- **Proper scoring rules aggregate over the whole distribution**, so a model may score well overall while being badly wrong in a region that matters.
- **Probabilities are still estimates**, with their own uncertainty that a single number does not convey. See [[uncertainty-quantification]].

## See Also

- [[probability-calibration]] · [[class-imbalance-evaluation]] · [[uncertainty-quantification]] · [[predictive-validity]] · [[model-selection]]
- [[bayesian-inference]] · [[split-half-reliability]] · [[interpretability]]
