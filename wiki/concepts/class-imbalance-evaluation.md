---
title: "Class Imbalance Evaluation"
type: concept
tags: [class-imbalance, evaluation, probabilistic-classification, calibration, machine-learning, statistics, proxy-target]
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

# Class Imbalance Evaluation

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Training and evaluating when one label is far rarer than the other. The training problem is well known; **the evaluation problem is the one that produces false conclusions.**

## The Trap

At a base rate of 0.2%, a model predicting the negative class for every input achieves 99.8% accuracy. That much is familiar.

The subtler version is the one that misleads. A **well-calibrated** model on rare data correctly outputs low probabilities for almost everything — because almost everything is negative. Threshold that at 0.5 and it predicts the negative class always, giving precision, recall and F1 of exactly zero.

> ### `f1-zero-can-mean-unthresholded-not-uninformative`
> **An F1 of zero on a rare target is the expected result for any calibrated model evaluated at 0.5, and says nothing about whether the model is informative. Reporting it as a finding about the model, rather than about the threshold, inverts the conclusion.**
> ^[generated. rests-on: imported:calibration-under-imbalance]

The diagnostic is simple: **sweep the threshold.** If ranking is informative, some threshold produces non-zero F1 and accuracy typically improves at a cutoff well below 0.5. If no threshold helps, the model genuinely has no signal — and that is a different claim, properly established.

## Which Metrics Survive Imbalance

| Metric | Under heavy imbalance |
|---|---|
| Accuracy | **Useless** — dominated by the majority class |
| ROC-AUC | Optimistic; the false-positive rate has a huge denominator |
| **Precision–recall AUC** | Better — both terms concern the rare class |
| **[[probabilistic-classification\|Brier / log loss]]** | Threshold-free, but dominated by easy negatives |
| **F1 at a swept threshold** | Informative, if the sweep is reported |

The recommendation that follows: **report a threshold-free proper score *and* a threshold-swept discrete metric.** Either alone is misleading — the first hides whether any operating point is usable, the second hides whether the probabilities mean anything.

## Responses on the Training Side

^[imported]

| Response | Mechanism | Cost |
|---|---|---|
| Resampling | Over-sample rare, under-sample common | Distorts the base rate, breaking [[probability-calibration\|calibration]] |
| Class weighting | Reweight the loss | Same calibration cost, less variance |
| Threshold tuning | Leave training alone, move the cutoff | **Preserves calibration** |
| [[rare-event-proxy-targets\|Proxy substitution]] | Predict a frequent correlate instead | Changes what is being measured |

Threshold tuning is the conservative choice and often sufficient. **Resampling and reweighting both trade calibration for discrete-metric performance**, which matters if anything downstream treats the output as a probability.

Proxy substitution is the most consequential and the least reversible: it reorganises the model around a different target, and whether the proxy stands in faithfully is a separate question the substitution itself cannot answer.

## Limitations

- **Imbalance is not always a problem.** If the base rate is the truth and the model is calibrated, nothing is wrong except the evaluation.
- **The minority class may be heterogeneous.** Rare events often have several distinct causes, and a single classifier averages over them.
- **Thresholds do not transfer.** A cutoff tuned on one population is not valid on another with a different base rate.

## See Also

- [[probabilistic-classification]] · [[probability-calibration]] · [[predictive-validity]] · [[uncertainty-quantification]] · [[model-selection]]
- [[rare-event-proxy-targets]] · [[selection-bias]] · [[split-half-reliability]]
