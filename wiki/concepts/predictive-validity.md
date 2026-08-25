---
title: "Predictive Validity"
type: concept
tags: [predictive-validity, evaluation, construct-validity, reliability, statistics, machine-learning, cognitive-science]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 20%
  generated: 8%
  imported: 70%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Predictive Validity

> ⚠️ **No held source.** Background knowledge from psychometrics and measurement theory, marked `imported:`.

Whether a measure forecasts an outcome it is supposed to forecast. **The strongest available evidence for a metric whose target is unobservable** — because the criterion sits outside the model that produced the score.

## Origin

The concept comes from psychometrics, where a test's worth is established by what it predicts rather than by what it appears to measure. An aptitude test has predictive validity if scores forecast later job performance; the test itself is never validated against "aptitude", which nobody observes.

That structure transfers wherever the quantity of interest is a construct rather than a measurement.

## Why It Outranks the Alternatives

| Criterion | Asks | Can be satisfied by |
|---|---|---|
| Face validity | Does it look right? | Anything plausible |
| [[split-half-reliability\|Reliability]] | Is it stable? | **A degenerate constant** |
| Construct validity | Does it agree with related measures? | **Agreeing with other flawed measures** |
| **Predictive validity** | Does it forecast? | Genuinely forecasting |

The second and third rows are the point. A metric that always returns 7 is perfectly reliable. A metric correlating with existing metrics may be inheriting their errors. **Only predictive validity requires the metric to be right about something that has not happened yet**, which is why it is hard to game by construction.

## The Criterion Must Be External

The critical requirement, and the one most often missed.

If a metric is validated against an outcome that was itself used in training, the exercise measures internal consistency rather than validity. The criterion must be **outside the pipeline that produced the score**: a future observation, an independent measurement, a real-world consequence.

> ### `self-prediction-is-not-validation`
> **A model predicting a quantity derived from its own outputs demonstrates consistency, not validity. The distinction is easy to lose when the criterion is convenient and hard to notice once the result is favourable.**
> ^[generated. rests-on: imported:criterion-validity-requirement]

## A Surprising Regularity

Where it has been measured across domains, **a well-constructed intermediate metric often forecasts an outcome better than the lagged outcome itself does.**

The reason is variance rather than magic: a rare terminal outcome is a noisy signal of underlying quality, while a frequent intermediate measure aggregates more evidence per unit time. Predicting next period's rare event from this period's rare event throws away most of the available information.

This is the same logic that motivates [[rare-event-proxy-targets|proxy substitution]] on the training side, arriving on the evaluation side.

## Limitations

- **The criterion may be wrong.** Forecasting a flawed outcome measure validates against a flawed target — job performance ratings, for instance, carry their own biases.
- **Selection effects.** Predictive validity is usually established on the subset that was selected, which is not the population the metric will be applied to. See [[selection-bias]].
- **Predictive is not causal.** A metric may forecast an outcome without any intervention on it changing that outcome.
- **Horizon matters.** A metric predictive at one horizon may be useless at another, and the horizon is rarely reported as a parameter.

## See Also

- [[capability-profiling]] · [[split-half-reliability]] · [[uncertainty-quantification]] · [[probability-calibration]] · [[model-selection]]
- [[selection-bias]] · [[rare-event-proxy-targets]] · [[class-imbalance-evaluation]] · [[interpretability]]
