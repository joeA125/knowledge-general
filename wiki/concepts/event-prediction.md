---
title: "Event Prediction"
type: concept
tags: [event-prediction, point-process, sequence-modelling, machine-learning, time-series, autoregressive-model, evaluation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 8%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Event Prediction

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Forecasting the next event in a sequence — typically its **time**, its **type**, and sometimes its **location** or other attributes. Distinct from ordinary sequence modelling because *when* is part of the prediction rather than an index.

## Why Time Changes the Problem

A sequence model predicts $x_{t+1}$ given $x_{\leq t}$, with $t$ an integer position. An event model predicts a **continuous** inter-event time alongside the event itself.

| | Sequence model | Event model |
|---|---|---|
| Time | An index | **A predicted quantity** |
| Gaps | Uniform by construction | Informative |
| Natural formalism | Autoregressive factorisation | [[point-process\|Point process]] |

The gaps carrying information is the substantive difference. A burst of rapid events and the same events spread over hours are different phenomena, and a model indexing by position cannot distinguish them.

## Multi-Component Prediction

Where an event has several attributes, the joint distribution is usually factorised by the chain rule and each component given its own network:

$$f(t, z, m \mid H) = f_t(t \mid H)\, f_z(z \mid t, H)\, f_m(m \mid t, z, H)$$

Two things follow that are easy to miss.

**Later components condition on earlier predictions**, not on earlier truths — so error propagates within a single event, and the last component in the chain absorbs the accumulated error.

**The ordering is a free parameter.** Any permutation is a valid factorisation of the same joint distribution, and with unlimited capacity all would be equivalent. In practice they are not, and the differences concentrate in the final component.

> ### `factorisation-order-is-an-unswept-parameter`
> **Where a model factorises a multi-component prediction, the ordering determines which conditionals are easy to learn. It is exact in theory and consequential in practice, and is almost always asserted rather than searched.**
> ^[generated. rests-on: imported:autoregressive-factorisation]

The practical heuristic: **put first the components that best predict the others.**

## Evaluation Is Awkward

^[imported]

Each component needs a different measure — a continuous density for time, a discrete score for type — and combining them into one number requires a weighting nobody derives.

Two consequences:

- **Total loss is not comparable across papers** that weight components differently, and the weighting is often unreported.
- **A model can improve overall while degrading on the component that matters**, which per [[capability-profiling]] is exactly what a composite hides.

Reporting per-component losses alongside the total is the fix, and it is cheap.

## Derived Metrics

A recurring pattern: build a forecaster, then define a metric from its predictions rather than from outcomes directly. The forecast becomes a measuring instrument.

This inherits everything about the forecaster, including its biases — and the metric's [[predictive-validity|validity]] is then a separate question from the model's accuracy. A well-fitted forecaster can yield a metric that predicts nothing useful, and nothing in the fit statistics reveals it.

## See Also

- [[point-process]] · [[neural-temporal-point-process]] · [[autoregressive-model]] · [[transformer]] · [[lstm]]
- [[capability-profiling]] · [[predictive-validity]] · [[model-selection]] · [[probabilistic-classification]] · [[generative-model]]
