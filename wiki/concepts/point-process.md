---
title: "Point Process"
type: concept
tags: [point-process, stochastic-process, statistics, time-series, event-prediction, density-estimation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 6%
  imported: 68%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Point Process

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

A stochastic model of **discrete events occurring in continuous time** (or space). The object modelled is not a value at each time but the set of times at which something happened.

## The Conditional Intensity

The standard characterisation. $\lambda^*(t)$ is the instantaneous rate of events at $t$ given the history $H_t$:

$$\lambda^*(t)\,dt = \mathbb{P}(\text{event in } [t, t+dt) \mid H_t)$$

The asterisk marks the conditioning on history, and it is where all the modelling lives. Given $\lambda^*$, everything else follows — the likelihood, the expected counts, the inter-event distribution.

## The Standard Families

| Family | $\lambda^*(t)$ | Behaviour |
|---|---|---|
| **Homogeneous Poisson** | Constant $\lambda$ | Memoryless; events independent |
| **Inhomogeneous Poisson** | $\lambda(t)$, no history | Rate varies with time, not with past events |
| **Hawkes** | $\mu + \sum_{t_i < t} \phi(t - t_i)$ | **Self-exciting** — each event raises the near-term rate |
| **Neural** | A learned function of encoded history | Arbitrary dependence |

The Hawkes process is the conceptually important one: it makes explicit that **events can cause events**. Earthquakes trigger aftershocks; a trade moves the book; a message prompts replies. Where that holds, a Poisson model is not merely imprecise, it is the wrong shape.

## Why Not Just Bin the Timeline

The obvious alternative — discretise time and count events per bin — and the reasons it is worse:

- **The bin width is an asserted parameter** that determines what is resolvable. Events within a bin become simultaneous.
- **Sparse bins dominate.** At fine resolution most bins are empty and the model spends capacity predicting zeros.
- **Exact timing is discarded**, which is the information a point process exists to use.

Binning is nonetheless common, because it converts the problem to familiar territory. **The cost is usually paid silently**, and the bin width is rarely swept — the gate problem from [[model-selection]] in another guise.

## Marked Point Processes

Where each event carries attributes — a type, a location, a magnitude — the process is *marked*, and the model must predict the mark alongside the time. That is the general form of the multi-component prediction described on [[event-prediction]].

## Fitting

^[imported]

The log-likelihood has a characteristic form:

$$\log L = \sum_i \log \lambda^*(t_i) - \int_0^T \lambda^*(u)\,du$$

The first term rewards high intensity where events occurred; the **integral penalises high intensity everywhere else**. That second term is what stops the trivial solution of predicting a high rate always, and it is the awkward part computationally — for neural intensity functions it usually has no closed form and must be approximated by Monte Carlo.

> ### `the-compensator-is-what-makes-intensity-models-hard`
> **The integral term in the point-process likelihood is what distinguishes it from ordinary sequence modelling, and it is also the reason flexible intensity functions are expensive. Approaches that predict inter-event times directly sidestep the integral at the cost of losing the intensity interpretation.**
> ^[generated. rests-on: imported:point-process-likelihood]

## See Also

- [[neural-temporal-point-process]] · [[event-prediction]] · [[autoregressive-model]] · [[model-selection]]
- [[probabilistic-classification]] · [[generative-model]] · [[bayesian-inference]] · [[uncertainty-quantification]]
