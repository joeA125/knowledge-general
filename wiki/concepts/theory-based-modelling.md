---
title: "Theory-Based Modelling"
type: concept
tags: [theory-based-modelling, feature-engineering, representation-learning, machine-learning, interpretability, model-selection]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 10%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Theory-Based Modelling

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Encoding what is known about a system as an explicit model — physics, geometry, a statistical mechanism — rather than leaving it to be learned. The output typically feeds a learned model as a feature, or replaces part of it outright.

## The Position It Occupies

| | Structure comes from | Fails when |
|---|---|---|
| **Purely learned** | The data | Data is scarce, or the structure is subtle |
| **Theory-based** | A model of the system | The theory is wrong |
| **Hybrid** | Theory supplies features, learning supplies the rest | Either component fails |

The hybrid is the common practical form and the one worth understanding: **use theory where it is reliable, learning where it is not.** A physical relationship that is genuinely known should not be re-derived from a thousand examples.

## Why It Persists

Three advantages that do not diminish with better models:

**Sample efficiency.** A known relationship encoded directly costs zero examples. The same relationship learned costs however many the architecture needs to discover it — and at small sample sizes, that may be more than exist.

**Extrapolation.** A learned function is reliable inside its training distribution and arbitrary outside it. A physical model derived from mechanism extrapolates as far as the mechanism holds, which is usually much further.

**Legibility.** The model states its assumptions in terms a domain expert can dispute. That is a different property from post-hoc [[interpretability|explanation]] and a stronger one — the assumptions are inspectable *before* the model runs.

## The Failure Mode

**A wrong theory is worse than no theory**, because it constrains the hypothesis space to exclude the truth and no amount of data recovers from it. A learned model given bad features can at least down-weight them; a model whose *structure* encodes a false mechanism cannot.

> ### `theory-based-errors-are-not-recoverable-by-data`
> **A misspecified learned model improves with more data; a misspecified structural model does not, because the error is in the hypothesis space rather than the estimate. The two failure modes look similar in-sample and diverge sharply out of it.**
> ^[generated. rests-on: imported:model-misspecification]

That asymmetry is the reason to prefer theory only where the theory is genuinely well-established, and to prefer learning where it is contested.

## Free Parameters Enter Anyway

A theory-based model usually contains constants — rates, thresholds, decay factors — that are not derived but chosen. These are the same asserted parameters catalogued on [[model-selection]], and they are easier to miss here because the surrounding structure looks principled.

**The appearance of derivation lends unearned confidence to the parts that were not derived.** A model with an equation and three fitted constants is often reported as though the constants inherited the equation's authority.

## Relation to Neighbours

- [[feature-engineering]] — the general activity; theory-based modelling is the subset justified by a mechanism rather than by intuition.
- [[representation-learning]] — the opposite pole; see `handcrafting-wins-where-the-data-cannot-support-discovery` for when each applies.
- [[agent-based-simulation]] — theory-based modelling at the level of agent rules, with emergence rather than a closed-form output.

## See Also

- [[feature-engineering]] · [[representation-learning]] · [[model-selection]] · [[interpretability]] · [[regularization]]
- [[agent-based-simulation]] · [[markov-game]] · [[uncertainty-quantification]] · [[predictive-validity]]
