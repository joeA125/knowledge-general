---
title: "Counterfactual Baseline"
type: concept
tags: [counterfactual, evaluation, machine-learning, statistics, model-selection, predictive-validity]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 26%
  generated: 10%
  imported: 62%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Counterfactual Baseline

Valuing an action or outcome by comparison against **what would otherwise have happened**, rather than against an absolute scale.

The general form is a difference: *observed* minus *reference*. Everything interesting is in the choice of reference.

## Three References, Three Questions

| Reference | Difference measures | Reads as |
|---|---|---|
| **A population average** | Deviation from typical | "Better than most" |
| **A predicted behaviour** | Deviation from expected | "Did something unexpected" |
| **An optimum** | Distance from best available | "Left value on the table" |

These answer genuinely different questions and are routinely conflated. A subject scoring well against the first may score badly against the third; the two are not versions of the same measurement.

> ### `disputes-about-the-metric-are-usually-disputes-about-the-reference`
> **Where a measure is defined as a difference, disagreement about what it shows is almost always disagreement about what it is differenced against — not about the arithmetic. Stating the reference explicitly resolves more argument than refining the estimate.**
> ^[generated. rests-on: imported:counterfactual-evaluation-practice]

## The Predicted-Reference Case Is Strange

The second row deserves separate attention, because it inverts the usual objective.

A model of *typical* behaviour is normally a means to an end. Used as a reference, it becomes the instrument — and two awkward consequences follow:

- **The objective changes.** A forecaster wants minimal error; a reference wants a well-calibrated notion of *normal*. Optimising the first can degrade the second.
- **Perfection destroys the measurement.** If the predictor were exact, the difference would be identically zero. **The metric requires its own reference to be wrong.**

That is not a flaw to be fixed but a property to be understood. It also means such metrics are **not portable**: a value computed against one predictor is a different quantity from the same value computed against another.

## Individuation

A counterfactual is often what makes a *per-unit* attribution possible at all. Where an aggregate outcome cannot be decomposed directly, intervening on one unit and re-evaluating supplies the decomposition.

But intervention is not the only route. A model with **per-unit parameters** — one value function per agent, one embedding per entity — individuates by construction, without any counterfactual. So the counterfactual is *a* mechanism for attribution, not the only one, and treating it as necessary overstates its role.^[generated]

## What Makes a Reference Defensible

- **Was it chosen before the result was seen?** A reference selected after the fact is a researcher degree of freedom.
- **Is it estimable, or asserted?** A population average is measurable; an "optimum" usually requires a model of what was possible, with its own assumptions.
- **Does it hold across the comparison set?** A reference that shifts between the units being compared makes the differences incommensurable.
- **What happens at the extremes?** Some constructions clip or floor negative values, which means the measure cannot penalise — a substantial and often unstated restriction.

## Relation to Neighbours

- [[counterfactual-simulation]] — where the reference is *generated* rather than computed, with all the compounding-error caveats that brings.
- [[predictive-validity]] — a differenced metric still has to forecast something to be worth using; the difference structure alone establishes nothing.
- [[selection-bias]] — if the reference is estimated from a selected sample, the whole comparison inherits the distortion.

## See Also

- [[counterfactual-simulation]] · [[generative-model]] · [[predictive-validity]] · [[selection-bias]] · [[model-selection]]
- [[imitation-learning]] · [[policy-modelling]] · [[uncertainty-quantification]] · [[capability-profiling]]
