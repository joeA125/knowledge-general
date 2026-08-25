---
title: "Capability Profiling"
type: concept
tags: [evaluation, model-decomposition, predictive-validity, construct-validity, cognitive-science, interpretability, uncertainty-quantification]
sources: [raw/papers/agi_definition.md]
confidence: 0.8
provenance:
  extracted: 45%
  inferred: 30%
  generated: 12%
  imported: 11%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Capability Profiling

Evaluating something by **decomposing its performance into components and reporting the vector**, rather than aggregating to a single score.

[[agi-definition|Hendrycks et al. (2025)]] make the case in its strongest form, grounding a definition of general intelligence in the Cattell–Horn–Carroll model from psychometrics — ten broad cognitive abilities, each measured separately, with the profile rather than the composite as the object of interest.

## The Core Claim

> ### `aggregates-assume-substitutability`
> **An aggregate is a fair summary only when its components are substitutes for one another. Where they are not, the composite is the least informative number available — and the one most likely to be quoted.**
> ^[generated. rests-on: source:hendrycks-chc-decomposition]

A single score implies that being better at one thing compensates for being worse at another. Sometimes true — for genuinely fungible components. Usually false, and the falseness is invisible in the number.

## Jaggedness

The property that determines whether an aggregate is safe.

- **A smooth profile** means the components are near each other, and the mean summarises them fairly.
- **A jagged profile** means the aggregate is a weighted average over things that are not substitutes, and two systems with identical scores may be jagged in opposite directions.

**Nothing in a single score tells you which case you are in.** That is the whole argument for reporting the vector: not that the composite is wrong, but that it cannot signal when it is misleading.

The held source's illustration is stark — a system may be at or above human level on most measured abilities while sitting near zero on long-term memory storage, and a composite obscures the gap entirely.

## Workarounds Mask Missing Capabilities

The subtler failure, and the one that survives careful aggregate reporting.

A system can substitute a workaround for a missing capability, scoring well on tasks that would otherwise expose the deficit. Large context windows standing in for memory storage; retrieval standing in for recall. The aggregate improves; the underlying capability does not exist.

> ### `a-proxy-that-satisfies-the-evaluation-becomes-the-definition`
> **Where an evaluation can be passed by a workaround, the workaround becomes what the metric measures. No amount of care in aggregating the components detects this, because the components are being measured correctly — it is the mapping from capability to task that has broken.**
> ^[generated. rests-on: source:hendrycks-capability-contortions]

This is [[rare-event-proxy-targets|proxy substitution]] arriving from the evaluation side rather than the training side, and it is harder to catch because nothing was substituted deliberately.

The defence is an **external criterion** — something a proxy cannot satisfy by construction. See [[predictive-validity]].

## Where Decomposition Helps Generally

The pattern is not specific to AI evaluation. Any composite over non-substitutable components has the same structure:

| Composite hides | Decomposition reveals |
|---|---|
| Equal scores from opposite strengths | Which strengths |
| A workaround compensating for a gap | The gap |
| Variance across sub-populations | Which populations |
| A metric conflating decision and execution | Which is weak |

The last is worth stating separately: **where a measure conflates two abilities, no aggregation of it separates them.** The decomposition has to happen in the measurement, not in the reporting.

## Limitations

- **Equal weighting is a choice, not a finding.** A profile collapsed under any weighting is an aggregate again, and nothing shows rankings are stable across weightings. See [[model-selection]].
- **Components must be separately valid.** Decomposing an unreliable measure produces several unreliable numbers — see [[split-half-reliability]] and `a-decomposition-is-only-as-good-as-its-least-validated-term` on [[structured-model-decomposition]].
- **Profiles are harder to act on.** A vector does not rank, and ranking is what most decisions require. The composite persists because it answers the question being asked, badly, rather than not at all.

## See Also

- [[structured-model-decomposition]] · [[predictive-validity]] · [[split-half-reliability]] · [[probability-calibration]] · [[model-selection]]
- [[rare-event-proxy-targets]] · [[class-imbalance-evaluation]] · [[uncertainty-quantification]] · [[interpretability]] · [[selection-bias]]
- [[agi-definition|Source Summary]]
