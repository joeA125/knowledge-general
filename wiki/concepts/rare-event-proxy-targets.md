---
title: "Rare-Event Proxy Targets"
type: concept
tags: [proxy-target, class-imbalance, machine-learning, statistics, evaluation, predictive-validity, model-selection]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 10%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Rare-Event Proxy Targets

> ⚠️ **No held source.** Background knowledge, marked `imported:`. This page arrived by copy during the 2026-08 migration and was rewritten when its sources proved to be football material held elsewhere.

Predicting a **frequent correlate** when the outcome of interest is too rare to learn from directly.

## The Problem

At a base rate below roughly 1%, a classifier has very few positives to learn from, and standard training produces a model that is either degenerate or so uncertain as to be useless. [[class-imbalance-evaluation]] covers why the *evaluation* also misleads in this regime.

Resampling and reweighting address the training imbalance without addressing the underlying scarcity: **there is only so much information in a hundred positive examples**, however they are weighted.

Proxy substitution takes a different route. Rather than predicting the rare outcome, predict something on its causal path that occurs orders of magnitude more often, and treat that as the target.

## What the Substitution Costs

The trade is not merely statistical. **The model now measures something else**, and whether the proxy stands in faithfully is a separate question the substitution itself cannot answer.

| | Rare target | Proxy target |
|---|---|---|
| Learnable | Barely | **Yes** |
| Measures what you want | **Yes** | Only if the proxy is faithful |
| Validatable against | The rare outcome | The proxy — circularly |

That last row is the trap. **A model trained on a proxy and evaluated on the proxy has established nothing about the outcome of interest.** Validation has to be against the rare outcome, on however few instances exist, or against an external criterion.

> ### `a-proxy-must-be-validated-against-the-thing-it-replaces`
> **Substituting a frequent correlate for a rare target makes the model learnable and makes it unfalsifiable at the same time, unless validation is explicitly carried back to the original outcome. Evaluating a proxy model on its proxy is a closed loop that can look arbitrarily good.**
> ^[generated. rests-on: imported:proxy-target-practice]

## Choosing a Proxy

Three properties, and they conflict:

- **Frequency.** Common enough to learn from — the whole point.
- **Faithfulness.** Movement in the proxy should imply movement in the target.
- **Manipulability.** A proxy that can be increased *without* increasing the target will be, once anything optimises against it.

The third is Goodhart's problem and it is the one most often ignored at design time. A proxy chosen purely on the first two properties becomes a target the moment a system is optimised against it, at which point its correlation with the real outcome is exactly what stops holding.^[imported]

## Where the Substitution Hides

Proxy targets are usually introduced deliberately and documented. Two places they enter without discussion:

**Reward shaping.** Adding an auxiliary reward because the true reward is too sparse to bootstrap from is proxy substitution inside a reward function. It is standard practice in [[reinforcement-learning]] and rarely framed as changing what the agent optimises — but it is. See [[proximal-policy-optimization]].

**Evaluation design.** Where a benchmark measures something easier than the capability it claims to assess, the benchmark has substituted a proxy on the evaluation side. See `a-proxy-that-satisfies-the-evaluation-becomes-the-definition` on [[capability-profiling]].

## Relation to Neighbours

- [[class-imbalance-evaluation]] — the same base-rate problem, addressed by thresholds and metrics rather than by changing the target
- [[predictive-validity]] — the correct check on a proxy model, and the one that closes the circular loop
- [[selection-bias]] — where the proxy's *availability* is itself non-random, the two problems compound

## See Also

- [[class-imbalance-evaluation]] · [[predictive-validity]] · [[probabilistic-classification]] · [[probability-calibration]] · [[model-selection]]
- [[capability-profiling]] · [[selection-bias]] · [[reinforcement-learning]] · [[proximal-policy-optimization]] · [[imitation-learning]]
