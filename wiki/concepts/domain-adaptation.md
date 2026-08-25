---
title: "Domain Adaptation"
type: concept
tags: [domain-adaptation, transfer-learning, machine-learning, simulator, agent-based-simulation, selection-bias, evaluation]
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

# Domain Adaptation

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Transferring a model across a shift between the environment it learned in (**source**) and the environment it must work in (**target**), where the *task* is unchanged and the *distribution* is not.

Distinct from [[transfer-learning]], which changes the task and takes the data as given.

## Kinds of Shift

^[imported]

| | What moves | Example |
|---|---|---|
| **Covariate shift** | $P(x)$ | Different sensors, different population |
| **Label shift** | $P(y)$ | Different base rate |
| **Concept shift** | $P(y \mid x)$ | The relationship itself changed |

The first two are correctable in principle by reweighting, given some target data. **The third is not** — if the mapping from inputs to outputs has changed, no reweighting of the source recovers it, and the model needs retraining on the target.

Distinguishing them requires target labels, which is usually what is missing.

## Sim-to-Real and Its Inversion

The two directions are not symmetric, and the asymmetry is the useful part.

| | **Sim-to-Real** | **Real-to-Sim** |
|---|---|---|
| Source | A simulator | **Real-world data** |
| Target | A physical system | A simulator |
| Source dynamics | **Known** — someone wrote them | **Unknown** — that is the problem |
| Correctable by | Randomising or tuning the source | **Not that way** |

Sim-to-Real is well studied because you control the source: perturb it, randomise it, measure the discrepancy. Real-to-Sim inverts that. **You cannot adjust the source dynamics to close the gap, because you do not know what they are** — the generating process of real-world behaviour is not written down anywhere.

On the real side there is no transition model to correct; the next state is simply read from the data. On the simulator side a transition model must be *assumed*. The gap between them is unmeasurable in principle, because measuring it would require the thing that is missing.

## Validating a Transfer Claim

The methodological trap, and it is easy to fall into honestly.

Establishing that behaviour transfers requires choosing a dimension on which to compare source and target. That dimension is almost always chosen because it is **measurable in both** — which is close to choosing it because the domain gap does not affect it.

> ### `transfer-evidence-is-conditional-on-the-dimension-chosen`
> **Measuring domain transfer requires a comparison dimension, and selecting one insensitive to the gap is both the methodologically defensible move and the one that limits what the result establishes. A transfer finding is a finding about the chosen dimension, not about the domains.**
> ^[generated. rests-on: imported:sim-to-real-validation-practice]

The corollary is that positive and negative transfer results can coexist without contradiction: **where the compared dimension factors out the physical gap, partial transfer appears; where it is central, transfer fails.** Both are true statements about the same pair of domains.

## Methods

^[imported]

| Approach | Mechanism |
|---|---|
| **Domain randomisation** | Vary the source widely so the target is one draw among many |
| **Adversarial alignment** | Learn features a discriminator cannot use to tell the domains apart |
| **Importance reweighting** | Reweight source samples toward the target distribution |
| **Fine-tuning on target** | Use the small amount of target data directly |

Domain randomisation is Sim-to-Real's characteristic move and is unavailable in the inverse direction, for the reason above.

## Relation to Selection Bias

Both concern a mismatch between the data a model was fitted on and the population it is applied to. [[selection-bias]] is the case where the mismatch arises from *how the sample was drawn*; domain adaptation is where it arises from *where the model is deployed*. The mathematics of the correction is often the same.

## See Also

- [[transfer-learning]] · [[selection-bias]] · [[agent-based-simulation]] · [[imitation-learning]] · [[reinforcement-learning]]
- [[representation-learning]] · [[model-selection]] · [[predictive-validity]] · [[counterfactual-simulation]]
