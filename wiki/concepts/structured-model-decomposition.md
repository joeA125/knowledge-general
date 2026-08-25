---
title: "Structured Model Decomposition"
type: concept
tags: [model-decomposition, machine-learning, statistics, interpretability, model-selection, uncertainty-quantification, evaluation]
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

# Structured Model Decomposition

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Estimating a quantity by breaking it into subcomponents, fitting each separately, and recombining — rather than fitting the whole thing end to end.

Typically the decomposition follows the chain rule, so it is **exact by construction** rather than an approximation: $P(A, B, C) = P(A \mid B, C)\,P(B \mid C)\,P(C)$. No independence is assumed; each term conditions on the ones after it.

## What the Decomposition Buys

**Each component answers a question you can pose independently**, which means it can be validated independently. A composite model that predicts well overall may be wrong in a component whose error is masked by a compensating error elsewhere; a decomposed model exposes both.

That is the strongest argument for the approach and the one most often left implicit: **decomposition converts one unverifiable claim into several checkable ones.**

Two further advantages:

- **Reuse.** A component validated once can serve several downstream models.
- **[[interpretability|Legibility]].** The structure states what the model believes the world is made of, before any output is produced.

## What It Costs

**Error compounds multiplicatively.** Three components each 90% accurate give roughly 73% end-to-end, and nothing in the individual scores signals it.

**Components are often not independent in practice** even when the factorisation is exact. Where the same submodel appears in two terms — a shared surface, a shared encoder — errors correlate, and the naive product understates the uncertainty. See [[uncertainty-quantification]].

**Each component brings its own free parameters**, so a decomposed model typically has more asserted choices than a monolithic one, distributed where they are harder to see. See [[model-selection]].

## The Weakest-Component Problem

> ### `a-decomposition-is-only-as-good-as-its-least-validated-term`
> **Where a factorisation combines one well-validated component with several asserted ones, the apparent rigour of the validated part lends unearned confidence to the whole. Reporting per-component validation status is what prevents this, and it is rarely done.**
> ^[generated. rests-on: imported:structured-estimation-practice]

The practical form: a model may have one term fitted against directly observable outcomes and another set by a hand-chosen functional form with two constants. Both enter the product identically. Nothing in the output distinguishes them.

**Asking which component was actually checked is usually more informative than asking how the composite performed.**

## Relation to End-to-End Learning

| | Structure comes from | Validated |
|---|---|---|
| **Decomposed** | The modeller | Per component, in principle |
| End-to-end | The data | Only at the output |

End-to-end learning dominates where data is plentiful, for the same reason [[representation-learning]] dominates [[feature-engineering]] there — the model discovers a better factorisation than the modeller would impose.

Where data is scarce, or where a component must be reused, or where the intermediate quantities are themselves of interest, decomposition remains preferable. The intermediate quantities being useful is often the deciding factor and is not a modelling consideration at all.

## See Also

- [[model-selection]] · [[interpretability]] · [[uncertainty-quantification]] · [[probabilistic-classification]] · [[theory-based-modelling]]
- [[representation-learning]] · [[feature-engineering]] · [[bayesian-inference]] · [[capability-profiling]]
