---
title: "Feature Engineering"
type: concept
tags: [feature-engineering, representation-learning, machine-learning, theory-based-modelling, model-selection, interpretability]
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

# Feature Engineering

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Constructing the inputs a model sees. The classical activity that [[representation-learning]] was meant to replace, and which persists in the places representation learning is weakest.

## What a Feature Encodes

A feature is a claim about what matters. Three kinds, with different justifications:

| Kind | Encodes | Justified by |
|---|---|---|
| **Aggregation** | Counts, rates, windows | The relevant timescale |
| **Domain structure** | A physical or theoretical relationship | A model of the system — see [[theory-based-modelling]] |
| **Interaction** | Products, ratios, differences | A hypothesis about non-additivity |

The second is the interesting case. Encoding a known relationship — a distance, an angle, a conservation law — gives the model something it would otherwise need data to discover, and often could not discover at all from the sample available.

## When Handcrafting Wins

The received view is that learned representations dominate given enough data. That is true and the qualifier does the work.

> ### `handcrafting-wins-where-the-data-cannot-support-discovery`
> **Encode structure when the representation cannot recover it *and* the data cannot support learning it. Where either condition fails — the architecture can express it, or there is enough data — handcrafting adds bias without adding information.**
> ^[generated: a reconciliation of the two positions; not stated by any held source. rests-on: imported:feature-engineering-vs-representation-learning]

Both conditions matter. A convolutional architecture already expresses translation invariance, so handcrafting it is redundant. A relationship requiring millions of examples to infer, from a dataset of thousands, will not be learned however expressive the model.

**The rule is not falsifiable as stated**, which is a weakness worth recording: any case where handcrafting failed can be attributed to sufficient data, and any case where it succeeded to insufficient data. It is a heuristic for deciding, not a claim that can be tested.

## The Cost Side

- **Bias.** A feature encoding a false relationship constrains the model to a wrong hypothesis space, and no amount of data recovers from it.
- **Leakage.** Features computed over the full dataset — normalisations, target encodings — smuggle test information into training.
- **Maintenance.** Handcrafted pipelines encode assumptions that decay as the data-generating process shifts, silently.
- **Opacity paradox.** Handcrafted features are often defended as interpretable, but a pipeline of fifty engineered features is not obviously more legible than a learned embedding. See [[interpretability]].

## Where It Sits

| | Who chooses the representation | Needs |
|---|---|---|
| **Feature engineering** | A person, in advance | Domain knowledge |
| [[representation-learning]] | The model, from data | Data, and an architecture that can express it |
| [[pre-train-then-fine-tune\|Transfer]] | Another model, on another task | A related large corpus |

The third route is why the question feels less pressing than it was: a pretrained encoder supplies a representation without either handcrafting or task-specific learning. That moves the problem rather than removing it — the pretraining corpus now carries the assumptions.

## See Also

- [[representation-learning]] · [[theory-based-modelling]] · [[tokenization]] · [[pre-train-then-fine-tune]] · [[transfer-learning]]
- [[model-selection]] · [[interpretability]] · [[selection-bias]] · [[regularization]]
