---
title: "Siamese Network"
type: concept
tags: [metric-learning, deep-learning, architecture, representation-learning, computer-vision, entity-embedding]
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

# Siamese Network

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Two (or more) identical networks with **shared weights**, applied to different inputs, whose outputs are compared. The architecture learns an embedding space in which distance means similarity.

## Why Shared Weights Matter

The weight sharing is the whole design, not an efficiency measure.

Because both inputs pass through **the same function**, the comparison is symmetric and the embedding is consistent: two identical inputs necessarily map to the same point. A pair of independently-parameterised networks would have neither property, and the learned "distance" would depend on which input went where.

## Learning the Space

^[imported]

| Loss | Operates on | Objective |
|---|---|---|
| **Contrastive** | Pairs | Pull similar together; push dissimilar apart beyond a margin |
| **Triplet** | Anchor, positive, negative | Anchor closer to positive than to negative by a margin |

The margin is a free parameter and it is load-bearing: too small and the space collapses, too large and most triplets are trivially satisfied and contribute no gradient. See [[model-selection]].

**Triplet loss depends heavily on mining.** Once training progresses most randomly-sampled triplets are already correct, so the useful signal comes from *hard* examples — and hard-negative mining is where most of the practical difficulty in these systems lives.^[imported]

## What It Is For

The defining use case is **open-set** retrieval and verification: matching against classes that were not in the training set.

That is the property a classifier cannot supply. A classifier learns a fixed set of labels and must be retrained to add one. A Siamese network learns a *space*, and a new class needs only a stored embedding.

> ### `metric-learning-trades-a-decision-boundary-for-a-space`
> **A classifier's output is a decision over known classes; a metric learner's output is a geometry. The second generalises to unseen classes at the cost of requiring a reference set at inference time, and of making quality depend on the sampling strategy rather than on the labels.**
> ^[generated. rests-on: imported:siamese-open-set-motivation]

## Where It Appears

Face and signature verification, one-shot recognition, image retrieval, and template matching against a reference database — including camera-pose estimation, where a query image is matched against synthetically rendered views.

The pattern generalises past vision: any problem framed as "is this the same thing as that" rather than "which class is this" is a candidate.

## Relation to Modern Practice

Contrastive learning at scale is the same idea with a much larger effective batch of negatives — the shared-encoder, pull-together-push-apart structure is unchanged. The connection to [[representation-learning]] is direct: **a Siamese network is a representation learner whose supervision comes from pair relationships rather than labels.**^[imported]

## See Also

- [[representation-learning]] · [[feature-engineering]] · [[semantic-segmentation]] · [[convolution]] · [[transformer]]
- [[model-selection]] · [[predictive-validity]] · [[feature-pyramid-network]]
