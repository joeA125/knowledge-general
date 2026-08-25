---
title: "Feature Pyramid Network"
type: concept
tags: [architecture, computer-vision, object-detection, semantic-segmentation, convolution, deep-learning, residual-learning]
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

# Feature Pyramid Network

> ⚠️ **No held source.** Background knowledge, marked `imported:`. Lin et al. (2017) is the primary source.

An architecture for detecting or segmenting objects **across a wide range of scales**, by building a feature pyramid with strong semantics at every level.

## The Scale Problem

A convolutional backbone already computes a hierarchy: early layers are high-resolution and semantically weak, late layers are low-resolution and semantically strong.

Using it naively forces a bad choice:

- **Predict from late layers only** — good semantics, and small objects have been downsampled away
- **Predict from early layers only** — good resolution, and no semantic content to classify with
- **Run the whole network on a resized image pyramid** — works, and multiplies compute by the number of scales

FPN's contribution is to get strong semantics at *every* resolution for roughly the cost of one forward pass.

## The Mechanism

A **top-down pathway with lateral connections**:

1. The bottom-up pathway is the ordinary backbone.
2. The top-down pathway upsamples the deepest, most semantic features.
3. **Lateral connections** merge each upsampled level with the same-resolution backbone level, via a $1\times1$ convolution to match channel depth.

The result is a pyramid where every level is both well-resolved and semantically meaningful. Predictions are made independently at each level, with each assigned the object scales it is best suited to.

## Where the Idea Recurs

The pattern — **downsample for context, upsample back, fuse across scales via skip connections** — is not specific to detection:

| Architecture | Same pattern, applied to |
|---|---|
| **FPN** | Object detection |
| **U-Net** | Biomedical dense prediction |
| **FCN with skips** | [[semantic-segmentation\|Semantic segmentation]] |

> ### `multi-scale-fusion-and-dilation-solve-the-same-problem-differently`
> **Pyramid fusion and [[dilated-convolution\|dilated convolution]] are alternative answers to the resolution-versus-context tension: one loses detail and reconstructs it, the other never loses it and pays in memory. They are rarely compared directly on the same task, so the choice between them is more often conventional than evidenced.**
> ^[generated. rests-on: imported:fpn-architecture, source:yu-koltun-dilation]

That comparison is cheap to run and would be informative wherever a dense-prediction architecture is being selected rather than inherited.

## Relation to Residual Backbones

FPN is a **neck**, not a backbone — it sits between a feature extractor and a prediction head, and composes with whatever provides the hierarchy. [[residual-connections|Residual]] networks are the usual choice because they supply a clean multi-stage hierarchy with stable gradients.

The separation matters: backbone, neck and head are independently swappable, which is why architecture papers in this area report combinations rather than monolithic designs.

## See Also

- [[semantic-segmentation]] · [[fully-convolutional-network]] · [[dilated-convolution]] · [[convolution]] · [[residual-connections]]
- [[transformer]] · [[representation-learning]] · [[model-selection]] · [[siamese-network]]
