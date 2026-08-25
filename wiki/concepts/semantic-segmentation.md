---
title: "Semantic Segmentation"
type: concept
tags: [semantic-segmentation, computer-vision, deep-learning, convolution, dilated-convolution, architecture, evaluation]
sources: [raw/papers/context-aggregation-dilated-convolutions.md]
confidence: 0.8
provenance:
  extracted: 40%
  inferred: 28%
  generated: 6%
  imported: 24%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Semantic Segmentation

Assigning a class label to **every pixel** in an image, rather than one label to the image. The canonical **dense prediction** task, and the setting where the resolution-versus-context tension is sharpest.

## The Central Tension

^[extracted from the held dilated-convolutions source]

A classification network builds context by pooling — each successive layer sees more of the image and resolves it less finely. That is exactly right for "is there a cat" and exactly wrong for "which pixels are cat", where both properties are needed simultaneously:

- **Fine resolution**, to place the boundary correctly
- **Wide context**, because a pixel's class depends on far more than its neighbourhood

Every architecture in this area is a different answer to that one problem.

## Three Answers

| Approach | Mechanism | Cost |
|---|---|---|
| **Encoder–decoder with skips** | Downsample for context, upsample back, fuse across scales | Detail lost in downsampling, recovered imperfectly |
| **[[dilated-convolution\|Dilated convolution]]** | Expand the receptive field in place, without downsampling | More memory; gridding artefacts at high dilation |
| **Multi-scale pyramids** | Predict at several scales and fuse | Redundant computation |

The first is [[fully-convolutional-network|FCN]] and U-Net; the second is the held Yu & Koltun work; the third is [[feature-pyramid-network|FPN]] and its relatives. **They are rarely compared directly on the same task**, which makes the choice between them more conventional than evidenced.

## Evaluation

^[imported]

**Mean intersection-over-union** is standard: per class, the overlap between prediction and ground truth divided by their union, averaged over classes.

Two properties worth knowing:

- **Averaging over classes, not pixels**, so rare classes count as much as common ones. That is deliberate and it means mIoU can fall while pixel accuracy rises.
- **IoU is unforgiving at boundaries.** A thin structure misplaced by a few pixels loses a large fraction of its IoU, which is why boundary-heavy classes score poorly regardless of architecture.

See [[class-imbalance-evaluation]] for the general form of the first point — a metric's aggregation choice determines what it rewards.

## The Supervision Assumption

Segmentation datasets label **every pixel**, which is enormously expensive and is the reason the field is dominated by a handful of benchmarks.

That matters when the architecture is transplanted. **The architectural machinery transfers to any dense-prediction problem; the supervision does not.** A task requiring a value at every location but supplying labels at only a few points needs a different training story, and the architecture alone does not provide one.^[generated]

## See Also

- [[fully-convolutional-network]] · [[dilated-convolution]] · [[feature-pyramid-network]] · [[convolution]] · [[conditional-gan]]
- [[class-imbalance-evaluation]] · [[model-selection]] · [[representation-learning]] · [[siamese-network]]
- [[context-aggregation-dilated-convolutions|Dilated Convolutions Summary]]
