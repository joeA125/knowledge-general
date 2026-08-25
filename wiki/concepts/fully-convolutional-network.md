---
title: "Fully Convolutional Network"
type: concept
tags: [architecture, deep-learning, computer-vision, semantic-segmentation, convolution, representation-learning]
sources: [raw/papers/context-aggregation-dilated-convolutions.md]
confidence: 0.75
provenance:
  extracted: 25%
  inferred: 40%
  generated: 8%
  imported: 25%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Fully Convolutional Network

A network built entirely from convolutional layers, with no fully-connected layers — so the output is a **spatial map** rather than a vector. Introduced for dense prediction by Long, Shelhamer & Darrell (2015).^[imported]

## The Structural Change

A conventional classification CNN ends by flattening its feature map and passing it through dense layers to a fixed-size output. Two things follow: **the input size is fixed**, and **all spatial arrangement is destroyed at the flatten**.

Removing the dense layers removes both constraints. The output becomes a grid whose cells correspond to input regions, so the network answers a question *per location* rather than once per input.

The consequence that matters most is **weight sharing across positions**. The same filters run everywhere, so the network learns a function of local context rather than a lookup keyed to absolute position.

> ### `weight-sharing-is-what-makes-sparse-supervision-viable`
> **Because a convolutional network applies one function at every position, a gradient from a label at a single location updates the function governing all locations. That is why dense prediction can be learned from sparse labels at all — and why the same is not true of an architecture with position-specific parameters.**
> ^[generated. rests-on: imported:fcn-weight-sharing]

## The Resolution Problem

Pooling and strided convolution build large receptive fields and destroy resolution. Dense prediction needs both. Three families of solution:

| Approach | Mechanism |
|---|---|
| **Encoder–decoder with skips** | Downsample, upsample, fuse across scales — the original FCN and U-Net |
| **[[dilated-convolution\|Dilated convolution]]** | Expand the receptive field in place, without downsampling |
| **Multi-scale pyramids** | Predict at several scales and fuse — [[feature-pyramid-network\|FPN]] |

They are complementary rather than exclusive, and **rarely compared directly on the same task**, which makes the choice between them more conventional than evidenced. See `multi-scale-fusion-and-dilation-solve-the-same-problem-differently` on [[feature-pyramid-network]].

## Beyond Images

Nothing about the architecture requires pixels. It requires a domain with a **regular grid and translation-equivariant structure** — where the same local pattern means the same thing wherever it appears.

That condition is more restrictive than it sounds, and where it fails partially, the standard repair is instructive: **supply the broken symmetry as an input channel rather than changing the architecture.**

A spatial domain with a distinguished location — a target, a boundary, a source — is not fully translation-invariant. Adding distance-to-target and angle-to-target as explicit layers lets the network recover the asymmetry from features while keeping the convolutional structure intact. See [[feature-engineering]] and [[theory-based-modelling]].

## See Also

- [[semantic-segmentation]] · [[dilated-convolution]] · [[feature-pyramid-network]] · [[convolution]] · [[residual-connections]]
- [[representation-learning]] · [[feature-engineering]] · [[theory-based-modelling]] · [[siamese-network]] · [[conditional-gan]]
- [[context-aggregation-dilated-convolutions|Dilated Convolutions Summary]]