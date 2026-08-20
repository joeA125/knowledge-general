---
title: "Fully Convolutional Network"
type: concept
tags: [architecture, deep-learning, computer-vision, semantic-segmentation, probability-surface]
sources: [raw/papers/expected_value_possession_framework.md, raw/papers/context-aggregation-dilated-convolutions.md]
confidence: 0.8
provenance:
  extracted: 45%
  inferred: 50%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-27
updated: 2026-07-27
---

# Fully Convolutional Network

A network built entirely from convolutional layers, with no fully-connected layers — so the output is a **spatial map** rather than a vector. Introduced for dense prediction by Long, Shelhamer & Darrell (2015).

## The Structural Change

A conventional classification CNN ends by flattening its feature map and passing it through dense layers to a fixed-size output. Two things follow: the input size is fixed, and all spatial arrangement is destroyed at the flatten.

Removing the dense layers removes both constraints. The output becomes a grid whose cells correspond to input regions, so the network can answer a question *per location* instead of once per input.

The consequence that matters most is **weight sharing across positions**. The same filters run everywhere, so the network learns a function of local context rather than a lookup keyed to absolute position. That is what makes [[single-pixel-supervision]] possible: a gradient at one location updates the function governing all of them.

## The Resolution Problem

Pooling and strided convolution build large receptive fields but destroy resolution — and dense prediction needs both wide context and fine output. Three families of solution:

| Approach | Mechanism | In this vault |
|---|---|---|
| Encoder–decoder with skips | Downsample, upsample, fuse across scales | [[soccermap]], original FCN |
| [[dilated-convolution\|Dilated convolution]] | Expand receptive field without downsampling | [[context-aggregation-dilated-convolutions\|Yu & Koltun]] |
| Multi-scale pyramids | Predict at several scales, fuse | [[feature-pyramid-network]] |

These are complementary rather than exclusive. [[soccermap]] takes the first route — predictions at 1×, ½× and ¼×, upsampled and merged through linear 1×1 fusion layers — and Yu & Koltun's dilated-convolution work, already held in the vault, is the direct alternative to it.

## Beyond Images

Nothing about the architecture requires pixels. It requires a domain with a **regular grid and translation-equivariant structure**, where the same local pattern means the same thing wherever it appears.

A football pitch qualifies, which is the observation [[soccermap]] rests on: a defender two metres from the ball means much the same thing in either half. The pitch is discretised to $104 \times 68$, tracking data is rendered as stacked layers, and segmentation machinery applies unchanged.

The analogy is imperfect in one respect worth noting. Images are genuinely translation-invariant; pitches are not entirely — the goals break the symmetry. The framework handles this by including distance-to-goal and angle-to-goal as explicit input layers, letting the network recover the asymmetry from features rather than architecture.

## See Also

- [[soccermap]] · [[probability-surface]] · [[single-pixel-supervision]]
- [[semantic-segmentation]] · [[dilated-convolution]] · [[feature-pyramid-network]]
- [[convolution]] · [[residual-connections]]
- [[context-aggregation-dilated-convolutions|Dilated Convolutions Summary]]
- [[expected-value-possession-framework|Source Summary]]
