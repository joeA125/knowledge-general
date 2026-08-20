---
title: "Multi-Scale Context Aggregation by Dilated Convolutions — Source Summary"
type: summary
tags: [deep-learning, computer-vision, semantic-segmentation, dilated-convolution, architecture]
sources: [raw/papers/context-aggregation-dilated-convolutions.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Multi-Scale Context Aggregation by Dilated Convolutions

**Authors:** [[fisher-yu]], [[vladlen-koltun]]
**Affiliations:** Princeton University, Intel Labs
**Published:** 2016 (ICLR 2016); arXiv:1511.07122

## Key Contribution

This paper introduces [[dilated-convolution]]s (also known as atrous convolutions) as a systematic tool for [[semantic-segmentation]], and proposes a plug-in context module that aggregates multi-scale contextual information without losing resolution or coverage. It also shows that simplifying classification networks adapted for dense prediction improves accuracy.

## The Problem

Semantic segmentation requires both pixel-level precision and multi-scale contextual reasoning. Classification networks achieve global context via successive pooling/subsampling, but this destroys resolution. Prior approaches either (a) used up-convolutions to recover lost resolution, or (b) processed multiple rescaled copies of the image. Neither is ideal.

## Dilated Convolutions

A dilated convolution with dilation factor $l$ applies a filter kernel at spaced-out positions:

$$(F *_l k)(\mathbf{p}) = \sum_{\mathbf{s}+l\mathbf{t}=\mathbf{p}} F(\mathbf{s}) k(\mathbf{t})$$

Standard convolution is the special case $l = 1$. With exponentially increasing dilation ($l = 1, 2, 4, 8, \dots$), the receptive field grows exponentially while the number of parameters grows linearly — and no resolution is lost.

With $3 \times 3$ filters and dilation factors $2^i$, the receptive field at layer $i+1$ is $(2^{i+2} - 1) \times (2^{i+2} - 1)$.

## Context Module

A plug-in module of 7 convolutional layers with dilations 1, 1, 2, 4, 8, 16, 1, plus a final $1 \times 1$ convolution. Takes $C$ feature maps as input and produces $C$ feature maps as output, achieving a $67 \times 67$ receptive field with only $\approx 64C^2$ parameters.

Two variants:
- **Basic:** All layers have $C$ channels.
- **Large:** Channels increase in deeper layers (up to $32C$), giving more capacity.

Initialised with identity initialization (each layer passes input directly to the next); backpropagation then learns to aggregate contextual information.

## Front-End Simplification

Adapted VGG-16 for dense prediction by removing the last two pooling/striding layers entirely (rather than keeping them as in FCN-8s or replacing stride with dilation as in DeepLab). Also removed intermediate padding. This simplified front end alone achieves 67.6% mIoU on VOC-2012 test — outperforming DeepLab+CRF (66.4%) without any structured prediction.

## Results

### Pascal VOC 2012 (test set)
| Configuration | mIoU |
|---|---|
| Front end alone | 71.3% |
| Front end + Large context | 73.5% |
| Front end + Large context + CRF | 74.7% |
| Front end + Large context + CRF-RNN | **75.3%** |

The context module improves accuracy in all configurations (no structured prediction, +CRF, +CRF-RNN), demonstrating it is synergistic with structured prediction.

### Urban Scene Understanding
| Dataset | Model | mIoU |
|---|---|
| CamVid | Dilation8 | **65.3%** |
| KITTI | Dilation7 | **59.2%** |
| Cityscapes (test) | Dilation10 | **67.1%** |

All results outperformed prior state-of-the-art at the time of publication.

## Impact

Dilated convolutions became a standard tool in dense prediction, adopted by DeepLab v2/v3, WaveNet (for audio), and many other architectures. The key insight — exponential receptive field growth without resolution loss — complemented [[residual-connections]] and attention as a fundamental building block.

## See Also

- [[dilated-convolution]]
- [[semantic-segmentation]]
- [[residual-connections]]
