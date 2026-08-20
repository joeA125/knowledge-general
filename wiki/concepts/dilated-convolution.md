---
title: "Dilated Convolution"
type: concept
tags: [deep-learning, architecture, computer-vision, dilated-convolution]
sources: [raw/papers/context-aggregation-dilated-convolutions.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Dilated Convolution

A dilated convolution (also called atrous convolution) generalises the standard discrete convolution by introducing a dilation factor $l$ that spaces out the filter elements:

$$(F *_l k)(\mathbf{p}) = \sum_{\mathbf{s}+l\mathbf{t}=\mathbf{p}} F(\mathbf{s}) k(\mathbf{t})$$

Standard convolution is the $l = 1$ case. The operator plays a key role in the algorithme à trous for wavelet decomposition (Holschneider et al., 1987).

## Key Property: Exponential Receptive Field Growth

With $3 \times 3$ filters at exponentially increasing dilations ($l = 1, 2, 4, \dots, 2^i$), the receptive field at layer $i+1$ is $(2^{i+2} - 1) \times (2^{i+2} - 1)$, growing exponentially while:
- The number of parameters grows only linearly (each layer has the same $3 \times 3$ kernel).
- No resolution is lost (no pooling or striding).
- No coverage gaps appear (unlike naïve sparse sampling).

## Context Module

[[context-aggregation-dilated-convolutions|Yu & Koltun (2016)]] designed a plug-in context module using 7 layers of $3 \times 3$ dilated convolutions (dilations 1, 1, 2, 4, 8, 16, 1) plus a $1 \times 1$ output layer, achieving a $67 \times 67$ receptive field with only $\approx 64C^2$ parameters. Initialised with identity initialization and trained to aggregate multi-scale context for [[semantic-segmentation]].

## Adoption

Dilated convolutions became widely adopted: DeepLab v2/v3 (Atrous Spatial Pyramid Pooling), WaveNet (causal dilated convolutions for audio generation), and many other dense prediction and generative architectures.

## Distinction

"Dilated convolution" refers to the modified convolution operator — not to constructing a physically dilated (zero-inserted) filter kernel. The implementation applies the original kernel at spaced positions.

## See Also

- [[semantic-segmentation]]
- [[context-aggregation-dilated-convolutions|Source Summary]]
