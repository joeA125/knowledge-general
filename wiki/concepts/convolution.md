---
title: "Convolution"
type: concept
tags: [deep-learning, architecture, computer-vision]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.8
provenance:
  extracted: 35%
  inferred: 60%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Convolution

Convolution is an operation that slides a small learnable filter (kernel) over an input signal, computing a weighted sum at each position to produce a feature map. It is the core building block of convolutional neural networks (CNNs), which dominate computer vision and also appear in sequence modelling.

## Properties

- **Local receptive field:** each output depends only on a small neighbourhood of the input.
- **Weight sharing:** the same kernel is applied at every position, giving translation equivariance and far fewer parameters than a dense layer.
- **Stacking for range:** the receptive field grows with depth; a single layer connects only nearby positions, so capturing long-range structure requires many layers (or [[dilated-convolution|dilation]]).

## Relation to the Transformer

The [[transformer]] deliberately uses neither recurrence nor convolution.^[inferred: the source contrasts self-attention against convolution rather than describing convolution in depth] The [[attention-is-all-you-need|Attention Is All You Need]] paper compares them by maximum path length: connecting two distant positions requires a stack of convolutional layers (path length $O(n / k)$ or $O(\log_k n)$ for dilated kernels), whereas self-attention links any two positions in $O(1)$. The position-wise [[feed-forward-network]] inside a Transformer block is equivalent to two kernel-size-1 convolutions.

## Elsewhere in the Vault

Convolutions underpin the vision architectures discussed in [[feature-pyramid-network]], [[residual-connections|ResNet]], and [[dilated-convolution]].

## See Also

- [[dilated-convolution]]
- [[feature-pyramid-network]]
- [[recurrence]]
- [[transformer]]
