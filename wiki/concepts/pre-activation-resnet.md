---
title: "Pre-Activation ResNet"
type: concept
tags: [deep-learning, architecture, residual-learning, computer-vision]
sources: [raw/papers/identity-mapping-residual-networks.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Pre-Activation ResNet

The pre-activation residual unit (He et al., 2016) rearranges the components of a [[residual-connections|residual block]] so that [[batch-normalization]] and ReLU appear **before** the weight layers, rather than after. This makes the after-addition function an identity mapping, enabling cleaner signal propagation through extremely deep networks.

## Original vs Pre-Activation

- **Original (post-activation):** Weight → BN → ReLU → Weight → BN → Addition → ReLU
- **Pre-activation:** BN → ReLU → Weight → BN → ReLU → Weight → Addition

In the pre-activation design, the skip connection carries an unmodified identity signal from input to output, and the addition is not followed by any non-linearity.

## Why It Works

With identity skip connections and identity after-addition activation, any layer's feature can be expressed as the sum of a shallow feature plus all preceding residual functions. The gradient decomposes into a direct term (that bypasses all weight layers) plus a residual term. This prevents vanishing gradients even in 1000+ layer networks.

## Key Results

- ResNet-1001 on CIFAR-10: 4.62% error (pre-activation) vs 7.61% (original) — the original overfits while pre-activation trains smoothly.
- ResNet-200 on ImageNet: 20.7% top-1 (pre-activation) vs 21.8% (original).

## Dual Benefit

1. **Optimisation:** Clean information paths make extremely deep networks trainable.
2. **Regularisation:** BN normalises inputs to all weight layers (unlike the original design where the shortcut addition disrupts normalisation).

## See Also

- [[residual-connections]]
- [[batch-normalization]]
- [[identity-mapping-residual-networks|Source Summary]]
