---
title: "Batch Normalization"
type: concept
tags: [deep-learning, normalization, training-technique]
sources: [raw/papers/identity-mapping-residual-networks.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 50%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-05-08
---

# Batch Normalization

Batch Normalization (BN; Ioffe & Szegedy, 2015) normalises activations across the mini-batch for each feature, reducing internal covariate shift and enabling faster training with higher learning rates.

## Mechanism

For a mini-batch $\mathcal{B}$, each feature is normalised to zero mean and unit variance, then scaled and shifted by learned parameters $\gamma$ and $\beta$:

$$\hat{x} = \frac{x - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

## Role in ResNets

In the original ResNet, BN is applied after each weight layer. The [[pre-activation-resnet]] moves BN before weight layers, which ensures all weight layer inputs are normalised — improving both optimisation and regularisation.

## Relation to Layer Normalization

[[layer-normalization]] normalises across features within a single example (batch-size independent), making it preferred for sequence models like the [[transformer]]. BN normalises across the batch for each feature, making it preferred for CNNs with large, stable batch sizes.

## See Also

- [[layer-normalization]]
- [[pre-activation-resnet]]
- [[residual-connections]]
