---
title: "Residual Connections"
type: concept
tags: [architecture, deep-learning, training-technique, residual-learning]
sources: [raw/papers/attention-is-all-you-need.md, raw/papers/identity-mapping-residual-networks.md]
confidence: 0.9
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-05-08
---

# Residual Connections

Residual connections (He et al., 2016) add the input of a sub-layer to its output: $\text{output} = x + \text{Sublayer}(x)$. This creates a shortcut path for gradients, enabling training of much deeper networks.

## Theory

With identity skip connections and identity after-addition activation, the feature at any layer $L$ is the input plus the sum of all preceding residual functions: $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$. The gradient contains an additive term of 1 that flows directly without passing through any weights, preventing vanishing gradients.

## Importance of Identity Mappings

[[identity-mapping-residual-networks|He et al. (2016b)]] showed that any modification to the skip connection (scaling, gating, 1×1 convolutions, dropout) introduces multiplicative factors that impede signal propagation in very deep networks. Identity shortcuts consistently achieve the best results despite having less representational power than parameterised alternatives.

## Pre-Activation

The [[pre-activation-resnet]] moves [[batch-normalization]] and ReLU before the weight layers, making the after-addition function an identity mapping. This enables training of 1000+ layer networks.

## In the Transformer

The [[transformer]] wraps every sub-layer (attention and feed-forward) with a residual connection followed by [[layer-normalization]]: $\text{LayerNorm}(x + \text{Sublayer}(x))$. All sub-layers and embedding layers produce outputs of the same dimension $d_{\text{model}} = 512$.

## See Also

- [[pre-activation-resnet]]
- [[batch-normalization]]
- [[layer-normalization]]
- [[transformer]]
