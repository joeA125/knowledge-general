---
title: "Identity Mappings in Deep Residual Networks — Source Summary"
type: summary
tags: [deep-learning, architecture, residual-learning, batch-normalization, computer-vision]
sources: [raw/papers/identity-mapping-residual-networks.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Identity Mappings in Deep Residual Networks

**Authors:** [[kaiming-he]], [[xiangyu-zhang]], [[shaoqing-ren]], [[jian-sun]]
**Affiliation:** [[microsoft-research]]
**Published:** 2016 (ECCV 2016); arXiv:1603.05027

## Key Contribution

This paper analyses the propagation formulations behind [[residual-connections]] and shows that using **identity mappings** for both the skip connection $h(\mathbf{x}_l) = \mathbf{x}_l$ and the after-addition activation $f(\mathbf{y}_l) = \mathbf{y}_l$ is essential for smooth information flow. This motivates the [[pre-activation-resnet|pre-activation residual unit]] (BN→ReLU→Weight), which makes training easier and improves generalisation, enabling training of 1001-layer networks.

## Theoretical Analysis

With identity skip connections and identity after-addition activation, any feature $\mathbf{x}_L$ at layer $L$ can be written as:

$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

This means: (i) the feature at any deep layer is the input plus a sum of all preceding residual functions (not a product of matrices); (ii) backpropagation yields a gradient with an additive term of 1 that flows directly without passing through any weight layers — preventing vanishing gradients even with arbitrarily small weights.

## Why Identity Skip Connections Matter

Breaking the identity shortcut (e.g., via scaling $\lambda$, gating, 1×1 convolutions, or dropout) introduces multiplicative factors $\prod_{i=l}^{L-1} \lambda_i$ in the gradient path. For very deep networks, these factors become exponentially large or small, blocking signal propagation.

### Skip Connection Ablations (ResNet-110, CIFAR-10)

| Variant | Error |
|---|---|
| **Identity shortcut (original)** | **6.61%** |
| Constant scaling (0.5/0.5) | 12.35% |
| Exclusive gating (best init) | 8.70% |
| Shortcut-only gating (init $b_g$=−6) | 6.91% |
| 1×1 conv shortcut | 12.22% |
| Dropout shortcut (0.5) | fail |

Gating and 1×1 conv shortcuts have strictly more representational power than identity shortcuts, yet they perform worse — confirming the degradation is caused by optimisation difficulties, not representational limits.

## Pre-Activation Residual Unit

To make the after-addition activation $f$ an identity mapping, BN and ReLU are moved **before** the weight layers ("pre-activation"), yielding: $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\hat{f}(\mathbf{x}_l), \mathcal{W}_l)$.

### Activation Placement Ablations (CIFAR-10)

| Variant | ResNet-110 | ResNet-164 |
|---|---|---|
| Original (post-activation) | 6.61% | 5.93% |
| BN after addition | 8.17% | 6.50% |
| ReLU before addition | 7.84% | 6.14% |
| ReLU-only pre-activation | 6.71% | 5.91% |
| **Full pre-activation (BN+ReLU)** | **6.37%** | **5.46%** |

## Results

- **CIFAR-10:** ResNet-1001 with pre-activation achieves **4.62%** error (vs 7.61% for original ResNet-1001). The original 1001-layer model overfits; the pre-activation version trains smoothly.
- **CIFAR-100:** ResNet-1001 pre-activation achieves **22.71%** (vs 27.82% original).
- **ImageNet:** Pre-activation ResNet-200 achieves **20.7%** top-1 error (vs 21.8% original ResNet-200, which overfits). With scale+aspect ratio augmentation: **20.1%** top-1, outperforming Inception v3.

## Two Benefits of Pre-Activation

1. **Easier optimisation:** Identity $f$ enables direct signal propagation across all layers, critical for extremely deep networks (1000+ layers).
2. **Better regularisation:** BN normalises the input to every weight layer (in the original design, the BN output is added to the shortcut before the next layer, so inputs are unnormalised).

## See Also

- [[pre-activation-resnet]]
- [[residual-connections]]
- [[batch-normalization]]
- [[layer-normalization]]
