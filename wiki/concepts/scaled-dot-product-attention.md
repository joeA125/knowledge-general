---
title: "Scaled Dot-Product Attention"
type: concept
tags: [attention, transformer, deep-learning]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-05-07
---

# Scaled Dot-Product Attention

Scaled dot-product attention is the core [[attention-mechanism]] used inside the [[transformer]].

## Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$, $K$, $V$ are matrices of queries, keys, and values, and $d_k$ is the key dimensionality.

## Why Scale?

For large $d_k$, unscaled dot products grow in magnitude (variance = $d_k$ when components are unit-variance), pushing the softmax into saturation where gradients vanish. Dividing by $\sqrt{d_k}$ normalises the variance back to 1.

## Masking

In decoder self-attention, illegal connections (attending to future positions) are masked by setting those entries to $-\infty$ before the softmax, enforcing the autoregressive property.

## See Also

- [[attention-mechanism]]
- [[multi-head-attention]]
- [[transformer]]
