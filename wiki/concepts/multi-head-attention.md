---
title: "Multi-Head Attention"
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

# Multi-Head Attention

Multi-head attention is a mechanism in the [[transformer]] that runs multiple [[scaled-dot-product-attention]] functions in parallel, each on a different learned linear projection of queries, keys, and values.

## Formula

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## Why Multiple Heads?

A single attention head averages over all positions, which can inhibit the model's ability to attend to information from different representation subspaces simultaneously. Multiple heads allow the model to jointly capture different types of relationships (e.g., syntactic structure in one head, coreference in another).

## Configuration in the Base Transformer

- $h = 8$ heads
- $d_k = d_v = d_{\text{model}} / h = 64$
- Total compute is similar to single-head attention with full dimensionality

## Empirical Findings

From [[attention-is-all-you-need|the original paper]]: single-head attention is 0.9 BLEU worse than 8 heads; performance also degrades slightly at 32 heads. The 8-head configuration offers the best balance.

## See Also

- [[scaled-dot-product-attention]]
- [[attention-mechanism]]
- [[transformer]]
