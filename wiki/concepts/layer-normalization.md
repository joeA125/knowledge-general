---
title: "Layer Normalization"
type: concept
tags: [normalization, deep-learning, training-technique]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 50%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-07
updated: 2026-05-07
---

# Layer Normalization

Layer normalization (Ba et al., 2016) normalises activations across features within a single training example, unlike batch normalization which normalises across the batch. It stabilises training and is independent of batch size.

## In the Transformer

The [[transformer]] applies layer normalization after each [[residual-connections|residual connection]]: $\text{LayerNorm}(x + \text{Sublayer}(x))$. This is used in both encoder and decoder stacks.

## See Also

- [[residual-connections]]
- [[transformer]]
