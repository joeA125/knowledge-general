---
title: "Feed-Forward Network"
type: concept
tags: [deep-learning, architecture, transformer]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.85
provenance:
  extracted: 60%
  inferred: 35%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Feed-Forward Network

A feed-forward network (FFN) is the simplest neural network topology: information flows in one direction from input to output through one or more layers of learned linear transformations and non-linearities, with no recurrent or feedback connections. In modern architectures the term most often refers to the **position-wise feed-forward sub-layer** found inside each [[transformer]] block.

## Position-wise FFN in the Transformer

Each encoder and decoder layer of the [[transformer]] contains, after its [[multi-head-attention]] sub-layer, a small feed-forward network applied independently and identically to every sequence position:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

This is two linear transformations with a ReLU activation between them. In the base model the inner dimension is $d_{ff} = 2048$ while the input/output dimension is $d_{\text{model}} = 512$, so the FFN expands then contracts the representation.^[inferred: dimensions extracted from source; general framing is standard background]

"Position-wise" means the same two weight matrices are shared across all positions in a sequence, but differ between layers. It can equivalently be viewed as two convolutions with kernel size 1.

## Role

While [[multi-head-attention]] mixes information *across* positions, the FFN transforms each position's representation *in isolation*, adding non-linear modelling capacity. The two sub-layers alternating — attention then FFN — is a defining pattern of the [[transformer]], each wrapped in [[residual-connections]] and [[layer-normalization]].

## See Also

- [[transformer]]
- [[multi-head-attention]]
- [[encoder-decoder]]
- [[residual-connections]]
- [[attention-is-all-you-need|Source Summary]]
