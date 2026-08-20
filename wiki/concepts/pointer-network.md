---
title: "Pointer Network"
type: concept
tags: [deep-learning, attention, sequence-modelling, architecture, pointer-mechanism, combinatorial-optimisation]
sources: [raw/papers/pointer-networks.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Pointer Network

A Pointer Network (Ptr-Net; [[oriol-vinyals|Vinyals]] et al., 2015) is a sequence-to-sequence architecture that uses [[additive-attention]] as a pointer to select members of the input sequence as the output, rather than producing outputs from a fixed vocabulary.

## Mechanism

Given encoder hidden states $(e_1, \dots, e_n)$ and decoder state $d_i$:

$$u_j^i = v^T \tanh(W_1 e_j + W_2 d_i) \quad j \in (1, \dots, n)$$
$$p(C_i | C_1, \dots, C_{i-1}, \mathcal{P}) = \text{softmax}(u^i)$$

The output distribution has size $n$ (the input length), which varies per instance. The decoder is conditioned on the previous output by feeding the corresponding input element $P_{C_{i-1}}$.

## Key Innovation

Standard [[additive-attention]] blends encoder states into a context vector. The Ptr-Net instead uses attention scores directly as output probabilities — attention becomes a pointer rather than a soft lookup. This enables handling of variable-size output dictionaries, which standard sequence-to-sequence models cannot do.

## Applications

The original paper demonstrated Ptr-Nets on three [[combinatorial-optimisation]] problems: planar convex hulls ($O(n \log n)$ exact), Delaunay triangulations ($O(n \log n)$ exact), and the planar Travelling Salesman Problem (NP-hard). The model learns approximate solutions purely from training examples and generalises to longer sequences than seen during training.

## Relation to Other Work

- Extends [[additive-attention]] (Bahdanau et al., 2014) from an internal mechanism to an output mechanism.
- Related to copy mechanisms in NMT and abstractive summarisation.
- Inspired later work on neural combinatorial optimisation.
- The pointer concept was later influential in models like CopyNet and the Transformer's cross-attention.

## See Also

- [[additive-attention]]
- [[attention-mechanism]]
- [[encoder-decoder]]
- [[pointer-networks|Source Summary]]
