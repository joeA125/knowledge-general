---
title: "Additive Attention"
type: concept
tags: [attention, deep-learning, machine-translation, sequence-modelling]
sources: [raw/papers/neural-machine-translation.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Additive Attention

Additive attention (also known as Bahdanau attention) is the [[attention-mechanism]] introduced in [[neural-machine-translation|Bahdanau et al., 2014]]. It was the first attention mechanism applied to neural machine translation.

## Mechanism

Given decoder hidden state $s_{i-1}$ and encoder annotation $h_j$, the alignment score is computed by a learned feedforward network:

$$e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

Scores are normalised via softmax to produce attention weights:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

The context vector is the weighted sum of annotations:

$$c_i = \sum_j \alpha_{ij} h_j$$

## Why "Additive"?

The compatibility function adds the projected query and key ($W_a s_{i-1} + U_a h_j$) then applies a non-linearity. This contrasts with [[scaled-dot-product-attention]], which computes compatibility as a dot product $q^\top k / \sqrt{d_k}$.

## Computational Note

$U_a h_j$ is independent of the decoding step $i$ and can be precomputed, reducing the per-step cost.

## Relation to Later Work

The [[attention-is-all-you-need|Transformer paper]] noted that additive and dot-product attention have similar theoretical complexity, but dot-product attention is faster in practice due to optimised matrix multiplication. Additive attention outperforms unscaled dot-product at large $d_k$, but [[scaled-dot-product-attention]] closes this gap.

## See Also

- [[attention-mechanism]]
- [[scaled-dot-product-attention]]
- [[encoder-decoder-bottleneck]]
- [[transformer]]
