---
title: "Attention Mechanism"
type: concept
tags: [attention, deep-learning, sequence-modelling, interpretability]
sources: [raw/papers/attention-is-all-you-need.md, raw/papers/neural-machine-translation.md, raw/papers/pointer-networks.md, raw/papers/transformer-point-process-football-event-modelling.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-07-23
---

# Attention Mechanism

An attention mechanism maps a query and a set of key-value pairs to an output, computed as a weighted sum of the values where weights are derived from a compatibility function between the query and each key.

## Variants

- **[[additive-attention]]** (Bahdanau et al., 2014): Uses a learned feed-forward network to compute compatibility. The first attention mechanism applied to NMT. Similar theoretical complexity to dot-product but slower in practice.
- **Dot-product / multiplicative attention:** Computes compatibility as $QK^T$. Faster due to optimised matrix multiplication.
- **[[scaled-dot-product-attention]]:** Scales by $1/\sqrt{d_k}$ to prevent softmax saturation at large $d_k$. Used in the [[transformer]].
- **[[multi-head-attention]]:** Runs multiple attention functions in parallel on linearly projected subspaces, then concatenates results.

## Attention as Output (Pointer Mechanism)

[[pointer-network|Pointer Networks]] (Vinyals et al., 2015) repurpose attention scores as the output distribution rather than using them to blend encoder states. This enables variable-size output dictionaries for combinatorial problems.

## Self-Attention

Self-attention (intra-attention) relates positions within a single sequence to compute a contextual representation. In the [[transformer]], self-attention replaces recurrence entirely.

## Role in the Transformer

The [[transformer]] uses attention in three ways:
1. **Encoder self-attention:** Each position attends to all positions in the previous layer.
2. **Decoder masked self-attention:** Each position attends only to earlier positions (preserving autoregression).
3. **Encoder-decoder cross-attention:** Decoder queries attend to all encoder outputs.

## Attention Weights as Diagnostic Output

Beyond their computational role, attention weights are inspectable, and can be read as a diagnostic on the model's *inputs* rather than its internals.

[[transformer-point-process-football-event-modelling|Yeung et al. (2023)]] use the last row of the self-attention matrix — the contribution of each historical event to the final history representation — to test whether their 40-event context window is appropriately sized. Weights lay between 0.01 and 0.06 with **no trend across the window**, which they take as evidence the window is neither too short (weights would pile up at the recent end, implying more history could help) nor too long (distant events would receive negligible weight).

This is a cheap and general check on context-length hyperparameters, applicable wherever a fixed-size window feeds a self-attention encoder.

A caveat worth keeping: attention weights are widely used as [[interpretability|interpretability]] evidence, but the broader literature is divided on whether they constitute genuine explanation of model behaviour. The diagnostic use above is more defensible than causal claims, since it asks only whether the model *distributes* attention across the available window rather than what any individual weight means.

## Beyond Language

Attention is architecture-agnostic about what a "sequence" contains. In this vault it appears over word sequences ([[transformer]], [[additive-attention]]), sets ([[read-process-write]]), memory locations ([[neural-turing-machine]]), image patches for retrieval ([[siamese-network]] in calibration), and football match events ([[nmstpp]]) — the mechanism is unchanged; only the tokens differ.

## See Also

- [[additive-attention]]
- [[scaled-dot-product-attention]]
- [[multi-head-attention]]
- [[pointer-network]]
- [[encoder-decoder-bottleneck]]
- [[neural-temporal-point-process]]
- [[transformer]]
