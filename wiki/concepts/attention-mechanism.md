---
title: "Attention Mechanism"
type: concept
tags: [attention, deep-learning, sequence-modelling, interpretability, transformer, architecture]
sources: [raw/papers/attention-is-all-you-need.md, raw/papers/neural-machine-translation.md, raw/papers/pointer-networks.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-08-14
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

Where a fixed-size context window feeds a self-attention encoder, the **distribution of weights across that window** tests whether the window is correctly sized. Weights piling up at the recent end suggest the window is too short and more history would help; distant positions receiving negligible weight suggest it is longer than the model can use. A flat distribution with no trend indicates neither.

> ### `attention-weights-diagnose-window-length`
> **The attention distribution over a fixed context window is a cheap test of whether the window is correctly sized — a trend toward either end indicates a misspecified horizon. The check costs nothing beyond reading a matrix the model already computes.**
> ^[generated. rests-on: imported:attention-window-diagnostics]

This matters because context length is otherwise an **asserted parameter**, set by convention and rarely varied. See [[model-selection]].

A caveat worth keeping: attention weights are widely used as [[interpretability|interpretability]] evidence, and the literature is divided on whether they constitute genuine explanation. **The diagnostic use is more defensible than the explanatory one**, since it asks only whether the model *distributes* attention across the window, not what any individual weight means. See `diagnostic-use-survives-what-explanatory-use-does-not` on [[interpretability]].

## Beyond Language

Attention is agnostic about what a "sequence" contains. It appears over word sequences ([[transformer]], [[additive-attention]]), sets ([[read-process-write]]), memory locations ([[neural-turing-machine]]), image patches for retrieval ([[siamese-network]]), and continuous-time event streams ([[neural-temporal-point-process]]) — the mechanism is unchanged; only the tokens differ.

Self-attention over a set with no positional encoding is [[message-passing]] on a fully-connected graph, which places it in the same family as [[graph-neural-network|GNNs]].

## See Also

- [[additive-attention]] · [[scaled-dot-product-attention]] · [[multi-head-attention]] · [[pointer-network]] · [[transformer]]
- [[encoder-decoder-bottleneck]] · [[neural-temporal-point-process]] · [[message-passing]] · [[graph-neural-network]]
- [[interpretability]] · [[model-selection]] · [[lstm]] · [[gated-recurrent-unit]]
- [[attention-is-all-you-need|Transformer Summary]] · [[neural-machine-translation|Bahdanau Summary]] · [[pointer-networks|Pointer Networks Summary]]
