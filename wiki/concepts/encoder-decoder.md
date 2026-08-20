---
title: "Encoder-Decoder Architecture"
type: concept
tags: [architecture, deep-learning, sequence-modelling]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.85
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-07
updated: 2026-05-07
---

# Encoder-Decoder Architecture

The encoder-decoder is a general neural network pattern for sequence transduction where an encoder maps an input sequence $(x_1, \dots, x_n)$ to continuous representations $\mathbf{z} = (z_1, \dots, z_n)$, and a decoder generates an output sequence $(y_1, \dots, y_m)$ autoregressively from $\mathbf{z}$.

## In the Transformer

The [[transformer]] implements this with stacked self-attention layers rather than recurrence:

- **Encoder stack:** 6 layers of [[multi-head-attention]] + [[feed-forward-network]] with [[residual-connections]] and [[layer-normalization]].
- **Decoder stack:** 6 layers adding masked self-attention and encoder-decoder cross-attention.

## Historical Context

Earlier encoder-decoder models used RNNs (Sutskever et al., 2014; Cho et al., 2014) or CNNs (Gehring et al., 2017). The [[transformer]] demonstrated that attention alone suffices.

## See Also

- [[transformer]]
- [[attention-mechanism]]
