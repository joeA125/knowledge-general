---
title: "Bidirectional RNN"
type: concept
tags: [deep-learning, rnn, sequence-modelling, architecture]
sources: [raw/papers/neural-machine-translation.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Bidirectional RNN

A bidirectional RNN (BiRNN; Schuster & Paliwal, 1997) processes an input sequence in both forward and backward directions, producing two sequences of hidden states that are concatenated to form annotations capturing both left and right context.

## Structure

- **Forward RNN** $\overrightarrow{f}$: reads from $x_1$ to $x_{T_x}$, producing $(\overrightarrow{h}_1, \dots, \overrightarrow{h}_{T_x})$.
- **Backward RNN** $\overleftarrow{f}$: reads from $x_{T_x}$ to $x_1$, producing $(\overleftarrow{h}_1, \dots, \overleftarrow{h}_{T_x})$.
- **Annotation:** $h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$, focused on words around position $j$ due to the recency bias of RNNs.

## In Bahdanau et al. (2014)

The [[neural-machine-translation|Bahdanau attention paper]] uses a BiRNN encoder so that each annotation summarises both preceding and following words. These annotations are then consumed by the [[additive-attention]] mechanism.

## See Also

- [[gated-recurrent-unit]]
- [[additive-attention]]
- [[encoder-decoder]]
