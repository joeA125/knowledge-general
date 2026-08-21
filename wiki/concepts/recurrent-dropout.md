---
title: "Recurrent Dropout"
type: concept
tags: [regularization, deep-learning, rnn, dropout, lstm, training-technique]
sources: [raw/papers/rnn-regularisation.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: archived
created: 2026-05-08
updated: 2026-07-07
---

# Recurrent Dropout

> **Archived duplicate.** This page describes the same concept and draws on the same source (Zaremba et al., 2014) as [[dropout-for-rnns]], which is now the canonical page. Its unique formulation has been merged there. Kept for history; intentionally has no inbound links.

---

Recurrent dropout is the technique of applying [[dropout]] only to the non-recurrent (inter-layer) connections in an RNN, while leaving the recurrent (temporal) connections untouched. Introduced by [[rnn-regularisation|Zaremba et al. (2014)]] (and independently by Pham et al., 2013).

## Why Not Drop Recurrent Connections?

Applying dropout to recurrent connections amplifies noise through time: each timestep compounds the corruption, making it difficult for the [[lstm]] to learn long-term dependencies. By restricting dropout to vertical connections, information flowing along the temporal axis is corrupted exactly $L + 1$ times (where $L$ is the network depth), regardless of sequence length.

## Formulation

For a deep LSTM, replace $h_t^{l-1}$ with $\mathbf{D}(h_t^{l-1})$ in the gate equations:

$$T_{2n,4n} \begin{pmatrix} \mathbf{D}(h_t^{l-1}) \\ h_{t-1}^l \end{pmatrix}$$

where $\mathbf{D}$ is the dropout operator.

## Impact

This technique enabled training much larger LSTMs (e.g. 1500 units per layer vs 200 without regularisation on PTB), achieving dramatically lower perplexity and becoming the standard regularisation approach for RNNs until the [[transformer]] era.

## See Also

- [[dropout-for-rnns]] — canonical page
- [[dropout]]
- [[lstm]]
