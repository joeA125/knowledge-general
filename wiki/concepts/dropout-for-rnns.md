---
title: "Dropout for RNNs"
type: concept
tags: [deep-learning, rnn, lstm, dropout, regularization]
sources: [raw/papers/rnn-regularisation.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-07-07
---

# Dropout for RNNs

Dropout for RNNs (Zaremba et al., 2014) is a technique for applying [[dropout]] to [[lstm]] and other recurrent architectures. The key principle is to apply dropout **only to non-recurrent connections** — the vertical connections between layers — and **not to recurrent connections** within a layer across time steps.

## Why Standard Dropout Fails for RNNs

Applying dropout to recurrent connections amplifies noise over time, destroying the LSTM's ability to maintain long-term memory. Each time step compounds the corruption, making it impossible for the network to learn long-range dependencies.

## The Solution

In a deep LSTM with $L$ layers, dropout is applied to $h_t^{l-1}$ (input from the layer below) but not to $h_{t-1}^l$ (recurrent state). This ensures information flowing through the network is corrupted by dropout exactly $L + 1$ times, regardless of how many time steps the information traverses.

## Formulation

Concretely, the dropout operator $\mathbf{D}$ is applied to the input from the layer below within the LSTM gate computation, leaving the recurrent state untouched:

$$T_{2n,4n} \begin{pmatrix} \mathbf{D}(h_t^{l-1}) \\ h_{t-1}^l \end{pmatrix}$$

## Effect

This approach enables training much larger LSTMs without overfitting. On Penn Treebank language modelling, a large regularised LSTM (1500 units) with 65% dropout achieved 78.4 test perplexity, far outperforming a non-regularised LSTM (200 units) at 114.5 perplexity.

## Relation to Other Work

- Independently discovered by Pham et al. (2013) for handwriting recognition.
- Later work introduced variational dropout (Gal & Ghahramani, 2016), which uses the same dropout mask across time steps.
- The [[transformer]] uses standard dropout (not recurrence-aware) since it has no recurrent connections.

## See Also

- [[dropout]]
- [[regularization]]
- [[lstm]]
- [[rnn-regularisation|Source Summary]]

---

*Note: this page is the canonical home for the "recurrent dropout" concept. The former `recurrent-dropout` page was an archived duplicate merged here on 2026-07-07.*
