---
title: "Recurrent Neural Network Regularization — Source Summary"
type: summary
tags: [deep-learning, rnn, lstm, dropout, regularization, language-modelling, speech-recognition, machine-translation]
sources: [raw/papers/rnn-regularisation.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Recurrent Neural Network Regularization

**Authors:** [[wojciech-zaremba]], [[ilya-sutskever]], [[oriol-vinyals]]
**Affiliations:** New York University (work done at [[google-brain]]), [[google-brain]]
**Published:** 2014 (ICLR 2015); arXiv:1409.2329

## Key Contribution

This paper shows how to correctly apply [[dropout]] to [[lstm]] networks. The key insight is that dropout should only be applied to **non-recurrent connections** (i.e., between layers), not to recurrent connections (within a layer across time steps). This preserves the LSTM's ability to remember information over long time horizons while still regularising the model.

## The Problem

[[dropout]] is the most successful regularisation technique for feedforward neural networks, but does not work well with RNNs when applied naïvely. Applying dropout to recurrent connections amplifies noise and disrupts the LSTM's long-term memory, making it difficult to learn long-range dependencies. As a result, practitioners were forced to use smaller RNNs that underfitted, because larger models would overfit without effective regularisation.

## Method

For a deep LSTM with $L$ layers, the dropout operator $\mathbf{D}$ is applied to $h_t^{l-1}$ (the input from the layer below) but **not** to $h_{t-1}^l$ (the recurrent state from the previous time step):

$$\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \text{sigm} \\ \text{sigm} \\ \text{sigm} \\ \text{tanh} \end{pmatrix} T_{2n,4n} \begin{pmatrix} \mathbf{D}(h_t^{l-1}) \\ h_{t-1}^l \end{pmatrix}$$

Information flowing through the network is corrupted by dropout exactly $L + 1$ times (independent of the number of time steps), preserving the LSTM's memorisation ability.

## Results

### Language Modelling (Penn Treebank)

| Model | Validation PPL | Test PPL |
|---|---|---|
| Non-regularised LSTM | 120.7 | 114.5 |
| Medium regularised LSTM (650 units, 50% dropout) | 86.2 | 82.7 |
| Large regularised LSTM (1500 units, 65% dropout) | 82.2 | **78.4** |

The large regularised LSTM achieved new state-of-the-art single-model perplexity. Ensembles of 38 large regularised LSTMs reached 68.7 test perplexity.

### Speech Recognition (Icelandic)
- Regularised LSTM: **70.5%** validation frame accuracy (vs 68.9% without dropout).
- Training accuracy dropped (69.4% vs 71.6%), confirming the noise prevents overfitting.

### Machine Translation (English→French, WMT'14)
- Regularised LSTM: **29.03 BLEU** (vs 25.9 without dropout).
- 4-layer LSTM, 1000 units, 20% dropout.

### Image Caption Generation (MSCOCO)
- Regularised: 24.3 BLEU / 7.99 perplexity (vs 23.5 / 8.47 without).
- Dropout mainly helps single models match ensemble performance.

## Training Details

- **Medium LSTM:** 2 layers × 650 units; SGD, lr=1, decay factor 1.2 after epoch 6; gradient clipping at 5; 39 epochs; ~half day on K20 GPU.
- **Large LSTM:** 2 layers × 1500 units; SGD, lr=1, decay factor 1.15 after epoch 14; gradient clipping at 10; 55 epochs; ~1 day on K20 GPU.
- Unrolled for 35 time steps, minibatch size 20.
- Hidden states carried across successive minibatches.

## See Also

- [[dropout-for-rnns]]
- [[dropout]]
- [[lstm]]
- [[gated-recurrent-unit]]
