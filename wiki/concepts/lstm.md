---
title: "Long Short-Term Memory"
type: concept
tags: [deep-learning, rnn, lstm, architecture, sequence-modelling]
sources: [raw/papers/rnn-regularisation.md, raw/papers/neural-machine-translation.md, raw/papers/transformer-point-process-football-event-modelling.md]
confidence: 0.9
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-07-23
---

# Long Short-Term Memory (LSTM)

The LSTM (Hochreiter & Schmidhuber, 1997) is a recurrent neural network architecture designed to learn long-term dependencies by maintaining explicit memory cells and using gating mechanisms to control information flow.

## Architecture

The LSTM cell uses four gates computed from the current input $h_t^{l-1}$ and previous hidden state $h_{t-1}^l$:

- **Input gate** $i$: controls what new information to store.
- **Forget gate** $f$: controls what information to discard from the cell.
- **Output gate** $o$: controls what information to output.
- **Input modulation gate** $g$: candidate values to add to the cell.

$$\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \text{sigm} \\ \text{sigm} \\ \text{sigm} \\ \text{tanh} \end{pmatrix} T_{2n,4n} \begin{pmatrix} h_t^{l-1} \\ h_{t-1}^l \end{pmatrix}$$

$$c_t^l = f \odot c_{t-1}^l + i \odot g$$
$$h_t^l = o \odot \tanh(c_t^l)$$

## Why LSTMs Work

The cell state $c_t$ provides a highway for gradients to flow across many time steps with minimal decay, addressing the vanishing gradient problem. The gates learn when to read, write, and reset the memory.

## LSTM vs Transformer for Sequence Encoding

The [[transformer]] displaced recurrent encoders for most long-sequence tasks, but the tradeoff is narrower than the displacement suggests. The recurring empirical finding is that **LSTMs remain competitive or marginally better on accuracy, while being substantially slower to train**, because their gradient computation is inherently sequential where self-attention parallelises.

[[transformer-point-process-football-event-modelling|Yeung et al. (2023)]] give a clean measurement on football event sequences, holding everything else fixed:

| Encoder | Total loss | Training time |
|---|---|---|
| Uni-LSTM | **4.51** | 129 min |
| Transformer | 4.57 | **47 min** |

The LSTM wins on loss by 0.06; the transformer trains **2.7× faster**. The same pattern was reported by [[seq2event|Simpson et al. (2022)]] and is noted across the [[neural-temporal-point-process|NTPP]] literature.

This is why LSTMs persist in settings where sequences are short, training budget is tight relative to inference budget, or parameter count matters — the LSTM variant above used 4K parameters against the transformer's 13K.

## Relation to GRU

The [[gated-recurrent-unit]] (Cho et al., 2014) simplifies the LSTM by merging the forget and input gates into a single update gate and combining the cell and hidden states, yielding fewer parameters.

## Regularisation

Standard [[dropout]] applied to recurrent connections harms LSTMs. [[dropout-for-rnns|Zaremba et al. (2014)]] showed dropout should only be applied to non-recurrent (inter-layer) connections.

## Uses in This Vault

- Sequence transduction and machine translation ([[encoder-decoder]], [[neural-machine-translation]])
- Controller for the [[neural-turing-machine]], where internal LSTM memory complements external addressable memory
- History encoding in [[neural-temporal-point-process|NTPP]] models, as one of the standard choices alongside GRU and transformer encoders
- Microtransition and sequence models across sports analytics

## See Also

- [[transformer]]
- [[gated-recurrent-unit]]
- [[dropout-for-rnns]]
- [[bidirectional-rnn]]
- [[encoder-decoder]]
- [[neural-temporal-point-process]]
