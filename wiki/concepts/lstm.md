---
title: "Long Short-Term Memory"
type: concept
tags: [deep-learning, rnn, lstm, architecture, sequence-modelling]
sources: [raw/papers/rnn-regularisation.md, raw/papers/neural-machine-translation.md]
confidence: 0.9
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-08-14
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

The [[transformer]] displaced recurrent encoders for most long-sequence tasks, but the trade-off is narrower than the displacement suggests. The recurring empirical finding is that **LSTMs remain competitive or marginally better on accuracy while being substantially slower to train**, because their gradient computation is inherently sequential where self-attention parallelises.

Typical measured differences, holding everything else fixed, put the LSTM slightly ahead on loss and the Transformer roughly **2–3× faster to train**, often at a fraction of the parameter count.^[imported: reported across the sequence-modelling and NTPP literature; not established by any held source]

> ### `recurrence-persists-where-sequences-are-short`
> **The Transformer's advantage is parallelisation across long sequences. Where sequences are short, training budget is tight relative to inference budget, or parameter count is constrained by data scale, recurrent encoders remain competitive or better.**
> ^[generated. rests-on: imported:lstm-transformer-comparisons]

The condition matters: a window of tens of steps is not a document of thousands of tokens, and the parallelisation advantage scales with the gap. See [[neural-temporal-point-process]], where short event histories make this the live question rather than a settled one.

## Relation to GRU

The [[gated-recurrent-unit]] simplifies the LSTM by merging the forget and input gates into a single update gate and combining the cell and hidden states, yielding roughly 25% fewer parameters. At small data scale that difference decides the choice; at large scale it rarely does. See [[gated-recurrent-unit]].

## Regularisation

Standard [[dropout]] applied to recurrent connections harms LSTMs — the noise compounds across timesteps and destroys the long-term memory the cell state exists to preserve. [[dropout-for-rnns|Zaremba et al.]] showed it should be applied to non-recurrent connections only.

## Where It Is Used

- Sequence transduction and machine translation — [[encoder-decoder]], [[neural-machine-translation]]
- Controller for the [[neural-turing-machine]], where internal memory complements external addressable memory
- History encoding in [[neural-temporal-point-process|NTPP]] models, alongside GRU and Transformer encoders
- Value networks in [[temporal-difference-learning|bootstrapped RL]], where the hidden state carries information across the bootstrap

## See Also

- [[transformer]] · [[gated-recurrent-unit]] · [[recurrence]] · [[bidirectional-rnn]] · [[encoder-decoder]] · [[encoder-decoder-bottleneck]]
- [[dropout-for-rnns]] · [[dropout]] · [[regularization]] · [[attention-mechanism]]
- [[neural-temporal-point-process]] · [[event-prediction]] · [[temporal-difference-learning]] · [[model-selection]]
- [[rnn-regularisation|Zaremba et al. Summary]] · [[neural-machine-translation|Bahdanau Summary]]
