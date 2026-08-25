---
title: "Gated Recurrent Unit"
type: concept
tags: [deep-learning, rnn, sequence-modelling, architecture, machine-translation, encoder-decoder-bottleneck, regularization]
sources: [raw/papers/neural-machine-translation.md]
confidence: 0.85
provenance:
  extracted: 55%
  inferred: 26%
  generated: 7%
  imported: 10%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-08-14
---

# Gated Recurrent Unit

A gated recurrent unit (Cho et al., 2014), simpler than the [[lstm|LSTM]] and comparable in capability. Introduced alongside — and used in both encoder and decoder of — the [[neural-machine-translation|Bahdanau attention model]].

## Mechanism

$$s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$$

| Gate | Controls |
|---|---|
| **Update** $z_i$ | How much of the previous state to retain |
| **Reset** $r_i$ | How much of the previous state feeds the candidate |
| **Candidate** $\tilde{s}_i$ | The proposed new content, $\tanh(W e + U[r_i \circ s_{i-1}] + C c_i)$ |

The update gate does the work that the LSTM splits between forget and input gates. Because $z_i$ appears as both $(1-z_i)$ and $z_i$, retention and replacement are **forced to trade off** rather than being independently controllable — which is the substantive simplification, not merely a parameter saving.

## Relation to LSTM

| | LSTM | GRU |
|---|---|---|
| Gates | Forget, input, output | Update, reset |
| State | Cell **and** hidden, separate | **Merged** |
| Parameters | More | **~25% fewer** |
| Typical performance | Comparable | Comparable |

No consistent winner across tasks. **The GRU is preferred where parameters are the binding constraint and the LSTM where the extra control helps**, and which applies is usually settled empirically rather than argued.

## Why Fewer Parameters Is Sometimes the Whole Argument

The parameter difference is a footnote at scale and decisive at small scale.

Where training data is limited — thousands of sequences rather than millions — a model's capacity is bounded by what will not overfit rather than by what the task needs. In that regime a [[transformer]] is badly over-parameterised, an LSTM is marginal, and a small GRU is the right size.

> ### `architecture-choice-tracks-data-scale-more-than-task-difficulty`
> **At small data scale the binding constraint is overfitting rather than expressiveness, so the architecture with fewest parameters that can represent the task often wins outright. The same task at larger scale reverses the ordering.**
> ^[generated. rests-on: imported:small-data-architecture-practice]

See [[regularization]] and [[theory-based-modelling]] for the two other responses to the same constraint — penalising weights, and encoding structure rather than learning it.

## Recurrence as Memory, Not Only Sequence Modelling

Worth separating because it is easy to miss. A recurrent hidden state serves two distinct purposes:

1. **Modelling the sequence** — the obvious one.
2. **Carrying state across a computation** that would otherwise see only the current input.

The second matters in [[temporal-difference-learning|bootstrapped value learning]], where the successor estimate can be conditioned on accumulated history rather than on the instantaneous state alone. **That partly substitutes for the [[deep-q-network|DQN]] stabiliser stack** — a recurrent value network and a target network address overlapping problems by different means.

Where recurrence performs this role, the effective horizon is set by **how far the hidden state actually propagates signal**, which is almost never measured or reported.

## See Also

- [[lstm]] · [[recurrence]] · [[bidirectional-rnn]] · [[encoder-decoder]] · [[encoder-decoder-bottleneck]] · [[additive-attention]] · [[transformer]]
- [[temporal-difference-learning]] · [[deep-q-network]] · [[reinforcement-learning]] · [[regularization]] · [[model-selection]]
- [[neural-temporal-point-process]] · [[representation-learning]] · [[theory-based-modelling]]
- [[neural-machine-translation|Bahdanau et al. Summary]]
