---
title: "Neural Temporal Point Process"
type: concept
tags: [point-process, event-prediction, deep-learning, sequence-modelling, rnn, transformer, encoder-decoder-bottleneck]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 8%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Neural Temporal Point Process

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

A [[point-process]] whose conditional intensity — or inter-event distribution — is parameterised by a neural network rather than a fixed functional form. The move that replaces "assume the influence of past events decays exponentially" with "learn how it decays".

## The Three-Step Structure

^[imported: the standard NTPP formulation]

1. **Represent each event** as a vector: its inter-event time, its type, any other marks.
2. **Encode the history** $(\vec{y}_1, \dots, \vec{y}_{i-1})$ into a fixed-size $\vec{h}_i$.
3. **Predict the next event** from $\vec{h}_i$.

Step 2 is worth pausing on, because it deliberately reintroduces something the field spent years removing.

## The Bottleneck Comes Back On Purpose

Compressing a variable-length history into a fixed vector is exactly the [[encoder-decoder-bottleneck]] that attention was invented to eliminate.

Here it is **not a problem**, and the reason is structural: the bottleneck hurts in sequence-to-sequence settings where each output position needs different information from the input. An NTPP predicts **one** next event, so one context vector suffices in principle.

> ### `attention-can-improve-compression-not-only-replace-it`
> **Attention was introduced to remove a fixed-length bottleneck. In single-prediction settings it is instead used to compress well — weighting history usefully rather than avoiding compression. The same mechanism, serving the opposite architectural purpose.**
> ^[generated. rests-on: imported:ntpp-history-encoding]

## Encoder Choice Is Not Settled

| Encoder | Typical finding |
|---|---|
| [[lstm\|LSTM]] / [[gated-recurrent-unit\|GRU]] | Marginally better loss; far fewer parameters |
| [[transformer\|Transformer]] | Substantially faster to train; slightly worse loss |

The Transformer's advantage is parallelisation across long sequences. Event histories are typically **short** — tens of events, not thousands of tokens — so that advantage is worth much less here than in language, while the parameter cost is real.

**Recurrent encoders remain competitive in this setting**, which is a live exception to the general displacement of recurrence. See [[lstm]].

## Modelling Time Itself

Two routes, and the choice determines what the model can express:

- **Intensity-based** — parameterise $\lambda^*(t)$ and integrate for the likelihood. Retains the intensity interpretation; the integral is expensive and often needs Monte Carlo.
- **Density-based** — predict the inter-event time distribution directly. Cheap and tractable; loses the intensity semantics, and with it the ability to ask "what is the instantaneous rate right now".

**Neither is strictly better and the trade is rarely stated.** A model chosen for tractability may quietly lose the quantity the analysis needed.

## Where the Value Comes From

The honest framing: an NTPP's advantage over a discretised sequence model is not usually accuracy. It is that **the timing is modelled rather than binned**, so questions about rate, burstiness and expected waiting time are answerable at all.

Where those questions are not asked, binning and a standard sequence model is simpler and often sufficient. The formalism should follow the question.

## See Also

- [[point-process]] · [[event-prediction]] · [[encoder-decoder-bottleneck]] · [[autoregressive-model]] · [[transformer]] · [[lstm]] · [[gated-recurrent-unit]]
- [[attention-mechanism]] · [[model-selection]] · [[generative-model]] · [[probabilistic-classification]]
