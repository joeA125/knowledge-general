---
title: "Gated Recurrent Unit"
type: concept
tags: [deep-learning, rnn, sequence-modelling, architecture, reinforcement-learning, temporal-difference, sports-analytics]
sources: [raw/papers/neural-machine-translation.md, raw/papers/action_valuation_football_agentic_reinforcement_learning.md]
confidence: 0.9
provenance:
  extracted: 68%
  inferred: 24%
  generated: 6%
  imported: 0%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-08-07
---

# Gated Recurrent Unit

The Gated Recurrent Unit (GRU; Cho et al., 2014a) is a gated recurrent neural network unit, similar to LSTM but with a simpler structure. It was used in both the encoder and decoder of the [[neural-machine-translation|Bahdanau attention model]].

## Mechanism

The hidden state update is:

$$s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$$

where:
- **Update gate** $z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i)$ controls how much of the previous state to retain.
- **Reset gate** $r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)$ controls how much of the previous state feeds into the candidate.
- **Candidate** $\tilde{s}_i = \tanh(W e(y_{i-1}) + U [r_i \circ s_{i-1}] + C c_i)$.

## Relation to LSTM

GRUs combine the LSTM's forget and input gates into a single update gate, and merge the cell state and hidden state. This gives fewer parameters while maintaining the ability to learn long-term dependencies through multiplicative gating.

## Why It Persists in Small-Data Settings

> **Added 2026-08-07.** This page had covered only the machine-translation origin. The GRU's more interesting recent role in this vault is elsewhere.

[[action-valuation-multi-agent-reinforcement-learning|Nakahara et al. (2023)]] use a **single 64-unit GRU layer with ReLU** as the value network for ten [[multi-agent-reinforcement-learning|per-player RL agents]], taking a 92-dimensional state plus a 14-dimensional one-hot action per timestep and emitting 14 Q-values.

That is a very small network in 2023, and the choice is instructive rather than dated. The training set is **1,669 possession sequences**. Against that, the [[transformer]] would be badly over-parameterised — the fewer-parameters property above stops being a footnote and becomes the reason for the choice. Compare [[nmstpp]], which does use a Transformer on football events, on a corpus of a different order.

The vault's general form of this observation is on [[theory-based-modelling]] and [[handcrafted-features-rule]]: **at football data scale, architecture capacity is usually not the binding constraint.** Nakahara et al. also apply $L_1$ [[regularization|regularisation]] to weights and biases explicitly to prevent overfitting "on the relatively small demonstration dataset".

### The recurrence is doing memory work, not just sequence work

Worth separating, because it is easy to miss. In a [[temporal-difference-learning|TD]] setting the GRU's hidden state serves a second purpose beyond modelling the sequence: it carries information across the bootstrap, so $Q(s_{t+1}, a_{t+1})$ is conditioned on the whole possession so far rather than on the instantaneous state alone.

That partly compensates for a state vector containing only current positions and velocities — no history, no possession phase, no scoreline. It also means the framework's effective credit horizon is set by **how far the GRU actually propagates signal**, which nobody reports. Since the same paper sets its discount factor to 1, recurrence depth is the *only* thing performing temporal credit assignment there. See [[temporal-discounting]].

Standard TD stabilisers — target networks, replay buffers — are absent; the recurrence over whole possessions is what stands in for them. The companion paper [[adaptive-action-supervision-multi-agent-rl|Fujii et al.]] uses the full stabiliser stack and **no recurrence at all**, which is the clearest illustration available that the two are substitutes for the same problem. See [[deep-q-network]].

## See Also

- [[bidirectional-rnn]] · [[lstm]] · [[recurrence]] · [[encoder-decoder]] · [[additive-attention]] · [[transformer]]
- [[temporal-difference-learning]] · [[deep-q-network]] · [[multi-agent-reinforcement-learning]] · [[reinforcement-learning]] · [[regularization]]
- [[trajectory-prediction]] · [[nmstpp]] · [[theory-based-modelling]] · [[representation-learning]] · [[temporal-discounting]]
- [[neural-machine-translation|Bahdanau et al. Summary]] · [[action-valuation-multi-agent-reinforcement-learning|Nakahara et al. Summary]]
