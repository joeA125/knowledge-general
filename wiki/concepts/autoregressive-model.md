---
title: "Autoregressive Model"
type: concept
tags: [deep-learning, generative-model, autoregressive-model, sequence-modelling, point-process, density-estimation]
sources: [raw/papers/variational-lossy-autoencoders.md, raw/papers/transformer-point-process-football-event-modelling.md]
confidence: 0.9
provenance:
  extracted: 55%
  inferred: 38%
  ambiguous: 7%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-07-23
---

# Autoregressive Model

An autoregressive model factorises a joint distribution using the chain rule: $p(\mathbf{x}) = \prod_i p(x_i \mid x_{<i})$, modelling each element conditioned on all previous elements. This decomposition is assumption-free and, with a sufficiently powerful model (e.g., RNN, PixelCNN, [[transformer]]), can represent arbitrary distributions.

## Examples

- **Language:** RNN and Transformer language models predict the next token given all previous tokens.
- **Images:** PixelRNN/PixelCNN model each pixel conditioned on previous pixels in raster-scan order.
- **Audio:** WaveNet uses causal [[dilated-convolution]]s for raw audio generation.

## Factorising *Within* an Element, Not Just Across a Sequence

The chain rule applies to any set of variables, not only to sequence positions. [[nmstpp|NMSTPP]] uses it to decompose the *components of a single event* — its time, location, and type:

$$f(t_i, z_i, m_i \mid H_i) = f_t(t_i \mid H_i) \; f_z(z_i \mid t_i, H_i) \; f_m(m_i \mid t_i, z_i, H_i)$$

Each component is predicted by its own network, but each conditions on the components already predicted. This is autoregression across an event's *attributes*, nested inside autoregression across the event *sequence*.

The payoff is measurable: severing the within-event links so each network sees only $H_i$ raises total loss from 4.40 to 4.44, with the entire degradation falling on the last component in the chain.

## Ordering Is Free in Theory, Not in Practice

Any ordering of the chain rule gives a valid factorisation of the same joint distribution — the decomposition is exact regardless. With unlimited capacity and data, ordering would be irrelevant.

It is not. [[transformer-point-process-football-event-modelling|Yeung et al. (2023)]] grid-search all six orderings of $(t, z, m)$ and find a 0.18 spread in total loss, with $(t, z, m)$ best and $(z, t, m)$ worst — the difference concentrated almost entirely in the final component's loss.

The same phenomenon appears elsewhere in autoregressive modelling: PixelCNN's raster-scan order over pixels is a choice, and the [[read-process-write|Order Matters]] paper makes the analogous point for sets, showing that input and output ordering materially affect seq2seq performance even when the underlying object is unordered.

The practical reading is that ordering determines *which conditionals are easy* — put the components that best predict the others first.

## Relation to VAEs

When used as a VAE decoder, autoregressive models can model data without using the latent code, causing the "information preference" problem addressed by the [[variational-lossy-autoencoder]].

## Relation to Point Processes

A temporal [[point-process]] shares exactly this structure — $f(t_1, t_2, \dots) = \prod_i f(t_i \mid H_i)$ — but over continuous time rather than a fixed sequence index. The extra work a point process does is modelling *when* the next element occurs, not merely what it is.

## See Also

- [[variational-lossy-autoencoder]]
- [[nmstpp]]
- [[point-process]]
- [[read-process-write]]
- [[transformer]]
- [[lstm]]
