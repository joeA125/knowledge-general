---
title: "Autoregressive Model"
type: concept
tags: [deep-learning, generative-model, autoregressive-model, sequence-modelling, point-process, density-estimation]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.9
provenance:
  extracted: 55%
  inferred: 38%
  ambiguous: 7%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-08-14
---

# Autoregressive Model

An autoregressive model factorises a joint distribution using the chain rule: $p(\mathbf{x}) = \prod_i p(x_i \mid x_{<i})$, modelling each element conditioned on all previous elements. This decomposition is assumption-free and, with a sufficiently powerful model (e.g., RNN, PixelCNN, [[transformer]]), can represent arbitrary distributions.

## Examples

- **Language:** RNN and Transformer language models predict the next token given all previous tokens.
- **Images:** PixelRNN/PixelCNN model each pixel conditioned on previous pixels in raster-scan order.
- **Audio:** WaveNet uses causal [[dilated-convolution]]s for raw audio generation.

## Factorising *Within* an Element, Not Just Across a Sequence

The chain rule applies to any set of variables, not only to sequence positions. Where a single element has several attributes — a time, a location, a type — the joint distribution over those attributes can itself be factorised:

$$f(a, b, c \mid H) = f_a(a \mid H)\; f_b(b \mid a, H)\; f_c(c \mid a, b, H)$$

Each component gets its own network, and each conditions on the components already predicted. **Autoregression across an element's attributes, nested inside autoregression across the sequence.**

The cost is that later components condition on earlier *predictions* rather than earlier truths, so error propagates within a single element and the last component in the chain absorbs it. Severing the within-element links removes that propagation and loses the dependence — which is usually the worse trade. See [[event-prediction]].

## Ordering Is Free in Theory, Not in Practice

Any ordering of the chain rule gives a valid factorisation of the same joint distribution — the decomposition is exact regardless. With unlimited capacity and data, ordering would be irrelevant.

It is not.

> ### `factorisation-order-is-an-unswept-parameter`
> **Where a model factorises a multi-component prediction, the ordering determines which conditionals are easy to learn. It is exact in theory and consequential in practice, and is almost always asserted rather than searched — with the differences concentrating in whichever component comes last.**
> ^[generated. rests-on: imported:autoregressive-ordering-effects]

The phenomenon recurs across autoregressive modelling: PixelCNN's raster-scan order over pixels is a choice, and [[sequence-to-sequence-sets|Order Matters]] makes the analogous point for sets, showing input and output ordering materially affect seq2seq performance even when the underlying object is unordered.

The practical reading: **put first the components that best predict the others.**

## Relation to VAEs

Used as a [[variational-autoencoder|VAE]] decoder, an autoregressive model can reconstruct the data without using the latent code at all — the information-preference problem. [[variational-lossy-autoencoders|VLAE]] turns that failure into a design lever by restricting the decoder's receptive field.

## Relation to Point Processes

A temporal [[point-process]] shares exactly this structure — $f(t_1, t_2, \dots) = \prod_i f(t_i \mid H_i)$ — but over continuous time rather than a fixed sequence index. **The extra work a point process does is modelling *when* the next element occurs, not merely what it is.**

## See Also

- [[generative-model]] · [[variational-autoencoder]] · [[transformer]] · [[lstm]] · [[gpt]] · [[tokenization]]
- [[point-process]] · [[neural-temporal-point-process]] · [[event-prediction]] · [[sequence-to-sequence-sets|Order Matters Summary]]
- [[model-selection]] · [[dilated-convolution]] · [[variational-lossy-autoencoders|VLAE Summary]]
