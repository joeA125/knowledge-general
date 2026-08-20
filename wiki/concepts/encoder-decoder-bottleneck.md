---
title: "Encoder-Decoder Bottleneck"
type: concept
tags: [deep-learning, machine-translation, sequence-modelling, architecture, point-process]
sources: [raw/papers/neural-machine-translation.md, raw/papers/transformer-point-process-football-event-modelling.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-07-23
---

# Encoder-Decoder Bottleneck

The encoder-decoder bottleneck refers to the problem that arises in [[encoder-decoder]] architectures when the encoder must compress all information from a variable-length input sequence into a single fixed-length vector $c$.

## The Problem

As input sequences grow longer, the fixed-length vector cannot retain all necessary information. Cho et al. (2014b) demonstrated that basic encoder-decoder performance degrades rapidly as sentence length increases.

## Solution: Attention

[[neural-machine-translation|Bahdanau et al. (2014)]] solved this by introducing [[additive-attention]], replacing the single fixed context vector with a dynamic, position-specific context vector $c_i = \sum_j \alpha_{ij} h_j$ computed at each decoding step. This frees the encoder from having to compress everything into one vector.

The [[transformer]] later eliminated recurrence entirely, using [[multi-head-attention]] with no fixed-length bottleneck at all.

## Where the Bottleneck Returns by Design

The bottleneck is a problem in *sequence-to-sequence* settings, where each output position may need different information from the input. It is not intrinsically a problem when there is a **single** thing to predict.

The [[neural-temporal-point-process|NTPP]] framework makes this explicit: its second step is to "encode the history $(\vec{y}_1, \dots, \vec{y}_{i-1})$ into a fixed-size vector $\vec{h}_i$" — deliberately reintroducing exactly the compression Bahdanau removed. Since only one next event is being predicted, a single context vector suffices in principle.

[[nmstpp]] compresses 40 events into a 31-dimensional history vector this way, and uses a transformer encoder to do it — attention here serves to *weight the history well while compressing it*, rather than to avoid compressing at all.

The cost is that all three predicted components (time, zone, action) must share one history representation. Whether a per-component context vector would help is untested; the model instead recovers dependence through its [[autoregressive-model|chain-rule factorisation]], letting later components condition on earlier predictions rather than on separately attended history.

## How to Tell If the Bottleneck Is Binding

Yeung et al. offer a practical diagnostic: inspect the self-attention weights across the history window. If weights concentrate at one end, the window is likely mis-sized. In their case, weights over 40 events lay between 0.01 and 0.06 with no systematic trend, which they read as evidence the window is neither too long nor too short for the compression to handle.

## See Also

- [[encoder-decoder]]
- [[additive-attention]]
- [[attention-mechanism]]
- [[transformer]]
- [[neural-temporal-point-process]]
- [[nmstpp]]
