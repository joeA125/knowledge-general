---
title: "Encoder-Decoder Bottleneck"
type: concept
tags: [deep-learning, machine-translation, sequence-modelling, architecture, point-process]
sources: [raw/papers/neural-machine-translation.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-08-14
---

# Encoder-Decoder Bottleneck

The encoder-decoder bottleneck refers to the problem that arises in [[encoder-decoder]] architectures when the encoder must compress all information from a variable-length input sequence into a single fixed-length vector $c$.

## The Problem

As input sequences grow longer, the fixed-length vector cannot retain all necessary information. Cho et al. (2014b) demonstrated that basic encoder-decoder performance degrades rapidly as sentence length increases.

## Solution: Attention

[[neural-machine-translation|Bahdanau et al. (2014)]] solved this by introducing [[additive-attention]], replacing the single fixed context vector with a dynamic, position-specific context vector $c_i = \sum_j \alpha_{ij} h_j$ computed at each decoding step. This frees the encoder from having to compress everything into one vector.

The [[transformer]] later eliminated recurrence entirely, using [[multi-head-attention]] with no fixed-length bottleneck at all.

## Where the Bottleneck Returns by Design

The bottleneck is a problem in *sequence-to-sequence* settings, where each output position may need different information from the input. It is **not** intrinsically a problem when there is a single thing to predict.

The [[neural-temporal-point-process|NTPP]] framework makes this explicit: its second step is to encode the history $(\vec{y}_1, \dots, \vec{y}_{i-1})$ into a fixed-size $\vec{h}_i$ — deliberately reintroducing exactly the compression Bahdanau removed. Since only one next event is predicted, one context vector suffices in principle.

Attention is still used, but for a different job: **weighting the history well while compressing it**, rather than avoiding compression.

> ### `attention-can-improve-compression-not-only-replace-it`
> **Attention was introduced to remove a fixed-length bottleneck. In single-prediction settings it is instead used to make that bottleneck lossy in the right places — the same mechanism serving the opposite architectural purpose.**
> ^[generated. rests-on: source:bahdanau-context-vector, imported:ntpp-history-encoding]

The cost is that every predicted component must share one history representation. Whether a per-component context vector would help is generally untested; models instead recover dependence through [[autoregressive-model|chain-rule factorisation]], letting later components condition on earlier *predictions* rather than on separately attended history. That is a cheaper fix and a weaker one.

## Diagnosing Whether It Binds

A practical check: inspect the self-attention weights across the history window. **Concentration at either end suggests the window is mis-sized** — at the recent end, more history would help; at the distant end, the window is longer than the model can use.

A flat distribution indicates the compression is handling the window it has been given. See `attention-weights-diagnose-window-length` on [[attention-mechanism]].

## See Also

- [[encoder-decoder]] · [[additive-attention]] · [[attention-mechanism]] · [[transformer]] · [[multi-head-attention]]
- [[neural-temporal-point-process]] · [[point-process]] · [[event-prediction]] · [[autoregressive-model]] · [[lstm]] · [[gated-recurrent-unit]]
- [[model-selection]] · [[neural-machine-translation|Bahdanau Summary]]
