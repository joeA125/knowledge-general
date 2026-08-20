---
title: "Recurrence"
type: concept
tags: [deep-learning, rnn, sequence-modelling]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.8
provenance:
  extracted: 40%
  inferred: 55%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Recurrence

Recurrence is the processing of a sequence by maintaining a hidden state that is updated one element at a time, so that the computation at step $t$ depends on the state produced at step $t-1$. It is the defining mechanism of recurrent neural networks such as the [[lstm]] and [[gated-recurrent-unit]].

## Mechanism

A recurrent layer computes a hidden state $h_t$ from the current input $x_t$ and the previous state:

$$h_t = f(h_{t-1}, x_t)$$

This sequential dependency lets the network carry information along a sequence of arbitrary length, but it also forces computation to proceed step by step.

## Why the Transformer Removes It

The [[transformer]] dispenses with recurrence entirely, relying on [[attention-mechanism|attention]] instead. The [[attention-is-all-you-need|Attention Is All You Need]] paper motivates this on three axes:^[inferred: framing summarised from source's self-attention comparison]

- **Parallelisation:** recurrence requires $O(n)$ sequential operations, preventing parallelism within a training example; self-attention needs only $O(1)$.
- **Path length:** in a recurrent model, information between distant positions must traverse $O(n)$ steps, making long-range dependencies hard to learn; self-attention connects any two positions directly.
- **Speed:** removing the sequential bottleneck was central to the Transformer's dramatically faster training.

Recurrence remains valuable where strict streaming or unbounded sequence length matters, and gated variants ([[lstm]], [[gated-recurrent-unit]]) were the dominant sequence models before attention-based architectures.

## See Also

- [[lstm]]
- [[gated-recurrent-unit]]
- [[bidirectional-rnn]]
- [[transformer]]
- [[convolution]]
