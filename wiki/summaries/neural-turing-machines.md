---
title: "Neural Turing Machines — Source Summary"
type: summary
tags: [deep-learning, architecture, attention, external-memory, neural-computation, lstm]
sources: [raw/papers/neural-turing-machines.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Neural Turing Machines

**Authors:** [[alex-graves]], [[greg-wayne]], [[ivo-danihelka]]
**Affiliation:** [[google-deepmind]]
**Published:** 2014; arXiv:1410.5401

## Key Contribution

This paper introduces the [[neural-turing-machine]] (NTM), a neural network architecture augmented with an external memory bank accessed via differentiable attention. The system is analogous to a Turing Machine but is fully differentiable and trainable end-to-end with gradient descent. NTMs can learn simple algorithms (copying, sorting, associative recall) from input-output examples and generalise beyond the training data.

## Architecture

The NTM has two components:

1. **Controller:** A neural network (feedforward or [[lstm]]) that interacts with the external world via input/output vectors and with memory via read and write heads.
2. **Memory:** An $N \times M$ matrix ($N$ locations, each an $M$-dimensional vector). All interactions are "blurry" (soft) — heads attend to memory with normalised weightings rather than hard addresses, making everything differentiable.

### Reading
The read vector is a convex combination of memory rows weighted by the attention weighting:
$$\mathbf{r}_t = \sum_i w_t(i) \mathbf{M}_t(i)$$

### Writing
Inspired by LSTM gates, writing decomposes into erase and add:
$$\tilde{\mathbf{M}}_t(i) = \mathbf{M}_{t-1}(i)[\mathbf{1} - w_t(i)\mathbf{e}_t]$$
$$\mathbf{M}_t(i) = \tilde{\mathbf{M}}_t(i) + w_t(i)\mathbf{a}_t$$

### Addressing
A hybrid addressing system combining:
1. **Content-based addressing:** Cosine similarity between a key vector $\mathbf{k}_t$ and memory rows, sharpened by key strength $\beta_t$.
2. **Location-based addressing:** Interpolation gate $g_t$ blends content weighting with previous weighting; shift weighting $\mathbf{s}_t$ applies circular convolution for relative position changes; sharpening factor $\gamma_t$ combats dispersion.

These mechanisms support three modes: pure content lookup, content lookup followed by shift (for accessing data adjacent to a match), and iterative shifting (for sequential traversal).

## Experimental Results

### Copy
NTM learns much faster than LSTM and generalises to sequences far longer than seen during training (up to 120, trained on ≤20). LSTM fails to generalise beyond training lengths. NTM learns a human-like algorithm: write inputs sequentially, return to start, read sequentially.

### Repeat Copy
NTM generalises to longer sequences and higher repeat counts. It extends the copy algorithm with an outer loop. However, it cannot generalise the counting mechanism beyond the training range.

### Associative Recall
Given a sequence of items, query with one item and return the next. NTM with feedforward controller learns within ~30K episodes (LSTM >1M). NTM generalises to 12 items (trained on ≤6). The network learns content-based lookup combined with location-based offset.

### Dynamic N-Grams
NTM approaches the Bayesian optimal estimator for predicting the next bit in sequences drawn from random 6-gram distributions, using memory as a re-writable table to track transition statistics.

### Priority Sort
NTM learns to sort by writing to memory locations proportional to priority values (a form of address-based sorting), then reading sequentially.

## Controller Choice

- **Feedforward controller:** More interpretable; memory use patterns are clearer. Limited by number of read/write heads (each head enables one more operand per step).
- **LSTM controller:** Internal memory complements external memory (analogous to CPU registers vs RAM). Can store read vectors across time steps, removing the head-count bottleneck.

## Impact

NTMs demonstrated that neural networks could learn algorithmic behaviour when given appropriate external memory, inspiring subsequent work including Memory Networks (Weston et al., 2015), the [[read-process-write]] architecture, and Differentiable Neural Computers (Graves et al., 2016).

## See Also

- [[neural-turing-machine]]
- [[attention-mechanism]]
- [[lstm]]
- [[read-process-write]]
