---
title: "Pointer Networks — Source Summary"
type: summary
tags: [deep-learning, attention, sequence-modelling, rnn, combinatorial-optimisation, pointer-mechanism]
sources: [raw/papers/pointer-networks.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Pointer Networks

**Authors:** [[oriol-vinyals]], [[meire-fortunato]], [[navdeep-jaitly]]
**Affiliations:** [[google-brain]], UC Berkeley (Department of Mathematics)
**Published:** 2015 (NeurIPS)

## Key Contribution

This paper introduces the [[pointer-network]] (Ptr-Net), a neural architecture that uses [[additive-attention]] as a pointer mechanism to select elements from the input sequence as outputs. This solves the fundamental problem that standard sequence-to-sequence models have fixed output dictionaries, whereas many combinatorial problems require output dictionaries whose size depends on the input length.

## The Problem

Standard [[encoder-decoder]] (sequence-to-sequence) models and even attention-augmented models (e.g., [[neural-machine-translation|Bahdanau et al., 2014]]) use a softmax over a fixed-size output vocabulary. This makes them inapplicable to problems where the output is a variable-length selection of input elements — such as convex hull computation, Delaunay triangulation, and the Travelling Salesman Problem (TSP).

## Architecture

The Ptr-Net modifies the content-based [[additive-attention]] mechanism:

1. **Encoder:** An LSTM reads the input sequence $\mathcal{P} = \{P_1, \dots, P_n\}$ producing hidden states $(e_1, \dots, e_n)$.
2. **Decoder:** An LSTM produces hidden states $(d_1, \dots, d_{m(\mathcal{P})})$.
3. **Pointer mechanism:** Instead of blending encoder states into a context vector, the attention scores are used directly as the output distribution:

$$u_j^i = v^T \tanh(W_1 e_j + W_2 d_i)$$
$$p(C_i | C_1, \dots, C_{i-1}, \mathcal{P}) = \text{softmax}(u^i)$$

The softmax has dictionary size equal to the input length $n$, varying per instance.

## Key Difference from Standard Attention

In [[additive-attention]], attention weights are used to blend encoder states into a context vector that augments the decoder. In a Ptr-Net, attention weights are the output — they "point" to input positions rather than blending hidden states. This makes the output dictionary inherently variable-sized.

## Experimental Results

### Convex Hull
- Ptr-Net achieves 72.6% exact accuracy at $n=50$ (vs 38.9% attention, 1.9% vanilla LSTM).
- Area coverage consistently ~99.9%.
- A single model trained on $n=5$–$50$ generalises to $n=500$ with 99.2% area coverage.

### Delaunay Triangulation
- 80.7% accuracy at $n=5$; 52.8% triangle coverage at $n=50$.

### Travelling Salesman Problem
- Near-optimal tours for $n \leq 20$ (trained on exact solutions).
- Competitive with heuristic algorithms at $n=50$.
- When trained on the worst algorithm (A1), the Ptr-Net outperforms it — learning to improve upon its training signal.
- Generalisation: trained on $n=5$–$20$, good results up to $n=30$, degrades at $n \geq 40$.

## Architecture Details

- Single-layer LSTM, 256 or 512 hidden units
- SGD with learning rate 1.0, batch size 128
- L2 gradient clipping at 2.0
- 1M training examples
- Beam search at inference (with validity constraints for TSP)
- Computational complexity: $O(n^2)$ per sequence

## Impact

Pointer Networks opened a new class of problems to neural network approaches by showing that attention can serve as an output mechanism, not just an internal routing mechanism. The architecture influenced later work on neural combinatorial optimisation, copy mechanisms in NMT, and abstractive summarisation.

## See Also

- [[pointer-network]]
- [[additive-attention]]
- [[attention-mechanism]]
- [[encoder-decoder]]
