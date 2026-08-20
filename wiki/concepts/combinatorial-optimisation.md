---
title: "Combinatorial Optimisation"
type: concept
tags: [combinatorial-optimisation, deep-learning, machine-learning]
sources: [raw/papers/pointer-networks.md]
confidence: 0.85
provenance:
  extracted: 55%
  inferred: 40%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Combinatorial Optimisation

Combinatorial optimisation is the problem of finding an optimal object from a finite but typically enormous set of discrete candidates — for example the shortest tour through a set of cities, or the convex hull of a point set. Many such problems (like the Travelling Salesman Problem) are NP-hard, so exact solutions are intractable at scale and much research targets good approximate solutions.

## Neural Approaches

A recurring theme in modern machine learning is *learning* to solve combinatorial problems from examples rather than hand-designing algorithms. The challenge for neural networks is that the output is a discrete structure whose size varies with the input, which standard fixed-vocabulary sequence models cannot produce.

The [[pointer-network|Pointer Network]] ([[oriol-vinyals|Vinyals]] et al., 2015) addressed this directly: by using [[additive-attention]] as a pointer into the input sequence, it produces outputs drawn from the input itself, with a variable-size output dictionary. The original paper demonstrated it on three geometric problems:

- **Planar convex hull** — exact $O(n \log n)$ classical solution; learned approximately.
- **Delaunay triangulation** — $O(n \log n)$ exact; learned approximately.
- **Planar Travelling Salesman Problem** — NP-hard; the network learns competitive approximate tours purely from training examples and generalises beyond training lengths.

This line of work seeded the broader field of *neural combinatorial optimisation*, which combines learned models with search or reinforcement learning to tackle routing, scheduling, and assignment problems.^[inferred: broader-field framing beyond the ingested source]

## See Also

- [[pointer-network]]
- [[additive-attention]]
- [[pointer-networks|Source Summary]]
