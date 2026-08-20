---
title: "Factor Graph"
type: concept
tags: [bayesian, statistics, probabilistic-graphical-model, inference]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Factor Graph

A factor graph is a bipartite graph consisting of variable nodes and factor nodes. The function represented by the graph (typically a joint distribution) is the product of all factor functions. The structure encodes conditional independence and enables efficient inference via message passing.

## Structure

- **Variable nodes** (circles): represent random variables.
- **Factor nodes** (squares): represent functions (potentials) that constrain or relate the connected variables.
- An edge connects a variable to a factor if and only if the variable appears as an argument of that factor.

## Message Passing

The sum-product algorithm on factor graphs computes exact single-variable marginals when the graph is acyclic. Messages propagate between variable and factor nodes:

- Variable-to-factor message: product of all incoming factor messages except the recipient.
- Factor-to-variable message: integral of the factor times incoming variable messages, marginalising out all variables except the recipient.

When exact messages are intractable, [[approximate-message-passing]] (e.g., [[expectation-propagation]]) can be used.

## In TrueSkill

The [[trueskill]] model is expressed as a factor graph with variables for skills, performances, team performances, and performance differences. Most messages are exact Gaussians; messages from comparison factors require approximation via moment matching.

## See Also

- [[approximate-message-passing]]
- [[expectation-propagation]]
- [[bayesian-inference]]
- [[trueskill]]
