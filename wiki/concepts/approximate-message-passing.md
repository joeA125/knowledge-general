---
title: "Approximate Message Passing"
type: concept
tags: [bayesian, statistics, inference, probabilistic-graphical-model]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.85
provenance:
  extracted: 65%
  inferred: 30%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Approximate Message Passing

Approximate message passing refers to inference algorithms on [[factor-graph]]s where exact messages are intractable and are instead approximated — typically by constraining them to a parametric family (e.g., Gaussians) and matching moments.

## In TrueSkill

In the [[trueskill]] factor graph, most messages are exact Gaussians, but messages from comparison factors ($\mathbb{I}(\cdot > \epsilon)$ and $\mathbb{I}(|\cdot| \leq \epsilon)$) are non-Gaussian. Following [[expectation-propagation]], the marginals over performance differences are approximated via moment matching (minimising KL divergence), and approximate messages are recovered by dividing the approximate marginal by the incoming cavity message.

Messages on the shortest path between approximate marginals are iterated until convergence.

## See Also

- [[factor-graph]]
- [[expectation-propagation]]
- [[bayesian-inference]]
- [[trueskill]]
