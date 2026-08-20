---
title: "Gaussian Density Filtering"
type: concept
tags: [bayesian, inference, approximation, statistics]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.85
provenance:
  extracted: 55%
  inferred: 40%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Gaussian Density Filtering

Gaussian density filtering (also called assumed-density filtering, ADF) is an online approximate-inference technique in which beliefs are constrained to remain Gaussian as observations are processed one at a time. After each update the true posterior — which may not be Gaussian — is approximated by the Gaussian that best matches it (by moment matching), and that Gaussian becomes the prior for the next observation.

## Relation to Expectation Propagation

Gaussian density filtering can be seen as a single forward pass of [[expectation-propagation]] (EP): ADF incorporates each factor once and never revisits it, whereas EP iterates, repeatedly refining each factor's approximation until convergence. ADF is therefore cheaper but less accurate, and is order-dependent.^[inferred: standard ADF/EP relationship; the source applies it rather than deriving it]

## Use in TrueSkill

[[trueskill|TrueSkill]] ([[microsoft-research]], 2006) uses Gaussian density filtering to carry skill beliefs across games: each player's skill is a Gaussian $\mathcal{N}(\mu, \sigma^2)$, and the posterior computed after one game — via [[approximate-message-passing]] on a [[factor-graph]] — becomes the prior for the next game. This online, incremental update is what lets TrueSkill converge to a player's true skill in only a handful of matches while continuously tracking uncertainty.

## See Also

- [[trueskill]]
- [[expectation-propagation]]
- [[approximate-message-passing]]
- [[factor-graph]]
- [[bayesian-inference]]
- [[bayesian-true-skill-rating|Source Summary]]
