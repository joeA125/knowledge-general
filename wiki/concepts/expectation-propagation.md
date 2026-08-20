---
title: "Expectation Propagation"
type: concept
tags: [bayesian, statistics, inference, approximation]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.85
provenance:
  extracted: 50%
  inferred: 40%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-05-08
---

# Expectation Propagation

Expectation Propagation (EP) is an approximate inference algorithm developed by [[tom-minka]] (2001). It iteratively refines approximate marginals by replacing intractable factors with members of an exponential family (typically Gaussians) via moment matching, which minimises the KL divergence $\text{KL}(p \| q)$.

## Key Idea

For each intractable factor:
1. Compute the "cavity distribution" by removing the current approximation of the factor from the overall approximate posterior.
2. Multiply the cavity by the true factor to get a hybrid distribution.
3. Project this hybrid back into the exponential family via moment matching.
4. Update the approximate factor as the ratio of the new approximation to the cavity.

## In TrueSkill

[[trueskill]] uses EP to approximate non-Gaussian messages from comparison factors in the [[factor-graph]]. The resulting Gaussian moment matching computes additive and multiplicative corrections ($V$ and $W$ functions) derived from truncated Gaussian integrals.

## See Also

- [[approximate-message-passing]]
- [[factor-graph]]
- [[bayesian-inference]]
- [[tom-minka]]
