---
title: "Bayes' Theorem"
type: concept
tags: [bayesian, statistics]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.95
provenance:
  extracted: 30%
  inferred: 60%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-05-08
---

# Bayes' Theorem

Bayes' theorem relates the conditional and marginal probabilities of two events:

$$P(A|B) = \frac{P(B|A) \, P(A)}{P(B)}$$

In the context of [[bayesian-inference]], it provides the rule for updating a prior belief $p(\theta)$ with observed data to obtain a posterior $p(\theta | \text{data})$.

## In TrueSkill

[[trueskill]] uses Bayes' theorem (Equation 2 of the paper) to derive the posterior distribution over player skills given the game outcome and team assignments:

$$p(\mathbf{s}|\mathbf{r}, A) = \frac{P(\mathbf{r}|\mathbf{s}, A) \, p(\mathbf{s})}{P(\mathbf{r}|A)}$$

## See Also

- [[bayesian-inference]]
- [[trueskill]]
