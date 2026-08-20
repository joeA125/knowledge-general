---
title: "Bayesian Inference"
type: concept
tags: [bayesian, statistics, inference]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.9
provenance:
  extracted: 40%
  inferred: 50%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-05-08
---

# Bayesian Inference

Bayesian inference is a framework for updating beliefs about unknown quantities (parameters, latent variables) in light of observed data, using [[bayes-theorem]]:

$$p(\theta | \text{data}) = \frac{p(\text{data} | \theta) \, p(\theta)}{p(\text{data})}$$

The posterior $p(\theta | \text{data})$ combines the likelihood of the data given the parameters with the prior belief $p(\theta)$.

## Key Properties

- **Uncertainty quantification:** The posterior is a full distribution, not a point estimate.
- **Sequential updating:** Today's posterior becomes tomorrow's prior (online learning / Gaussian density filtering).
- **Principled model comparison** via marginal likelihoods.

## In TrueSkill

[[trueskill]] applies Bayesian inference to skill rating: each player's skill has a Gaussian prior, and after each game the posterior is computed (approximately, via [[expectation-propagation]]) and used as the prior for the next game.

## See Also

- [[bayes-theorem]]
- [[factor-graph]]
- [[expectation-propagation]]
- [[trueskill]]
