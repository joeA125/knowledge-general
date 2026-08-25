---
title: "Gaussian Process"
type: concept
tags: [gaussian-process, bayesian, statistics, inference, uncertainty-quantification, approximation, hierarchical-model]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 26%
  generated: 8%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Gaussian Process

> ⚠️ **No held source.** Background knowledge, marked `imported:`. Rasmussen & Williams (2006) is the standard reference.

A distribution over **functions**, such that any finite collection of function values has a joint Gaussian distribution. Specified by a mean function and a covariance (kernel) function:

$$f(x) \sim \mathcal{GP}\big(m(x),\, k(x, x')\big)$$

Conditioning on observed data gives a posterior that is again a GP, available in closed form — which is the property that makes them usable at all.

## What the Kernel Does

**The kernel is the model.** It encodes what "similar inputs give similar outputs" means, and every assumption about the function lives there:

| Kernel | Assumes |
|---|---|
| Squared exponential | Very smooth, infinitely differentiable |
| Matérn | Smooth to a chosen degree — usually more realistic |
| Periodic | Repeating structure |
| Linear | The function is linear |

Kernels compose: sums model additive structure, products model interactions. **Choosing a kernel is model specification, not a hyperparameter choice**, and the smoothness assumption in particular is consequential and easy to make by default — the squared-exponential kernel is a common choice and assumes more smoothness than most real functions have.

## Why Use One

Two properties, and the second is the point.

**Nonparametric flexibility.** The function is not restricted to a parametric family; complexity grows with data.

**Calibrated uncertainty that widens away from the data.** A GP's posterior variance grows in regions where nothing was observed, and does so automatically rather than by construction.

> ### `a-gp-knows-where-it-has-not-looked`
> **The posterior variance of a Gaussian process increases with distance from observed data, so extrapolation is flagged rather than silent. Most flexible models extrapolate confidently and wrongly; a GP extrapolates toward its prior mean with widening uncertainty, which is the honest failure mode.**
> ^[generated. rests-on: imported:gp-posterior-properties]

That property is why GPs remain the default for Bayesian optimisation and active learning, where knowing what you do not know is the whole task. See [[uncertainty-quantification]].

## The Cost

**Cubic in the number of observations.** Exact inference requires inverting an $n \times n$ covariance matrix, so $O(n^3)$ time and $O(n^2)$ memory — which caps exact GPs at a few thousand points.

The standard approximations trade exactness for scale:

- **Inducing points / sparse GPs** — summarise the data with $m \ll n$ pseudo-observations
- **Markov random field approximations** — impose conditional independence to get a sparse precision matrix, which is what makes spatial GP models tractable on large grids
- **Random feature expansions** — approximate the kernel by an explicit finite feature map

The second is the route taken by most large spatial applications, and it works because **a GP with a suitable kernel is equivalent to a sparse Gaussian Markov random field**, converting a dense inverse into a sparse one.^[imported]

## Where They Are Used

Spatial and spatio-temporal modelling, Bayesian optimisation, surrogate modelling of expensive simulations, and geostatistics — where the method originated as kriging.

The recurring condition is **few observations, expensive to obtain, and a real need for uncertainty**. Where data is plentiful and uncertainty is not required, a GP is usually the wrong tool.

## See Also

- [[bayesian-inference]] · [[uncertainty-quantification]] · [[bayes-theorem]] · [[expectation-propagation]] · [[message-passing]]
- [[model-selection]] · [[probability-calibration]] · [[identifiability]] · [[theory-based-modelling]] · [[selection-bias]]
