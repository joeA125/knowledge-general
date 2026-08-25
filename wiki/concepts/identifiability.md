---
title: "Identifiability"
type: concept
tags: [identifiability, statistics, model-selection, inference, bayesian, mixture-model, interpretability]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 26%
  generated: 10%
  imported: 62%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Identifiability

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Whether a model's parameters are **uniquely determined** by the distribution they induce. A model is identifiable if two different parameter settings cannot produce the same distribution over observables.

Where it fails, the data cannot distinguish between parameter values — and **no amount of additional data helps**, because the information is not missing, it does not exist.

## Why It Is Not a Sample-Size Problem

The distinction that matters most, and the one most often confused.

| | Cause | Fixed by |
|---|---|---|
| **High variance** | Too few observations | **More data** |
| **Non-identifiability** | The model structure | **A constraint, or a different model** |

Both present as unstable estimates. Only one improves with data. **A non-identifiable parameter estimated on a million observations is exactly as undetermined as on a hundred** — the estimator will report a confident-looking value, chosen by the optimiser's path rather than by the evidence.

> ### `non-identifiability-produces-confident-arbitrary-answers`
> **An unidentified parameter still gets a value: the optimiser returns whatever point it reached. Nothing in the fit statistics distinguishes that from an estimate the data determined, which is why non-identifiability is usually discovered by re-running with a different initialisation rather than by any diagnostic.**
> ^[generated. rests-on: imported:identifiability-theory]

## Common Sources

^[imported]

| Source | Example |
|---|---|
| **Scale invariance** | Latent strengths defined only up to a common factor |
| **Label switching** | Mixture components are exchangeable — permuting them changes nothing observable |
| **Rotation invariance** | Factor models identified only up to an orthogonal transform |
| **Over-parameterisation** | More parameters than the likelihood constrains |
| **Collinearity** | Perfectly correlated predictors; only their sum is determined |

The [[bradley-terry-model|Bradley-Terry]] case is the cleanest illustration: latent strengths are determined only up to a common scale, so a constraint is required — fix one competitor at zero, or fix the sum. **This is why rating scales are arbitrary and cross-system comparisons are meaningless without a shared anchor.**

## Standard Repairs

- **Impose a constraint.** Fix a reference level, normalise, or order the components. Cheap, and the choice is a modelling decision that should be reported.
- **Add a prior.** In [[bayesian-inference|Bayesian]] treatment a proper prior yields a proper posterior even where the likelihood is flat — but **the posterior then reflects the prior in that direction**, which is not the same as having learned it.
- **Change the model.** Often the right answer where non-identifiability signals that the model claims more structure than the data can support.
- **Collect different data**, not more of the same — a design change rather than a sample-size change.

The second is worth caution. A Bayesian fit to a non-identified model produces clean posteriors and gives no obvious signal that some of what it reports came from the prior rather than the evidence.

## Weak Identifiability

The practically common case sits between the extremes: parameters are identifiable in principle but the likelihood is nearly flat along some direction, so estimates are extremely sensitive to small perturbations.

This is more dangerous than outright non-identifiability, because **the model is technically identified and every diagnostic passes.** Symptoms are strong correlations between parameter estimates, sensitivity to initialisation, and estimates that move substantially under small data changes. See [[model-selection]] and [[uncertainty-quantification]].

## Why It Matters for Interpretation

Where model parameters are read substantively — components as parts, factors as constructs, weights as importance — identifiability is the precondition for that reading to mean anything.

A decomposition that changes under reinitialisation describes **one solution, not the data**. See [[non-negative-matrix-factorization]], whose non-negativity constraint buys interpretable parts and gives up the uniqueness that [[eigenvector|spectral]] methods retain, and [[interpretability]] more broadly.

## See Also

- [[model-selection]] · [[uncertainty-quantification]] · [[bayesian-inference]] · [[interpretability]] · [[selection-bias]]
- [[bradley-terry-model]] · [[non-negative-matrix-factorization]] · [[eigenvector]] · [[structured-model-decomposition]] · [[probabilistic-classification]]
