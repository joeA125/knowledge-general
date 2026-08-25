---
title: "Gradient Boosting"
type: concept
tags: [gradient-boosting, machine-learning, random-forest, regression, model-selection, interpretability, feature-attribution]
sources: []
confidence: 0.65
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

# Gradient Boosting

> ⚠️ **No held source.** Background knowledge, marked `imported:`. Friedman (2001) is the primary source; XGBoost, LightGBM and CatBoost are the standard implementations.

An ensemble method building an additive model of weak learners — almost always shallow decision trees — where **each new tree fits the gradient of the loss with respect to the current predictions.**

$$F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$$

with $h_m$ fitted to the residual gradient and $\eta$ a learning rate shrinking each contribution.

## Boosting Against Bagging

The contrast with **random forests** is the clearest way to understand it:

| | Random forest | Gradient boosting |
|---|---|---|
| Trees are | **Independent**, trained in parallel | **Sequential**, each correcting the last |
| Reduces | **Variance** | **Bias** |
| Individual trees | Deep, low bias, high variance | **Shallow**, high bias, low variance |
| Overfits with more trees | **No** — averaging is stable | **Yes** — each tree fits residual noise more closely |

The last row is the practically important one. **Adding trees to a random forest is safe; adding trees to a boosted model is not**, which is why boosting needs early stopping on a validation set and forests do not.

## Why It Dominates Tabular Data

Three reasons, and they are structural rather than incidental:

- **Trees handle heterogeneous features natively.** No scaling, no encoding of monotone transformations, mixed types accepted.
- **Axis-aligned splits suit tabular structure.** Where meaningful features are individually informative, splitting on them one at a time is efficient. Neural networks must learn that structure from data.
- **Interactions are captured to tree depth.** A depth-6 tree expresses up to sixth-order interactions with no specification.

**Neural networks are not obviously better on tabular problems and are frequently worse**, which remains true despite considerable effort to change it. The advantage reverses where the input has strong spatial or sequential structure that weight sharing can exploit — see [[fully-convolutional-network]].

## The Regularisation Burden

Boosting has more knobs than most methods, and they interact:

| Parameter | Controls |
|---|---|
| Learning rate $\eta$ | Contribution per tree |
| Number of trees | Total capacity — **coupled to $\eta$** |
| Max depth | Interaction order |
| Subsample ratio | Stochastic gradient boosting |
| Column subsample | Feature-level decorrelation |
| $L_1$ / $L_2$ on leaf weights | Leaf value shrinkage |

$\eta$ and tree count trade off directly: halving the learning rate roughly doubles the trees needed. **Tuning either in isolation is meaningless**, which makes this a case where a grid over the pair is genuinely required rather than merely thorough. See [[model-selection]].

> ### `boosting-buys-accuracy-with-selection-effort`
> **Gradient boosting's tabular advantage is partly real and partly an artefact of how much tuning it receives. A heavily-tuned boosted model compared against a default-configuration neural network is not a comparison of methods, and this asymmetry is common in published benchmarks.**
> ^[generated. rests-on: imported:tabular-benchmark-practice]

## Interpretability

Individual trees are legible; an ensemble of several hundred is not. The standard recourse is post-hoc **feature attribution** — SHAP values in particular, which have an efficient exact algorithm for tree ensembles and are the main reason boosted models are considered explainable at all.

That should be held loosely. **An attribution over an opaque ensemble explains an approximation of the model**, and the fidelity of that approximation is rarely reported. See [[interpretability]].

## See Also

- [[model-selection]] · [[regularization]] · [[interpretability]] · [[feature-engineering]] · [[representation-learning]]
- [[probabilistic-classification]] · [[probability-calibration]] · [[class-imbalance-evaluation]] · [[predictive-validity]] · [[theory-based-modelling]]
