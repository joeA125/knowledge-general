---
title: "Uncertainty Quantification"
type: concept
tags: [uncertainty-quantification, statistics, bayesian, calibration, probabilistic-classification, ranking-system, evaluation, inference]
sources: [raw/papers/bayesian-true-skill-rating.md, raw/papers/llm_factcheck_consistency_certainty.md]
confidence: 0.8
provenance:
  extracted: 32%
  inferred: 40%
  generated: 8%
  imported: 18%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Uncertainty Quantification

Estimating not just what a model predicts but **how much to trust it**. The distinction that organises the field:

| | Aleatoric | Epistemic |
|---|---|---|
| Source | Irreducible randomness in the outcome | The model's own ignorance |
| Reducible by more data? | **No** | **Yes** |
| Expressed as | A [[probabilistic-classification\|calibrated probability]] | A distribution *over* parameters or predictions |

A coin flip is aleatoric: a perfectly calibrated 0.5 is the right answer and no amount of data improves it. A model predicting 0.5 because it has never seen this input is epistemic — and **the two are indistinguishable from the output alone.** That confusion is the most common failure in this area, and it matters because the correct response differs: gather more data, or accept the limit.

## Where Epistemic Uncertainty Is Handled Well

The rating systems are the clearest worked case, and the progression shows exactly what the uncertainty buys.

[[elo-rating-system|Elo]] carries a point estimate and handles new competitors with ad-hoc provisional flags. [[glicko-rating-system|Glicko]] represents skill as a Gaussian with a **rating deviation**, giving two behaviours Elo cannot express:

- **Update size adapts.** A result against an uncertain opponent moves your rating less; your own uncertain rating moves further.
- **Inactivity widens uncertainty.** A returning competitor adjusts quickly rather than being anchored to stale evidence.

Glicko-2 separates volatility from uncertainty; [[trueskill|TrueSkill]] carries Gaussian beliefs through [[message-passing]] with [[expectation-propagation]] handling the intractable factors.

**The general lesson is that the uncertainty is not an add-on — it changes the update rule.** See `a-point-estimate-forces-a-global-learning-rate` on [[elo-rating-system]].

## Where It Is Discarded

Computing uncertainty and *using* it are different things, and the gap is common.

A system may track a posterior correctly at every stage and then pass a **point estimate** to whatever consumes it. The distinction between an estimate resting on abundant evidence and one resting on almost none is erased precisely at the interface where it would have been useful.

> ### `uncertainty-is-usually-computed-and-then-discarded-at-the-interface`
> **Pipelines that model uncertainty internally frequently expose only a mean, so downstream consumers cannot distinguish a well-measured estimate from a poorly-measured one. The loss occurs at the boundary rather than in the model, which is why it survives review of either component alone.**
> ^[generated. rests-on: source:trueskill-conservative-display]

TrueSkill's $\mu - 3\sigma$ display is the counter-example worth copying: a single number that **carries** the uncertainty rather than dropping it, by being deliberately conservative.

## Calibration Is Necessary and Insufficient

[[probability-calibration|Calibration]] addresses aleatoric uncertainty — do stated probabilities match observed frequencies? It fails to be sufficient in two ways:

**A base-rate predictor is perfectly calibrated and useless.** Under heavy [[class-imbalance-evaluation|class imbalance]] this is not hypothetical: a model with an excellent Brier score can have an F1 of zero, and reporting only the first inverts the conclusion.

**Calibration says nothing about coverage.** A model is calibrated where data was observed; elsewhere it is extrapolating from its inductive bias, and a reported calibration error speaks to none of that region. See [[selection-bias]].

## Neural Networks Are Overconfident

^[imported: Guo et al. 2017 and subsequent work; not held here]

Modern networks are systematically overconfident, and increasingly so with capacity — counter to the intuition that better accuracy brings better probabilities. Post-hoc temperature scaling is the standard remedy and is close to free: a single positive scalar preserves ranking exactly, so accuracy and AUC are provably unchanged.

Worth noting that the correction is not always in the expected direction. Fitting a temperature $T < 1$ **sharpens** rather than softens, implying the model was *under*-confident — which happens and is easy to miss if the correction is assumed rather than fitted.

## In Language Models

[[llm-factcheck-consistency-certainty|The PCC work]] separates a model's **stated certainty** from its **consistency across samples**. That is a direct analogue of the aleatoric/epistemic split, with sampling variance standing in for parameter uncertainty.

The general form is that **self-reported confidence and behavioural consistency are different measurements**, and a model can be high on one and low on the other. Ensembles and dropout-at-inference do the same job by other means.

## See Also

- [[probability-calibration]] · [[probabilistic-classification]] · [[class-imbalance-evaluation]] · [[predictive-validity]] · [[split-half-reliability]]
- [[glicko-rating-system]] · [[trueskill]] · [[elo-rating-system]] · [[bradley-terry-model]] · [[bayesian-inference]] · [[expectation-propagation]] · [[message-passing]]
- [[model-selection]] · [[selection-bias]] · [[counterfactual-simulation]] · [[interpretability]]
- [[bayesian-true-skill-rating|TrueSkill Summary]] · [[llm-factcheck-consistency-certainty|PCC Summary]]
