---
title: "Uncertainty Quantification"
type: concept
tags: [uncertainty-quantification, statistics, bayesian, calibration, probabilistic-classification, ranking-system, evaluation, inference]
sources: [raw/papers/bayesian-true-skill-rating.md, raw/papers/expected_value_possession_framework.md, raw/papers/llm_factcheck_consistency_certainty.md]
confidence: 0.8
provenance:
  extracted: 35%
  inferred: 60%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-27
updated: 2026-07-27
---

# Uncertainty Quantification

Estimating not just what a model predicts but **how much to trust it**. The distinction that organises the field:

| | Aleatoric | Epistemic |
|---|---|---|
| Source | Irreducible randomness in the outcome | The model's own ignorance |
| Reducible by more data? | **No** | **Yes** |
| Expressed as | A [[probabilistic-classification\|calibrated probability]] | A distribution *over* parameters or predictions |

A coin flip is aleatoric: a perfectly calibrated 0.5 is the right answer and no amount of data improves it. A model predicting 0.5 because it has never seen this input is epistemic — and the two are **indistinguishable from the output alone.** That confusion is the most common failure in this area.

## Where the Vault Handles Epistemic Uncertainty Well

The rating systems are the clear case, and it is what distinguishes them from their predecessor.

[[elo-rating-system|Elo]] carries a point estimate with no uncertainty, handling new players by ad-hoc "provisional" flags. [[glicko-rating-system|Glicko]] represents skill as a Gaussian with a **rating deviation**, which yields two behaviours Elo cannot express:

- **Update size adapts.** A result against an uncertain opponent moves your rating less; your own uncertain rating moves further.
- **Inactivity widens uncertainty.** A returning competitor's rating adjusts quickly rather than being anchored to stale evidence.

Glicko-2 adds volatility; [[trueskill]] carries Gaussian beliefs through [[factor-graph|factor-graph]] [[message-passing]]. The Bayesian machinery elsewhere — [[gaussian-process|GPs]], [[car-prior|CAR priors]], [[inla]] in [[martingale-epv]] — produces posteriors rather than point estimates for the same reason.

## Where It Is Discarded

Worth flagging, because computing uncertainty and *using* it are different things.

Both football applications of Glicko-2 — [[duel-skill-rating]] and [[league-strength-rating]] — track rating deviation and then feed **point estimates** downstream. The [[transfer-performance-prediction|transfer model]] consumes a club's rating as a number, discarding whether that rating rests on many matches or few.

That is a missed opportunity rather than an error: knowing *which* ratings are reliable is exactly the information a recruitment model should propagate, and a club rated 1700 on thin continental evidence is not the same input as one rated 1700 on a full season.

## Calibration Is Not the Whole Story

[[probability-calibration]] addresses aleatoric uncertainty: do stated probabilities match observed frequencies? Necessary, and insufficient in two ways.

**A base-rate predictor is perfectly calibrated and useless.** [[vdep|VAEP's conceding classifier]] has the best Brier score in its comparison and an F1 of 0.000. See [[class-imbalance-evaluation]].

**Calibration says nothing about coverage.** A [[probability-surface|pass surface]] is calibrated only where passes were observed; the rest is extrapolation from the convolutional prior, and the reported ECE speaks to none of it. See [[single-pixel-supervision]].

## Neural Networks Are Overconfident

Guo et al. (2017): modern networks are systematically overconfident, counter to the intuition that better accuracy brings better probabilities. Post-hoc temperature scaling is the standard remedy and is close to free — monotonic, so rankings and accuracy are provably unchanged.

[[expected-value-possession-framework|Fernández et al.]] apply it and obtain $T < 1$ for two components, which *sharpens* rather than softens — the opposite of the usual correction, implying those models were **under**-confident.

## In Language Models

[[llm-factcheck-consistency-certainty|The PCC work]] separates a model's stated certainty from its consistency across samples — a direct analogue of the aleatoric/epistemic split, where sampling variance stands in for parameter uncertainty. Ensembles and dropout-at-inference do the same job in the general case.

## See Also

- [[probability-calibration]] · [[probabilistic-classification]] · [[class-imbalance-evaluation]]
- [[glicko-rating-system]] · [[trueskill]] · [[elo-rating-system]] · [[bayesian-inference]]
- [[gaussian-process]] · [[inla]] · [[car-prior]] · [[message-passing]]
- [[league-strength-rating]] · [[transfer-performance-prediction]] · [[single-pixel-supervision]]
- [[bayesian-true-skill-rating|TrueSkill Summary]] · [[llm-factcheck-consistency-certainty|PCC Summary]]
