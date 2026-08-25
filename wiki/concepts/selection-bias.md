---
title: "Selection Bias"
type: concept
tags: [selection-bias, statistics, evaluation, machine-learning, positive-unlabeled-learning, predictive-validity, domain-adaptation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 8%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Selection Bias

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Systematic non-representativeness of a sample relative to the population it is meant to describe. The sample is not wrong about itself — it is wrong about what it is taken to stand for.

## Why It Is Not Fixed by More Data

The property that distinguishes it from noise, and the reason it is dangerous.

**More data from a biased process gives a more precise estimate of the wrong quantity.** Confidence intervals narrow around a value that was never the target. Every diagnostic that measures variance says the estimate is improving.

That is the opposite of the usual relationship between sample size and reliability, and it means **no internal check detects it.** Establishing that a sample is representative requires information from outside the sample.

## Common Mechanisms

^[imported]

| Mechanism | Effect |
|---|---|
| **Survivorship** | Only surviving units observed — failures absent by construction |
| **Collider conditioning** | Conditioning on a common effect induces a correlation between its causes |
| **Self-selection** | Units choose whether to be observed, on grounds related to the outcome |
| **Measurement availability** | Only what is cheap to measure gets measured |

Collider conditioning is the least intuitive and the most common in practice. Restricting attention to a subgroup defined by an outcome — successful cases, admitted patients, funded projects — **creates** associations among the causes that do not exist in the population.

## The Deployment Version

The training set is one sample; the deployment population is another. Where they differ, the model's measured performance describes neither.

This is the same structure as [[domain-adaptation]], arriving from the sampling side rather than the environment side. **The correction methods are largely shared** — reweighting, importance sampling — and so is the central difficulty: they need some information about the target, which is usually what is unavailable.

## Selection on the Outcome

The sharpest case, because it is often invisible.

If the decision to observe a unit depends on the outcome being modelled, the model learns the *selection process* alongside the phenomenon and cannot separate them. Historical decisions become training labels; the model reproduces the decision rule and is evaluated as though it had predicted the phenomenon.

> ### `a-model-trained-on-selected-outcomes-learns-the-selector`
> **Where observation depends on the outcome, a model fitted to observed cases predicts what the selector chose, not what was true. Accuracy against the observed set can be high while the model has learned nothing about the unobserved population — and the unobserved population is usually the one the model will be applied to.**
> ^[generated. rests-on: imported:selection-on-outcome]

Where only positives are labelled and the rest are unlabelled rather than negative, the standard classifier setup is misspecified from the start — the formalised version of this problem.

## What Helps

- **State the sampling frame explicitly.** Most selection bias survives because nobody wrote down how the sample was obtained.
- **Compare against an external benchmark** — a census, a known base rate, an independently drawn sample.
- **Test on a deliberately different population**, not a random split of the same one. A random split preserves the bias exactly.
- **Sensitivity analysis** — how strong would the selection have to be to overturn the result?

The third is the most under-used. **A held-out split shares the sampling frame**, so it validates against the same distortion it was drawn from.

## See Also

- [[domain-adaptation]] · [[predictive-validity]] · [[class-imbalance-evaluation]] · [[model-selection]]
- [[probability-calibration]] · [[uncertainty-quantification]] · [[agent-based-simulation]] · [[counterfactual-simulation]]
