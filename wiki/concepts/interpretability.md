---
title: "Interpretability"
type: concept
tags: [interpretability, machine-learning, evaluation, deep-learning, attention, feature-attribution, uncertainty-quantification]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 22%
  generated: 8%
  imported: 68%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Interpretability

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

The degree to which a model's outputs can be explained in human terms. Not one property but several, often conflated — and the conflation is where most disputes about it come from.

## Four Different Questions

| Question | Answered by |
|---|---|
| **How does the model work?** | Transparency — the mechanism is legible |
| **Why this output?** | Local explanation — attribution for one case |
| **What does the model rely on generally?** | Global explanation — feature importance |
| **What would change the output?** | Counterfactual explanation |

A model can be strong on one and useless on another. A linear model is transparent and gives poor counterfactual guidance in an interacting system; a SHAP-explained gradient-boosted ensemble gives good local attribution with no transparency at all.

**Asking whether a model "is interpretable" is usually asking the wrong question.** Which of the four is needed depends on what the explanation is for.

## Intrinsic and Post-Hoc

^[imported]

- **Intrinsic** — the model is simple enough to inspect: linear coefficients, shallow trees, explicit rules.
- **Post-hoc** — a separate procedure explains an opaque model: attribution methods, surrogate models, saliency.

The trade is usually presented as accuracy against interpretability. That framing is too neat. **At small data scale the trade often does not exist** — a constrained model may match or beat a flexible one because the flexibility cannot be supported by the data, in which case transparency is free.

Where the trade is real, post-hoc methods carry a specific risk: they explain *an approximation of* the model, and the fidelity of that approximation is rarely reported.

## Attention Weights Are Not Explanations

^[imported: an active dispute in the literature]

A recurring case worth stating because the temptation is strong. [[attention-mechanism|Attention]] weights are visible, sum to one, and look like a distribution over evidence — so they are routinely read as showing what a model "used".

The objection is that attention weights can often be substantially altered without changing the output, which means they are not uniquely determined by the prediction and therefore cannot be its explanation.

**A weaker use survives the objection.** Asking whether attention is *distributed* across a context window — rather than what any individual weight means — is a claim about the model's structure rather than its reasoning, and is a legitimate diagnostic. Concentration at one end of a fixed window suggests the window is mis-sized.

> ### `diagnostic-use-survives-what-explanatory-use-does-not`
> **A quantity too unstable to serve as an explanation may still be stable enough to serve as a diagnostic. The distinction is between claiming what a model attended to and claiming how its attention was shaped.**
> ^[generated. rests-on: imported:attention-is-not-explanation-debate]

## Why It Is Wanted

Three reasons that pull in different directions:

- **Trust** — the user must believe the output before acting on it
- **Debugging** — the developer must find the failure
- **Actionability** — the output must say what to *do*, not only what will happen

The third is the one most often unmet and least often stated. A model that accurately predicts an outcome may be useless for changing it, and "interpretable" is frequently used to mean "prescriptive" without the substitution being noticed.

## Limitations

- **Interpretability is not correctness.** A legible wrong model is still wrong, and legibility makes it more persuasive.
- **Explanations can be gamed** — a model can be constructed to produce plausible attributions independent of its actual computation.
- **Human plausibility is not fidelity.** An explanation that satisfies a reader may bear no relation to the mechanism.

## See Also

- [[attention-mechanism]] · [[model-selection]] · [[uncertainty-quantification]] · [[predictive-validity]] · [[probability-calibration]]
- [[transformer]] · [[representation-learning]] · [[capability-profiling]] · [[class-imbalance-evaluation]]
