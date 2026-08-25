---
title: "Trajectory Prediction"
type: concept
tags: [trajectory-prediction, deep-learning, graph-neural-network, vae, rnn, spatiotemporal, generative-model, multi-agent, counterfactual]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 24%
  generated: 8%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Trajectory Prediction

> ⚠️ **No held source.** Background knowledge, marked `imported:`. This page arrived by copy during the 2026-08 migration and was rewritten when its only source proved to be football material held elsewhere.

Forecasting the future positions of one or more moving agents from their observed past. Distinct from ordinary time-series forecasting in two ways: the agents **interact**, and the future is **genuinely multimodal**.

## Why Averaging Fails

The central difficulty, and the one that determines the architecture.

A deterministic model minimising expected error over a multimodal future returns the **mean of the modes**. Where an agent might plausibly go left or right, the mean is straight ahead — a path neither mode contains, and often physically or contextually invalid.

> ### `deterministic-trajectory-models-predict-the-gap-between-options`
> **Minimising expected squared error over a multimodal distribution returns a point no mode occupies. In trajectory prediction this produces outputs that are not merely inaccurate but incoherent, and the loss registers them as good.**
> ^[generated. rests-on: imported:multimodal-trajectory-forecasting]

The standard response is a **stochastic latent**: sample a latent variable per rollout, so different samples commit to different modes rather than blending them. See [[variational-autoencoder]] and `averaging-over-modes-produces-invalid-outputs` on [[generative-model]].

## Interaction

The second difficulty. Agents do not move independently — each responds to the others, and a model treating them separately misses exactly the behaviour that makes the problem interesting.

| Approach | Mechanism | Cost |
|---|---|---|
| **Independent per-agent** | One model per agent, others in the input | No interaction modelled |
| **Pooled** | Aggregate neighbours into a context vector | Loses who is where |
| **[[graph-neural-network\|Graph]]** | [[message-passing]] between agents | Permutation-equivariant; over-smooths if deep |
| **[[transformer\|Attention]]** | Learned weighting over agents | Same family as graph; more parameters |

The last two are the same construction, as set out on [[message-passing]]. Both address the ordering problem directly: **a set of agents has no canonical order**, and imposing one is an arbitrary choice the model then learns from.

## The Horizon Problem

Error compounds. A model accurate at one second may be useless at five, because each predicted step becomes the input to the next and small errors accumulate into physically implausible paths.

This is the same compounding-error structure as autoregressive generation — [[imitation-learning|exposure bias]] under another name — and it sets a hard practical ceiling. **Reported accuracy is meaningless without the horizon**, and horizons vary widely across the literature, which makes cross-paper comparison harder than it looks.

## Evaluation

^[imported]

| Metric | Measures |
|---|---|
| **ADE** — average displacement error | Mean error across all predicted timesteps |
| **FDE** — final displacement error | Error at the horizon only |
| **Best-of-$N$** | The closest of $N$ sampled rollouts |

Best-of-$N$ deserves scepticism. It rewards a model for *covering* the outcome among many samples, not for assigning it high probability — so a model producing wildly diverse nonsense can score well. **Reporting best-of-$N$ without also reporting the likelihood of the true path overstates performance**, and the practice is common.

## The Model as a Reference

A use worth separating. A trajectory model trained on a population predicts what a *typical* agent would have done — which becomes a baseline against which a specific agent's deviation is measured.

The objective then inverts: the model is wanted for a well-calibrated notion of *normal* rather than for accuracy, and under a perfect predictor the measured deviation is identically zero. See [[counterfactual-baseline]].

## See Also

- [[variational-autoencoder]] · [[generative-model]] · [[graph-neural-network]] · [[message-passing]] · [[transformer]] · [[lstm]] · [[gated-recurrent-unit]]
- [[multi-agent-reinforcement-learning]] · [[agent-based-simulation]] · [[imitation-learning]] · [[counterfactual-baseline]] · [[counterfactual-simulation]]
- [[event-prediction]] · [[model-selection]] · [[uncertainty-quantification]]
