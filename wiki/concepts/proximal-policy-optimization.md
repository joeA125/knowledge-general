---
title: "Proximal Policy Optimization"
type: concept
tags: [policy-gradient, reinforcement-learning, deep-learning, machine-learning, training-technique, alignment, markov-model]
sources: [raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.75
provenance:
  extracted: 25%
  inferred: 28%
  generated: 7%
  imported: 38%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-08
updated: 2026-08-14
---

# Proximal Policy Optimization

A policy-gradient method (Schulman et al., 2017) that improves a policy by ascending a **clipped surrogate objective**, alternating between optimising it and collecting fresh interaction data:

$$J(\theta) = \mathbb{E}\left[\min\left(r(\theta)\hat{A}(s,a),\ \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}(s,a)\right)\right]$$

where $r(\theta) = \pi_\theta(a \mid s)/\pi_{\theta_{old}}(a \mid s)$ is the probability ratio between new and old policies, and $\hat{A}(s,a)$ estimates the advantage $Q(s,a) - V(s)$.

## The Clip Is the Whole Idea

Plain policy gradient can take a step so large it destroys the policy, because the gradient is estimated under the **old** policy and stops being valid once you move far from it.

The clip caps how much the objective can improve from moving the probability ratio outside $[1-\varepsilon, 1+\varepsilon]$. Taking the **minimum** of clipped and unclipped means an update gets no credit for moving far in a favourable direction, but takes the full penalty for moving far in an unfavourable one — a deliberately pessimistic bound that keeps successive policies close.

**"Proximal" names a constraint on how far the policy may move, not on what it may learn.**

It is the practical successor to TRPO, which enforced a similar constraint via a KL trust region requiring second-order optimisation. PPO achieves most of the benefit with a first-order method and a few lines of code, which is the main reason it displaced it.^[imported]

## Where It Sits Among the Families

| Family | Learns | Needs |
|---|---|---|
| [[value-iteration\|Dynamic programming]] | $V$ or $Q$ | A known transition model |
| [[temporal-difference-learning\|Temporal difference]] | $Q$, by bootstrapping | Samples |
| **Policy gradient** | $\pi$ **directly** | Samples, and on-policy data |

> ### `policy-gradient-forecloses-action-valuation`
> **Policy-gradient methods optimise the policy without ever materialising per-action values, so a framework built on them cannot be repurposed to score individual actions. The algorithm family silently determines what analysis is available downstream.**
> ^[generated. rests-on: imported:rl-family-taxonomy]

A value-based method hands you $Q(s,a)$ for every action, which is what makes counterfactual action scoring possible. A policy-gradient method hands you $\pi(a \mid s)$ and nothing else. **If the eventual product is a per-action metric rather than a controller, the choice of family has already decided the question.**

## The Held Application

[[training-lm-follow-instructions-with-human-feedback|InstructGPT]] is the vault's one held use, and it is unusual in three ways:

| | Standard RL | InstructGPT |
|---|---|---|
| Reward | Specified | **Learned** from human preference comparisons |
| Episode | Many steps, state transitions | **One step** — emit a completion, receive a score |
| Objective | Maximise reward | Maximise reward **minus a [[kl-divergence\|KL]] penalty** to the starting policy |

The third row is the transferable part. **PPO's clip anchors to the *previous* policy; the KL penalty anchors to a *reference* policy.** The clip stabilises optimisation without expressing any preference about where the policy ends up; the KL term does express one.

That distinction matters wherever a model is being moved away from a starting point that had value in itself — which is the whole structure of alignment. See [[rlhf]] and [[imitation-learning]].

## Practical Caveats

- **On-policy**, so data cannot be reused across large policy changes. Sample-inefficient relative to [[deep-q-network|DQN]] with replay, and typically needing tens to hundreds of millions of environment steps.^[imported]
- **$\varepsilon$ is a free parameter**, commonly 0.1–0.3, and its effect is rarely swept. See [[model-selection]].
- **Advantage estimation is a separate design choice** — generalised advantage estimation brings its own $\lambda$, also usually asserted.
- **Reward shaping does the heavy lifting** in sparse-reward settings, and the shaped component is rarely ablated. See [[rare-event-proxy-targets]].

## Why It Became the Default

PPO dominated deep RL for roughly 2017–2023 by being simpler than TRPO and more robust than vanilla policy gradient — a middle option that was good enough almost everywhere and easy to implement correctly.

Its role in [[rlhf|RLHF]] made it, for a period, the most economically consequential RL algorithm in use, which is a strange fate for a method designed for continuous control.

## See Also

- [[reinforcement-learning]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[value-iteration]] · [[markov-game]] · [[policy-modelling]]
- [[rlhf]] · [[kl-divergence]] · [[imitation-learning]] · [[rare-event-proxy-targets]] · [[model-selection]] · [[agent-based-simulation]]
- [[openai]] · [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
