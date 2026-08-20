---
title: "Proximal Policy Optimization"
type: concept
tags: [policy-gradient, reinforcement-learning, deep-learning, machine-learning, training-technique, simulator, alignment, markov-model]
sources: [raw/papers/ai_football_reinforcement_learning.md, raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.8
provenance:
  extracted: 42%
  inferred: 26%
  generated: 7%
  imported: 25%
  ambiguous: 0%
lifecycle: draft
created: 2026-08-08
updated: 2026-08-08
---

# Proximal Policy Optimization

A policy-gradient method (Schulman et al., 2017) that improves a policy by ascending a **clipped surrogate objective**, alternating between optimising it and collecting fresh interaction data:

$$J(\theta) = \mathbb{E}\left[\min\left(r(\theta)\hat{A}_{\theta_{old}}(s,a),\ \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_{\theta_{old}}(s,a)\right)\right]$$

where $r(\theta) = \pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$ is the probability ratio between new and old policies, and $\hat{A}(s,a)$ estimates the advantage $Q(s,a) - V(s)$.

## The Clip Is the Whole Idea

Plain policy gradient can take a step so large it destroys the policy, because the gradient is estimated under the *old* policy and stops being valid once you move far from it.

The clip caps how much the objective can improve from moving the probability ratio outside $[1-\varepsilon, 1+\varepsilon]$. Taking the **minimum** of clipped and unclipped means the update gets no credit for moving far in a favourable direction, but still gets full penalty for moving far in an unfavourable one — a deliberately pessimistic bound that keeps successive policies close.

**"Proximal" names a constraint on how far the policy may move, not on what it may learn.**

## Where It Sits in the Vault's RL Coverage

This page fills a real hole. The vault had [[value-iteration|dynamic programming]], [[temporal-difference-learning|TD]] and [[deep-q-network|DQN]] — **all value-based** — and no policy-gradient method at all, despite [[rlhf|RLHF]] depending on one.

| Family | Learns | Held examples |
|---|---|---|
| **Dynamic programming** | $V$ or $Q$, from known dynamics | [[expected-threat\|xT]] |
| **Temporal difference** | $Q$, by bootstrapping | [[action-valuation-multi-agent-reinforcement-learning\|Nakahara et al.]] (SARSA), [[adaptive-action-supervision-multi-agent-rl\|Fujii et al.]] (DDQN) |
| **Policy gradient** | $\pi$ **directly** | [[ai-football-reinforcement-learning\|Scott et al.]] (PPO), [[training-lm-follow-instructions-with-human-feedback\|InstructGPT]] (PPO) |

The distinction matters for what a framework can produce. **A value-based method hands you $Q(s,a)$ for every action — which is what makes [[action-valuation-multi-agent-reinforcement-learning|Nakahara et al.'s]] counterfactual valuation possible. A policy-gradient method hands you $\pi(a|s)$ and no per-action value.**

> ### `policy-gradient-forecloses-action-valuation`
> **Policy-gradient methods optimise the policy without ever materialising per-action values, so an RL framework built on them cannot be repurposed into an action-valuation metric. The algorithm family choice silently determines whether a valuation product is available downstream.**
> ^[generated: no source states this; drawn from the contrast between the two held football RL lines. rests-on: source:scott-ppo-setup, source:nakahara-q-per-action]

That may explain a pattern otherwise easy to miss: **Scott et al. produce no player metric at all**, analysing pass-network structure instead, while the two later Fujii-group papers using value-based methods both produce or attempt one. The methodological turn from PPO to SARSA/DDQN across these three papers is also a turn from *describing* agents to *valuing* actions.

## The Two Held Applications Use It Very Differently

| | [[ai-football-reinforcement-learning\|Scott et al.]] | [[training-lm-follow-instructions-with-human-feedback\|InstructGPT]] |
|---|---|---|
| Reward | Goals ± , plus a **checkpoint reward** for sparse early goals | A **learned reward model** from human preferences |
| Policy net | IMPALA CNN over a [[action-space-design\|Super Mini Map]] | A pretrained language model |
| Anchoring | The clip alone | Clip **plus a [[kl-divergence\|KL]] penalty** to the base model |
| Purpose | Produce competitive agents to analyse | Align a model to human intent |

The **checkpoint reward** is worth noting: a small shaped reward added because goal reward is too sparse to bootstrap from. That is the [[rare-event-proxy-targets|Fujii-group signature]] appearing a third time — proxy substitution inside a reward function, exactly as [[adaptive-action-supervision-multi-agent-rl|Fujii et al.]] later add a shot reward "because the goal reward was sparse and limited". Here it arrives from the GFootball authors rather than from the group.

InstructGPT's KL penalty is the better-behaved arrangement for the [[imitation-reward-tradeoff|imitation/reward trade-off]], because the anchoring strength is an explicit reported coefficient. PPO's clip anchors to *the previous policy* rather than to a reference policy, which stabilises optimisation without expressing any preference about where the policy ends up.

## Practical Caveats

- **On-policy**, so data cannot be reused across large policy changes — sample-inefficient compared to [[deep-q-network|DQN]] with replay. Scott et al. needed **50–200 million timesteps**.
- **$\varepsilon$ is a free parameter** and is not reported in the held football application. See [[free-parameters-load-bearing]].
- **Advantage estimation is a separate design choice** (GAE and its own $\lambda$), also unreported.
- **Reward shaping does the heavy lifting** in sparse-reward settings, and the shaped component is rarely ablated.

## Beyond Sport

PPO became the default deep-RL algorithm for roughly the period 2017–2023 on the strength of being simpler than TRPO and more robust than vanilla policy gradient, which is why it appears in both a football simulator and a language-model alignment pipeline in this vault with no connection between them. Its role in [[rlhf|RLHF]] made it, for a while, the most economically consequential RL algorithm in use.

## See Also

- [[reinforcement-learning]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[value-iteration]] · [[markov-game]] · [[policy-modelling]]
- [[rlhf]] · [[kl-divergence]] · [[imitation-reward-tradeoff]] · [[rare-event-proxy-targets]] · [[action-space-design]]
- [[residual-connections]] · [[adam-optimizer]] · [[agent-based-simulation]] · [[google-research-football]] · [[free-parameters-load-bearing]]
- [[atom-scott]] · [[keisuke-fujii]] · [[openai]]
- [[ai-football-reinforcement-learning|Scott et al. Summary]] · [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
