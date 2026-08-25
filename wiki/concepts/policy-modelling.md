---
title: "Policy Modelling"
type: concept
tags: [policy-modelling, reinforcement-learning, imitation-learning, markov-model, machine-learning, evaluation]
sources: [raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.65
provenance:
  extracted: 8%
  inferred: 22%
  generated: 8%
  imported: 60%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Policy Modelling

Estimating the policy agents **actually follow**, rather than solving for the one they should. In [[markov-game|MDP]] terms: learning $\pi(a \mid s) = \mathbb{P}(A = a \mid s)$ from observed behaviour and treating it as the quantity of interest.

> ⚠️ **60% `imported:`.** The framing here is standard in offline RL and inverse RL; neither literature is held.

## The Inversion

Reinforcement learning treats the behaviour policy as something to *improve*, or to correct for via importance weighting. Policy modelling treats it as the **estimand**.

Two reasons this is worth doing, and both generalise beyond RL:

**The policy is itself the finding.** Knowing what agents actually do in a given situation — the distribution over their choices — is directly informative. Where behaviour diverges from what a value model rewards, that gap is the actionable output.

**The optimal-policy counterfactual is often unfounded.** An optimal-policy value function must evaluate actions nobody took. Over a large action space that is extrapolation, not estimation, and no data contains perfect play to learn it from.

## Where the Choice Shows Up Mechanically

It is not only a framing preference. The [[temporal-difference-learning|TD]] update encodes it:

| | On-policy (SARSA) | Off-policy (Q-learning) |
|---|---|---|
| Target | $Q(s_{t+1}, a_{t+1})$ | $\max_a Q(s_{t+1}, a)$ |
| Learns | Value of observed behaviour | Value of optimal behaviour |
| As a metric | "What was this worth, given how they act?" | "What would this be worth, played perfectly?" |

**A one-line substitution converts a descriptive framework into a prescriptive one**, and papers rarely discuss the choice. See [[deep-q-network]] for the off-policy case and its stabilisers.

## The Gap Between Observed and Optimal

Where both quantities are computed, their difference is the deliverable: what the observed policy leaves on the table relative to the best available action.

Structurally this is the advantage function $A(s,a^*) = Q(s,a^*) - V(s)$, repurposed as an interpretive device rather than a training signal.

> ### `the-optimality-gap-depends-on-the-counterfactual-assumption`
> **Every method computing an observed-versus-optimal gap must fill the counterfactual arm somehow — a learned surface, a simulation, an imitation prior. The gap's size depends on that assumption, and reporting the gap without the assumption makes a modelling choice look like a finding.**
> ^[generated: drawn across the methods above; no held source states it. rests-on: imported:offline-rl-counterfactual-problem]

[[rlhf|RLHF]] is the held instance where this is handled well: the [[kl-divergence|KL]] coefficient anchoring the policy to its starting point is exactly such an assumption, and it is reported and swept as standard.

## Caveats

- **A fitted policy is a population average.** It describes how agents behave, not how *this* agent behaves. Conditioning on identity gives individual policies, at the cost of data per agent.
- **Behaviour encodes constraint, not only judgement.** An agent may act as it does because it was instructed to. The policy conflates individual decision-making with imposed structure, and nothing separates them. See [[imitation-learning]].
- **Circularity risk.** If the value model is trained on outcomes generated *under* this policy, then "the policy is suboptimal by the value model's lights" is a claim about internal consistency, not about the world.

## See Also

- [[reinforcement-learning]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[proximal-policy-optimization]] · [[markov-game]] · [[value-iteration]]
- [[imitation-learning]] · [[rlhf]] · [[kl-divergence]] · [[multi-agent-reinforcement-learning]]
- [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
