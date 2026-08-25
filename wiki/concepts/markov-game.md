---
title: "Markov Decision Process"
type: concept
tags: [markov-model, reinforcement-learning, machine-learning, dynamic-programming, game-theory, multi-agent]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 15%
  generated: 5%
  imported: 78%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Markov Decision Process

> ⚠️ **No held source.** This page is entirely background knowledge, marked `imported:` throughout. It exists because [[reinforcement-learning]] and its methods are unreadable without the formalism. **Acquiring a foundational RL text would ground it.**

The standard formalism for sequential decision-making under uncertainty. An MDP is a tuple $(S, A, T, R, \gamma)$:

| Element | Meaning |
|---|---|
| $S$ | State space |
| $A$ | Action space — see the discretisation problem below |
| $T(s' \mid s, a)$ | Transition function |
| $R(s, a)$ | Reward function |
| $\gamma \in [0, 1]$ | Discount factor |

## The Markov Property

**The next state depends only on the current state and action, not on the history.** $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)$.

This is the assumption doing all the work, and it is usually false as stated. The standard repair is to widen the state until it *is* true — folding relevant history into $s$ — which trades the Markov property against state-space size. **A model that appears to satisfy the Markov property often does so because its state representation was designed to.**

Where history cannot be folded in, the problem becomes a **partially observable** MDP, and the agent must maintain a belief distribution over states rather than knowing $s$ outright.

## What the Discount Factor Does

$\gamma$ serves two purposes that are worth separating:

- **Mathematical:** it guarantees the infinite-horizon return $\sum_t \gamma^t r_t$ converges.
- **Behavioural:** it encodes preference for earlier reward.

For episodic tasks with bounded horizons, $\gamma = 1$ is admissible and the first purpose falls away. Where a discount is applied anyway, it is a *modelling choice about impatience or credit assignment*, and should be argued for rather than assumed. See [[temporal-difference-learning]].

## Extensions

| | Agents | Adds |
|---|---|---|
| **MDP** | One | — |
| **POMDP** | One | Partial observability; belief states |
| **Markov game** | Several | Joint actions; equilibrium rather than optimum |

The multi-agent case is not a small generalisation. With several learners, each agent's environment is **non-stationary** — it changes as the others learn — so the fixed-point arguments underpinning single-agent convergence no longer hold. The solution concept shifts from *optimal policy* to *equilibrium*, which brings in [[game-theory]]. See [[multi-agent-reinforcement-learning]].

## Why the Action Space Matters More Than It Looks

$A$ is usually presented as given. It rarely is.

Where behaviour is continuous, $A$ is a discretisation someone chose, and **that choice fixes which counterfactuals are expressible.** A framework cannot ask "would a different action have been better?" about an action its action space cannot represent. Coarser action spaces make optimal-policy analysis tractable and make the answer concern a coarsened problem.^[generated: not stated in the standard formalism, which treats A as given]

## Limitations

- **The Markov property is an assumption, not a finding**, and is usually secured by state design rather than verified.
- **$T$ is often unknown**, which is what motivates the sample-based methods — [[temporal-difference-learning]] and [[proximal-policy-optimization|policy gradient]] — over [[value-iteration|dynamic programming]].
- **Reward must be specified.** Everything downstream optimises what was written, not what was meant.

## See Also

- [[reinforcement-learning]] · [[value-iteration]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[proximal-policy-optimization]]
- [[multi-agent-reinforcement-learning]] · [[policy-modelling]] · [[imitation-learning]] · [[agent-based-simulation]]
- [[bayesian-inference]] · [[uncertainty-quantification]]
