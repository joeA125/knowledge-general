---
title: "Value Iteration"
type: concept
tags: [dynamic-programming, reinforcement-learning, markov-model, machine-learning, approximation]
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

# Value Iteration

> ⚠️ **No held source.** Background knowledge, marked `imported:` throughout.

Solving a [[markov-game|Markov decision process]] by repeatedly applying the Bellman optimality equation as an update rule until the value function converges:

$$V_{k+1}(s) \leftarrow \max_a \Big[ R(s,a) + \gamma \sum_{s'} T(s' \mid s, a) V_k(s') \Big]$$

The Bellman operator is a contraction under $\gamma < 1$, so iteration converges to the unique optimal $V^*$ regardless of initialisation. The optimal policy is then read off greedily.

## The Requirement That Rules It Out

**Value iteration needs $T$ — the transition model — explicitly.** That is the single condition separating it from everything else in the RL toolkit.

| | Needs a model | Needs complete episodes | Bootstraps |
|---|---|---|---|
| **Value iteration** | **Yes** | No | Yes |
| Monte Carlo return | No | **Yes** | No |
| [[temporal-difference-learning\|Temporal difference]] | No | No | Yes |

Where transitions are known — a board game, a grid world, a queueing system — value iteration is exact and there is no reason to sample. Where they are not, it is unavailable, which is most of the interesting cases.

## Value Iteration and Policy Iteration

^[imported]

Two ways to reach the same fixed point:

- **Value iteration** — improve values, extract a policy at the end.
- **Policy iteration** — alternate full policy evaluation with policy improvement.

Policy iteration converges in fewer iterations, each of which is more expensive. Value iteration is the more common starting point precisely because each sweep is cheap and the algorithm is a single line.

## The Practical Constraint

The sum over $s'$ must be computed for every state and action, so cost scales with $|S| \times |A|$ per sweep — and $|S|$ is usually the killer. Any state space rich enough to be interesting is too large to enumerate.

**That is the whole reason for function approximation.** Replacing the table $V(s)$ with a parameterised $V(s; \theta)$ makes large state spaces tractable and gives up the convergence guarantee at the same stroke: the contraction argument holds for the exact operator, not its projected approximation. See [[deep-q-network]] for the machinery invented to manage the resulting instability.

## Why the Distinction Recurs

The choice between dynamic programming and sampling tracks **how coarse the state representation is**, not any deeper commitment. Coarsen a continuous problem enough — bin it into a grid — and the transition matrix becomes estimable, at which point DP is available again. That is a common move, and its cost is that the answer concerns the coarsened problem.^[generated]

## See Also

- [[markov-game]] · [[reinforcement-learning]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[proximal-policy-optimization]]
- [[approximate-message-passing]] · [[expectation-propagation]] · [[bayesian-inference]]
