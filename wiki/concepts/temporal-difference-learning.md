---
title: "Temporal-Difference Learning"
type: concept
tags: [temporal-difference, reinforcement-learning, machine-learning, markov-model, dynamic-programming, discounting, deep-learning]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 18%
  generated: 6%
  imported: 74%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Temporal-Difference Learning

> ⚠️ **No held source.** Background knowledge, marked `imported:` throughout. The vault's only RL source is an application paper using [[proximal-policy-optimization|policy gradient]], not TD.

Learning a value function by updating each estimate toward **the next estimate plus the reward observed in between**, rather than toward a completed return. The defining move is *bootstrapping*: a guess is corrected using another guess.

$$\mathcal{L}_{TD} = \sum_t \big( r_{t+1} + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \big)^2$$

## What Bootstrapping Buys

| Method | Needs a model | Needs a complete episode | Bootstraps |
|---|---|---|---|
| [[value-iteration\|Dynamic programming]] | **Yes** | No | Yes |
| Monte Carlo return | No | **Yes** | No |
| **Temporal difference** | **No** | **No** | **Yes** |

TD occupies the only cell where neither a model nor a finished episode is required. That combination is why it, rather than DP, is the default when transitions are unknown and episodes are long — and it is why it can learn online, updating mid-episode.

The cost is bias. A Monte Carlo return is an unbiased sample of the true return with high variance; a TD target is biased by whatever the current estimate gets wrong, with much lower variance. **TD trades variance for bias, and the trade is usually worth taking.**

## SARSA and Q-Learning

The two canonical algorithms differ in one term of the target, and the difference decides what the learned $Q$ *means*.

| | SARSA | Q-learning |
|---|---|---|
| Target uses | $Q(s_{t+1}, a_{t+1})$ — the action **actually taken** | $\max_a Q(s_{t+1}, a)$ — the **best** action |
| Policy | **On-policy** | **Off-policy** |
| Converges toward | Value of the behaviour policy | Value of the optimal policy |
| Reads as | "What is this worth, given how the agent behaves?" | "What would this be worth, played perfectly?" |

> ### `on-policy-describes-off-policy-prescribes`
> **The on/off-policy choice determines whether a learned value function is a description of observed behaviour or a prescription for better behaviour. It is a one-line change in the update and a complete change in what the output claims.**
> ^[generated: the algorithms are standard; this framing of the consequence is drawn here. rests-on: imported:sarsa-qlearning-targets]

Where the goal is to *measure* an existing policy rather than improve on it, SARSA is the correct choice and Q-learning answers a different question. See [[policy-modelling]].

## Function Approximation Breaks the Guarantees

With a lookup table, the TD update is well-behaved and converges. With a neural network it becomes a loss with a **moving target** — the same parameters produce both $Q(s_t,a_t)$ and $Q(s_{t+1},\cdot)$, so the thing being regressed toward shifts with every step.

Combined with off-policy data and bootstrapping, this is the configuration known to diverge. [[deep-q-network|DQN's]] target networks and replay buffers exist specifically to manage it, and their necessity scales with **how much the agent influences its own training distribution** — an agent learning purely offline from fixed logged data needs far less of the apparatus.

A recurrent network is an alternative to some of it: carrying hidden state across the bootstrap does part of what a target network does, by conditioning the successor estimate on more than the instantaneous state. See [[gated-recurrent-unit]].

## TD($\lambda$) and the Spectrum

^[imported]

TD and Monte Carlo are endpoints of a continuum. TD($\lambda$) interpolates by weighting $n$-step returns geometrically: $\lambda = 0$ recovers one-step TD, $\lambda = 1$ recovers Monte Carlo. Eligibility traces implement this without storing the whole episode.

The practical reading is that **the bias–variance trade is a dial, not a binary**, and most applications sit somewhere in between rather than at either end.

## The Vocabulary Travels Further Than the Method

Any framework computing $Q(S_i) - Q(S_{i-1})$ is using the *shape* of a TD residual with no reward term. That resemblance is shallow: those frameworks typically fit $Q$ by supervised learning against an outcome label and then difference it.

**Differencing a supervised model is not bootstrapping**, because nothing in training ever used one estimate as another's target. The distinction is worth holding onto, since the borrowed vocabulary makes far more work look like RL than actually is.^[generated]

## See Also

- [[reinforcement-learning]] · [[markov-game]] · [[value-iteration]] · [[deep-q-network]] · [[proximal-policy-optimization]] · [[policy-modelling]]
- [[gated-recurrent-unit]] · [[lstm]] · [[regularization]] · [[multi-agent-reinforcement-learning]]
