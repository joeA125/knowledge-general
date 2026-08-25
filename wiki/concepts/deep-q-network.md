---
title: "Deep Q-Network"
type: concept
tags: [reinforcement-learning, temporal-difference, deep-learning, experience-replay, training-technique, machine-learning, markov-model]
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

# Deep Q-Network

> ⚠️ **No held source.** Background knowledge, marked `imported:` throughout. Mnih et al. (2015) is the primary source and would ground this page directly.

Approximating the action-value function $Q(s,a;\theta)$ with a neural network emitting one value per action. The network is trained on the [[temporal-difference-learning|TD]] residual — but doing that naively diverges, and **most of what "DQN" names is the machinery that stops it.**

## The Deadly Triad

^[imported: the term and the analysis are standard]

Three ingredients, each individually fine, that together destabilise value learning:

1. **Function approximation** — the value function is parameterised, so updating one state's estimate perturbs others
2. **Bootstrapping** — targets depend on the current estimate
3. **Off-policy data** — the behaviour distribution differs from the policy being evaluated

DQN uses all three. The stabilisers exist to make that survivable rather than to remove the tension.

## The Three Stabilisers

| Device | Problem addressed |
|---|---|
| **Target network** $\theta'$ | The bootstrap target moves with every update, since one parameter set produces both $Q(s_t,a_t)$ and $Q(s_{t+1},\cdot)$. Freezing a copy every $\tau$ steps holds it still |
| **Replay buffer** | Consecutive transitions are strongly correlated, violating the i.i.d. assumption gradient descent expects. Uniform resampling decorrelates them |
| **Double Q-learning** | A max over noisy estimates is biased upward. DDQN *selects* with the online network and *evaluates* with the target network |

The DDQN loss:

$$J(Q) = \sum_t \big( r_t + \gamma\, Q(s_{t+1}, a^{\max}_{t+1}; \theta') - Q(s_t, a_t; \theta) \big)^2, \qquad a^{\max}_{t+1} = \arg\max_a Q(s_{t+1}, a; \theta)$$

**Separating selection from evaluation is the entire content of the double-Q correction**, and it is a one-line change that removes a systematic bias.

**Prioritised replay** refines the buffer further, sampling high-TD-error transitions more often — on the reasoning that transitions the model is most wrong about carry the most information.

## The Stabilisers Are Not Free-Standing

> ### `stabilisers-track-the-feedback-loop`
> **The DQN stabiliser set is required in proportion to how much a learning agent influences its own training distribution. Purely offline value estimation from fixed logged data needs little of it, and adopting the full apparatus regardless imports machinery for a problem the setting does not have.**
> ^[generated: drawn from what each stabiliser addresses. rests-on: imported:dqn-stabiliser-rationale]

An agent that never acts — estimating values from a fixed dataset — has no feedback loop. Its data distribution is static, transitions arrive in their real order, and there is nothing to decorrelate. A replay buffer over a fixed dataset is resampling, not decorrelation.

The corollary is that **the stabilisers are hyperparameters**, and consequential ones: target update period $\tau$, buffer size, priority exponent. They are rarely swept and rarely reported.

## Learning from Demonstrations

^[imported: DQfD, Hester et al. 2018]

**Deep Q-learning from Demonstrations** extends DQN with expert data, in three phases: pre-train on demonstrations, sample actions in the environment, then train with both losses running. Demonstration transitions are never evicted from the replay buffer.

Its supervision is a **large-margin classification loss** forcing the expert action's value above all others by at least a margin $l$:

$$J_{MS}(Q) = \sum_t \max_{a}\big[Q(s_t,a) + l(a^E_t, a)\big] - Q(s_t, a^E_t)$$

An alternative is cross-entropy on the softmax of $Q$, treating the value function as an implicit policy. The margin formulation is the more conservative — it constrains only the *ordering* of values, not their magnitudes — and is reported to struggle where demonstration data is scarce.^[imported]

See [[imitation-learning]] for the broader family this belongs to.

## Limitations

- **Discrete action spaces only.** DQN requires a max over actions, so continuous control needs a different family — see [[proximal-policy-optimization|policy gradient]].
- **Overestimation persists** even with double Q, just less severely.
- **Sample-hungry**, though replay makes it far less so than on-policy methods.
- **No convergence guarantee** under function approximation. The stabilisers are empirical fixes, not proofs.

## See Also

- [[temporal-difference-learning]] · [[reinforcement-learning]] · [[value-iteration]] · [[markov-game]] · [[proximal-policy-optimization]] · [[policy-modelling]]
- [[imitation-learning]] · [[multi-agent-reinforcement-learning]] · [[gated-recurrent-unit]] · [[regularization]] · [[adam-optimizer]]
