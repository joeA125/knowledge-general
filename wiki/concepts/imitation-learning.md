---
title: "Imitation Learning"
type: concept
tags: [imitation-learning, machine-learning, reinforcement-learning, policy-modelling, teacher-forcing, alignment, auxiliary-loss]
sources: [raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.7
provenance:
  extracted: 10%
  inferred: 20%
  generated: 6%
  imported: 62%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Imitation Learning

Learning a policy by **mimicking observed behaviour** rather than by optimising a reward. Given demonstrations $\{(s_t, a_t)\}$, learn $\pi(a \mid s)$ reproducing them.

> ⚠️ **62% `imported:`.** The only held source touching this is [[training-lm-follow-instructions-with-human-feedback|InstructGPT]], whose supervised fine-tuning stage is behavioural cloning. The imitation-learning literature proper is not held.

## Why Imitate Rather Than Optimise

^[imported: the standard motivation]

[[reinforcement-learning|RL]] needs a reward signal and an environment to interact with. Two conditions make demonstrations the better route:

1. **The reward is hard to specify.** Where "good behaviour" is easier to recognise than to write down, demonstrations encode it implicitly.
2. **Interaction is expensive or unsafe.** Driving, clinical decisions, anything with real consequences.

Where both hold, imitation sidesteps the two hardest parts of RL at once. Where neither holds, it is a weaker method — it can never exceed the demonstrator.

## Three Roles, Not One

The same technique serves three genuinely different purposes, and conflating them causes confusion.

**1. As a policy.** The imitator's output *is* the deliverable — behavioural cloning in the plain sense.

**2. As a measuring instrument.** Train a model of *typical* behaviour, then measure how far a specific agent deviates from it. The imitator becomes a reference rather than a product.

Two consequences follow, both awkward:

- **The objective changes.** A forecaster wants minimal error; a reference wants a well-calibrated notion of *normal*. These are not the same target, and optimising the first can degrade the second.
- **Perfection destroys the measurement.** A metric defined as deviation-from-predicted is identically zero under a perfect predictor.

**3. As a prior on competence.** An imitation term added as an auxiliary loss to a value objective — not to deploy a policy, but to constrain a value function where data is absent. Most state-action pairs are never visited, so their values are otherwise shaped only by network smoothness.

The assumption imported is that **demonstrators act better than random.** That is not derivable from the data.

> ### `imitation-weight-tunes-the-conclusion`
> **Where an imitation term regularises a value function, its weight controls how much apparent suboptimality survives into the results. Turn it up and the demonstrator looks wise; turn it down and they look arbitrary. The weight is a free parameter and is rarely reported.**
> ^[generated: drawn from what role 3 does. rests-on: imported:auxiliary-imitation-loss]

## The Compounding-Error Problem

^[imported: DAgger and the covariate-shift literature]

Behavioural cloning is formally supervised learning. What distinguishes imitation learning as a field is the **sequential** setting: the learner's own actions determine the states it later sees.

A small error moves the agent slightly off the demonstrated distribution, where it has less training signal, so the next error is larger. Over a long rollout this compounds — the same failure that [[teacher-forcing|exposure bias]] describes in autoregressive generation, arriving from the control side.

Standard mitigations are querying the expert on states the learner actually visits, and keeping rollouts short.

## Relation to Neighbouring Regimes

| | Learns from | Optimises |
|---|---|---|
| **Imitation learning** | Demonstrations | Match to observed behaviour |
| [[reinforcement-learning]] | Reward signal | Expected return |
| Inverse RL | Demonstrations | A **reward** explaining them |
| **Imitation as auxiliary loss** | **Both** | A weighted sum |
| Supervised learning | Labels | Match to labels |

[[rlhf|RLHF]] spans two rows: its supervised stage is behavioural cloning, its reward model is inverse-RL-flavoured, and its optimisation stage is RL — with a [[kl-divergence|KL]] penalty that is itself an imitation term anchoring the policy to where it started.

**That KL coefficient is the best-documented instance of the weight in the claim above**, because RLHF reports and sweeps it as standard. See [[proximal-policy-optimization]].

## Caveats

- **Demonstrations encode constraint, not only skill.** A demonstrator may act as they do because of instruction, fatigue, or role. An imitator learns all of it undifferentiated.
- **"Average behaviour" is a population, not a person** — an abstraction no individual instantiates.
- **In role 3, the caveat becomes the conclusion.** If demonstrations encode constraint rather than judgement, a value function regularised toward them is regularised toward constraint.

## See Also

- [[reinforcement-learning]] · [[policy-modelling]] · [[proximal-policy-optimization]] · [[rlhf]] · [[deep-q-network]] · [[markov-game]]
- [[teacher-forcing]] · [[kl-divergence]] · [[generative-model]] · [[multi-agent-reinforcement-learning]] · [[domain-adaptation]]
- [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
