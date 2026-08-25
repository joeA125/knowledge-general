---
title: "Multi-Agent Reinforcement Learning"
type: concept
tags: [multi-agent, reinforcement-learning, markov-model, game-theory, machine-learning, simulator]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 20%
  generated: 8%
  imported: 70%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Multi-Agent Reinforcement Learning

> ⚠️ **No held source.** Entirely background knowledge, marked `imported:`. This is among the thinnest pages in the vault.

[[reinforcement-learning|RL]] where several decision-makers act in a shared environment, each with its own policy — as opposed to folding the others into the environment dynamics.

## The Problem the Single-Agent Formalism Hides

In single-agent RL the environment is **stationary**: $T(s' \mid s, a)$ does not change. With several learners it is not — each agent's environment includes the others, and they are all changing as they learn.

That breaks the fixed-point arguments underpinning convergence. The target an agent is chasing moves because the chase itself moves it.

Three consequences follow:

| | Single-agent | Multi-agent |
|---|---|---|
| Environment | Stationary | **Non-stationary by construction** |
| Solution concept | Optimal policy | **Equilibrium** — see [[markov-game]] |
| Convergence | Guaranteed in the tabular case | No general guarantee |

## Three Positions on the Spectrum

^[imported: this taxonomy is standard]

| | Controls | Others are | Can attribute per-agent |
|---|---|---|---|
| **Team as one agent** | A joint policy | Absorbed into the abstraction | **No** |
| **Central control, one active** | One agent at a time | Scripted or fixed | **No** |
| **Per-agent policies** | Every agent | Independent learners or coordinated | **Yes** |

The middle position is easy to overlook and common in practice — it permits acting at every timestep while sidestepping non-stationarity, at the cost of the other agents being a fixed backdrop rather than participants.

## "Independent" Is Doing a Lot of Work

The cheapest multi-agent setup gives each agent its own value function and lets it treat the others as environment. Teammates *appear* in the state vector; no agent reasons about what a teammate will do.

| | Independent | Interactive / centralised |
|---|---|---|
| Each agent models others as | Part of the environment | Agents with policies |
| Non-stationarity | Unhandled | Addressed explicitly |
| Solution concept | Separate value functions | Joint policy or equilibrium |

> ### `agent-count-is-not-interaction`
> **Instantiating one value function per agent makes a system multi-agent in the count of learners, not in the modelling of interaction. Independent learners in a shared state space are closer to several single-agent problems with correlated inputs than to a genuine multi-agent solution.**
> ^[generated: drawn from what the independence assumption removes. rests-on: imported:independent-marl-formulation]

The consequence bites hardest on counterfactuals. Asking "what if this agent had acted differently?" under independence answers it **holding every other agent's behaviour fixed** — which is exactly what would not happen.

## Centralised Training, Decentralised Execution

^[imported: CTDE is the dominant practical compromise]

Train with access to the joint state and all agents' actions; execute using only each agent's local observation. This addresses non-stationarity during learning while keeping deployment realistic.

Worth noting that **centralised training is not automatically better.** Where the binding constraint is elsewhere — environment fidelity, reward specification, data scale — moving from decentralised to centralised learning may change nothing, and the comparison is worth running before assuming.

## Where It Is Used

Any domain where logged behaviour comes from a group credited jointly: trading desks, clinical teams, vehicle fleets, collective animal behaviour, team sports.

The recurring trap is the first row of the table above. **An aggregate reward invites an aggregate agent, and an aggregate agent cannot attribute.** Decomposing into per-agent policies restores attribution at the price of an independence assumption that is usually false — and the choice between those two failures is rarely stated as a choice.

## See Also

- [[reinforcement-learning]] · [[markov-game]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[proximal-policy-optimization]] · [[policy-modelling]]
- [[imitation-learning]] · [[agent-based-simulation]] · [[domain-adaptation]] · [[message-passing]] · [[trajectory-prediction]]
