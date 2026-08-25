---
title: "Agent-Based Simulation"
type: concept
tags: [agent-based-simulation, simulator, multi-agent, reinforcement-learning, domain-adaptation, counterfactual, evaluation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 22%
  generated: 10%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Agent-Based Simulation

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Simulating a system by giving individual agents rules or policies and letting collective behaviour **emerge**, rather than modelling the aggregate directly.

## Why Simulate at All

The argument is about **experimental control**, not compute.

Many systems generate enormous observational data and permit no experiments. Nobody randomises a city's traffic policy, a market's structure, or a population's contact patterns to see what happens. Where the question is causal and intervention is impossible, simulation is the only route to a controlled manipulation.

> ### `observational-abundance-does-not-substitute-for-control`
> **A field can have exponentially growing data and remain unable to answer causal questions, because volume addresses estimation error and not confounding.**
> ^[generated: the motivation is standard; this compressed form is drawn here. rests-on: imported:abs-experimental-motivation]

That claim connects directly to [[selection-bias]] and to the causal caveats on [[counterfactual-simulation]]: a generative model fitted to observational data learns the observational distribution, and intervening on it gives the right causal answer only under assumptions nobody checks. **ABS's promise is that the intervention is real, inside the simulator.** Its problem is that the simulator may not be the system.

## The Validation Trap

This is the structural difficulty and it is not incidental.

**ABS is adopted where experiments are impossible — which is exactly where the simulator cannot be validated against experiment.** The condition that makes ABS necessary is the condition that makes it unverifiable.

The partial answer, common across every field that uses it: compare simulated behaviour against real *observational* data on some dimension, and argue about whether the dimension is the right one.

> ### `validation-dimension-is-usually-chosen-for-measurability`
> **Where a simulator is validated on some dimension, that dimension is typically selected because it is measurable across both simulated and real settings — which is close to selecting it because the domain gap does not affect it. A transfer result is a finding about the chosen dimension, not about the domains.**
> ^[generated. rests-on: imported:abs-validation-practice]

See [[domain-adaptation]], where the same problem appears from the transfer side.

## Emergence Is the Point and the Problem

ABS is used to observe behaviour that was not programmed — coordination, segregation, congestion, collective motion arising from local rules.

The interpretive difficulty is symmetrical:

- **When emergent behaviour resembles reality**, it is evidence the simulation captures something.
- **When it does not**, the fault may lie with the rules, the parameters, the initialisation, or the question — and nothing in the output distinguishes them.

A simulator that reproduces a stylised fact has not thereby explained it, since many rule sets can produce the same aggregate pattern. **Reproduction is weak evidence for mechanism**, which is the standard critique of the method and the reason ABS results are usually reported as existence proofs rather than explanations.^[imported]

## Relation to Neighbouring Approaches

| | Simulates | Assumes |
|---|---|---|
| **Agent-based simulation** | Individual agents with rules | That emergence is informative |
| Equation-based modelling | The aggregate directly | That aggregate dynamics are closed |
| [[multi-agent-reinforcement-learning\|MARL]] | Agents that **learn** | That reward specifies the objective |
| [[counterfactual-simulation]] | An intervention on a fitted model | That the model captured the dependencies |

MARL is the case where the agents' rules are *learned rather than specified*, which removes one modelling burden and adds another — the reward function now carries the assumptions the rules used to.

## Where It Is Used

Epidemiology, traffic engineering, ecology, economics, crowd dynamics, and collective animal behaviour. The common thread is a system of many interacting units where aggregate behaviour is the object of interest and controlled experiment is unavailable.

## See Also

- [[multi-agent-reinforcement-learning]] · [[reinforcement-learning]] · [[markov-game]] · [[domain-adaptation]] · [[counterfactual-simulation]]
- [[selection-bias]] · [[generative-model]] · [[trajectory-prediction]] · [[proximal-policy-optimization]]
