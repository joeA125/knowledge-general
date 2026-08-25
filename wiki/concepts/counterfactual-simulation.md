---
title: "Counterfactual Simulation"
type: concept
tags: [counterfactual, generative-model, machine-learning, simulator, agent-based-simulation, domain-adaptation, evaluation]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.65
provenance:
  extracted: 12%
  inferred: 26%
  generated: 10%
  imported: 50%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Counterfactual Simulation

Using a model to answer *"what would have happened if…?"* — generating outcomes under conditions that were never observed.

Distinct from prediction in that the conditioning is **hypothetical**. Not "what happens next given this", but "what would happen in a situation that did not occur".

## Three Routes, With Different Costs

| | **Re-scoring** | **Re-generation** | **Value-function readout** |
|---|---|---|---|
| Procedure | Hold the observed sequence fixed, substitute the entity, re-evaluate | Substitute, then generate a new sequence | Read $Q(s, a')$ for an action nobody took |
| Answers | How would this entity *value* these situations? | How would this entity *change what happens*? | How good would this *action* have been? |
| Generates | No | **Yes** | No |
| Compounding error | None | **Substantial over long rollouts** | None |
| Requires | A scoring model | A transition model good enough to roll out in | An enumerable action space |

The trade is legible: **re-generation buys a world that responds, at the price of accumulated error; readout buys zero error, at the price of a world that does not respond.**

Readout's cost is easy to miss because nothing visibly goes wrong. Off-policy action values look like estimates and are frequently assumptions — constrained by network smoothness and whatever prior was imposed, not by data. See [[reinforcement-learning]].

## Re-Generation Fails Where the Representation Is Fine

The observation worth carrying, and it cuts against intuition.

Re-generation over **coarse, discrete tokens** — events, actions, symbols — is comparatively robust: a generated token sequence cannot be *physically* impossible, because the representation cannot express physical impossibility.

Re-generation over **continuous, fine-grained state** — trajectories, dynamics — is far harder. There, generative error compounds into behaviour that is not merely unlikely but incoherent.

> ### `regeneration-fidelity-scales-with-representation-coarseness`
> **Re-generation counterfactuals succeed where the representation is coarse enough that the model's errors stay inside the discretisation, and fail where it is fine enough that they compound into physically wrong behaviour. A coarse representation is therefore not obviously a weakness — it may be what makes the counterfactual survivable.**
> ^[generated. rests-on: imported:generative-rollout-practice]

The uncomfortable corollary: **a counterfactual that works because the representation hides the physics is not obviously more trustworthy than one that fails visibly.** It may only be less falsifiable.

## What a Model Needs to Support It

1. **Generative**, for the second route — able to produce sequences, not only score them.
2. **A long enough horizon.** Fragment-level generation forces the remaining outcome to be approximated.
3. **Explicit, surgical conditioning.** The intervention must isolate the entity; if identity is entangled with everything else, the counterfactual is not isolated.
4. **A transition model that survives the rollout** — the requirement most often assumed rather than checked. See [[agent-based-simulation]].

## Estimation and Validation

Generation is stochastic, so **a single rollout is a sample, not an estimate.** Monte Carlo over many rollouts is required, and the variance is itself informative — the readout route produces a deterministic number with no uncertainty attached at all, which is a convenience and a warning. See [[uncertainty-quantification]].

Two validation modes, ascending in strength:

- **Self-to-self reconstruction** — simulate with the *actual* entity and compare against what really happened. Necessary, not sufficient, and it fails informatively when the simulated value exceeds ground truth.
- **Out-of-sample intervention** — simulate the entity into a genuinely new context and compare against what actually happened there. Much stronger, much rarer.

## The Causal Caveat

The vocabulary is borrowed from causal inference and the borrowing is loose.

A generative model trained on observational data learns the **observational** distribution. Intervening on it yields the correct causal answer only if the model captured the right dependency structure and nothing important is unmeasured — and neither condition is typically checked.

**"Generative" is not "causal"**, and treating them as equivalent is the most common error in this area. See [[selection-bias]] for the version of the problem that arises before the model is fitted.

## See Also

- [[generative-model]] · [[counterfactual-baseline]] · [[agent-based-simulation]] · [[domain-adaptation]] · [[selection-bias]]
- [[variational-autoencoder]] · [[autoregressive-model]] · [[reinforcement-learning]] · [[imitation-learning]] · [[uncertainty-quantification]]
- [[variational-lossy-autoencoders|VLAE Summary]]
