---
title: "Game Theory"
type: concept
tags: [game-theory, multi-agent, markov-model, reinforcement-learning, statistics, evaluation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 26%
  generated: 8%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Game Theory

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

The study of strategic interaction between agents whose outcomes depend on each other's choices. A game is specified by its **players**, their **strategies**, and the **payoffs** each strategy profile produces.

## Why It Is Not Just Optimisation

The defining difference from single-agent decision theory: **there is no fixed environment to optimise against.**

An agent's best action depends on what others do, and their best actions depend on it. That circularity means "optimal" is not well defined for one agent in isolation — the solution concept has to describe a *joint* configuration.

**Nash equilibrium** is the standard one: a strategy profile where no player can improve by unilaterally deviating. Every finite game has at least one, possibly in mixed (randomised) strategies.

## What Equilibrium Assumes

Four assumptions that are rarely all true, and which determine how much an equilibrium prediction is worth:

- **Rationality.** Every player maximises their own payoff.
- **Common knowledge of rationality.** Everyone knows everyone is rational, and knows that they know.
- **Correct payoffs.** The modeller has specified what players actually care about.
- **Equilibrium selection.** Where several equilibria exist, something picks one — and the theory does not say what.

> ### `an-equilibrium-prediction-is-conditional-on-the-opponent-model`
> **Computing what a rational opponent would do gives the right answer only if the opponent is rational in the modelled sense. Where opponents systematically deviate, the equilibrium prescription is wrong in a direction the model cannot detect — and observed behaviour departing from equilibrium is evidence about the assumption as much as about the players.**
> ^[generated. rests-on: imported:nash-equilibrium-assumptions]

That claim matters wherever equilibrium analysis is used prescriptively: telling an agent to play the equilibrium strategy is only good advice against opponents who are also playing it.

## The Tractability Constraint

Equilibrium computation requires **payoffs for every strategy profile**, including profiles never observed. That is what makes the approach powerful and what limits it.

| Strategy space | Equilibrium analysis |
|---|---|
| A handful of discrete options | **Tractable, and the payoff table is the explanation** |
| Large or continuous | Generally intractable without strong structure |

The practical move is therefore to **coarsen until the space is enumerable** — which buys tractability and means the equilibrium concerns the coarsened game rather than the real one. Whether that coarsening preserves the strategic structure is a separate question, usually unasked.

## Relation to Multi-Agent RL

The two approaches to the same problem, and they are complementary in an awkward way:

| | [[game-theory\|Game theory]] | [[multi-agent-reinforcement-learning\|MARL]] |
|---|---|---|
| Other agents | **Modelled as reasoning** | Often folded into the environment |
| Solution | **Equilibrium** | Learned policies |
| Needs | Payoffs for all profiles | Samples |
| Scales to many agents | Poorly | **Yes** |
| Explains *why* | **By construction** — the payoff table is the argument | Poorly, without further analysis |

**Independent MARL learners do not model each other at all**, which means a system described as multi-agent may model strategic interaction *less* than a two-player game-theoretic treatment does. See [[markov-game]], where the Markov-game formalism is the bridge between them.

## Beyond Equilibrium

^[imported]

Several refinements address the assumptions above: **correlated equilibrium** allows a shared signal; **evolutionary game theory** replaces rational choice with selection over strategies; **quantal response equilibrium** assumes noisy rather than perfect best-response, which fits observed behaviour better.

The last is the most useful correction in practice, precisely because it weakens the assumption that most often fails.

## See Also

- [[markov-game]] · [[multi-agent-reinforcement-learning]] · [[reinforcement-learning]] · [[policy-modelling]] · [[agent-based-simulation]]
- [[bradley-terry-model]] · [[model-selection]] · [[counterfactual-baseline]] · [[imitation-learning]]
