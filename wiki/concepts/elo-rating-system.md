---
title: "Elo Rating System"
type: concept
tags: [ranking-system, paired-comparison, statistics, matchmaking, gaming, evaluation]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.75
provenance:
  extracted: 25%
  inferred: 25%
  generated: 6%
  imported: 42%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Elo Rating System

A skill rating system for two-player games, developed by Arpad Elo for chess. Each competitor holds a single number; after each game the winner takes points from the loser, scaled by how surprising the result was.

$$R'_A = R_A + K\,(S_A - E_A), \qquad E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

$E_A$ is the expected score from the rating difference; $S_A$ is the actual result. **The update is proportional to the surprise** — beating a stronger opponent moves the rating more.

## What It Assumes

^[imported: the standard formulation]

- Performance in a game is a **random variable around a latent skill**, and skill is what the rating estimates.
- The difference in performances determines the outcome probability, via a logistic function.
- Skill changes slowly relative to the rate of games.

The first two are the [[bradley-terry-model|Bradley-Terry]] model in disguise — Elo is essentially an online estimator for it, with the update rule standing in for a full re-fit after every game.

## The Three Limitations That Motivated Everything After

| Limitation | Consequence | Addressed by |
|---|---|---|
| **A point estimate, no uncertainty** | A new player and a veteran with the same rating are treated identically | [[glicko-rating-system\|Glicko]] |
| **No draws** | Modelled as half a win, which is not the same claim | [[trueskill\|TrueSkill]] |
| **Two players only** | Teams and multiplayer need a separate mechanism | TrueSkill |

The first is the consequential one. **$K$ is a single global constant controlling how fast every rating moves** — set it high and established ratings are noisy, set it low and new players take hundreds of games to reach their level. There is no value that serves both, because the right step size depends on how well the player is already known, and Elo has no representation of that.

That is precisely what a Bayesian treatment supplies: a belief with a variance, where the update size falls out of the uncertainty rather than being set by hand.

> ### `a-point-estimate-forces-a-global-learning-rate`
> **Any rating system tracking only a point estimate must choose one update magnitude for all competitors, because it has no per-competitor measure of how much it already knows. The uncertainty is not merely unreported — its absence changes the update rule.**
> ^[generated. rests-on: imported:elo-k-factor, source:trueskill-convergence-comparison]

## Why It Survives Anyway

Elo remains widely used despite all of this, and the reasons are worth stating rather than dismissing:

- **It is one line.** No inference machinery, no distributions, no libraries.
- **Ratings are interpretable and comparable** across time in a way posterior means are not.
- **It is good enough** where players are numerous and games are plentiful, since $\sigma$ shrinks with data anyway and the missing uncertainty matters least in the steady state.

The held TrueSkill paper measures the gap directly: Elo takes **hundreds of games** to converge where TrueSkill takes roughly ten in an eight-player match. That difference matters for matchmaking and matters little for a long-running league.

## See Also

- [[glicko-rating-system]] · [[trueskill]] · [[bradley-terry-model]] · [[bayesian-inference]] · [[uncertainty-quantification]]
- [[probabilistic-classification]] · [[predictive-validity]] · [[model-selection]]
- [[bayesian-true-skill-rating|TrueSkill Summary]]
