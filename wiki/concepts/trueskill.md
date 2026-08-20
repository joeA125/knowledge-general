---
title: "TrueSkill"
type: concept
tags: [bayesian, ranking-system, matchmaking, gaming, statistics, reinforcement-learning, evaluation, simulator, agent-based-simulation]
sources: [raw/papers/bayesian-true-skill-rating.md, raw/papers/ai_football_reinforcement_learning.md]
confidence: 0.9
provenance:
  extracted: 78%
  inferred: 15%
  generated: 4%
  imported: 0%
  ambiguous: 3%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-08-08
---

# TrueSkill

TrueSkill is a Bayesian skill rating system developed by [[microsoft-research]] (Herbrich, Minka & Graepel, 2006) that generalises the [[elo-rating-system]]. It is used for player matchmaking in online gaming, most notably on Xbox Live.

## Key Properties

- **Uncertainty tracking:** Each player's skill is a Gaussian belief $\mathcal{N}(\mu, \sigma^2)$, not a point estimate.
- **Draw modelling:** Explicit draw margin $\epsilon$ models the probability of tied outcomes.
- **Team support:** Individual skills are inferred from team results by modelling team performance as the sum of individual performances.
- **Multi-player support:** Handles arbitrary numbers of competing players/teams via pairwise team performance difference comparisons.
- **Fast convergence:** Approaches target skill in ~10 games for 8-player matches, near the information-theoretic limit.

## How It Works

1. Model the game as a [[factor-graph]] with variables for skills, performances, team performances, and performance differences.
2. Run [[approximate-message-passing]] (based on [[expectation-propagation]]) to compute posterior skill distributions.
3. Use [[gaussian-density-filtering]]: the posterior after each game becomes the prior for the next.

## Skill Display

The displayed rating is a conservative estimate: $\mu - 3\sigma$. This ensures leaderboard tops are populated only by players who are both highly skilled and well-measured.

## Matchmaking

Match quality is derived from draw probability relative to the maximum possible draw probability, aligning fair matchmaking with informative experimental design.

## Rating Agents Rather Than People

> **Added 2026-08-08** on ingest of [[ai-football-reinforcement-learning|Scott, Fujii & Onishi]], the vault's first application of TrueSkill to something other than human competitors.

They use TrueSkill to rank **15 RL agents** — five training checkpoints (20%, 40%, 60%, 80%, 100%) from each of three curricula (easy, medium, hard built-in opponents) — via **50 round-robin tournaments, 5,250 matches**, in the [[google-research-football|GFootball]] simulator.

The fit is unusually good, and for a reason the original paper could not have anticipated:

| TrueSkill assumes | Human leagues | **RL checkpoints** |
|---|---|---|
| Skill is latent and unobservable | Yes | Yes |
| Matches are expensive to run | Yes | **No — arbitrarily many** |
| Skill drifts over time | Yes | **No — a checkpoint is frozen** |
| Competitors are numerous | Yes | 15 |

**The uncertainty machinery that justifies TrueSkill for matchmaking is largely wasted here.** With frozen agents and 5,250 cheap matches, $\sigma$ collapses and the ranking is essentially a well-regularised round-robin win rate. What TrueSkill supplies is a *principled single number* on a common scale — which is what the paper needs, since its entire analysis is correlating play-style statistics against that number.

> ### `latent-skill-models-suit-frozen-agents-better-than-people`
> **Skill-rating systems built for humans transfer unusually cleanly to model checkpoints, because checkpoints satisfy the stationarity assumption that humans violate. The cost is that the uncertainty apparatus, which is the reason to prefer such systems over win rates, becomes redundant once matches are cheap.**
> ^[generated: no source states this; drawn from the assumption set of the original paper read against Scott et al.'s use. rests-on: source:herbrich-trueskill-assumptions, source:scott-trueskill-tournament]

## The Anomaly That Followed

Scott et al.'s ranking produced a result they call counter-intuitive and do not resolve: **agents trained against the *easy* bot rank 1, 2 and 3**, above agents trained four times longer against the hard bot. They suggest better training strategies exist and defer the question.

This matters beyond their paper because **every correlation they report is measured against this axis** — including the finding that better agents shoot more, which contradicts [[optimal-decisions-shot-taking-situations|Yeung & Fujii]] on real football. See [[observed-versus-optimal-decisions]].

Two readings, undistinguished by anything held:

- **The ranking is right and the curriculum intuition is wrong** — training against weak opponents may produce more transferable play than over-fitting to one strong opponent's exploitable habits.
- **The ranking is measuring something narrower than competence** — round-robin performance against *this particular pool of 15* rather than football ability.

The second is the sort of thing TrueSkill's own framing warns about: a rating is defined relative to the population it was estimated in. **A leaderboard of fifteen agents that only ever played each other is a closed system**, and nothing anchors it to external football quality.

## See Also

- [[elo-rating-system]] · [[glicko-rating-system]] · [[bradley-terry-model]] · [[league-strength-rating]]
- [[factor-graph]] · [[approximate-message-passing]] · [[expectation-propagation]] · [[gaussian-density-filtering]] · [[bayesian-inference]]
- [[agent-based-simulation]] · [[google-research-football]] · [[proximal-policy-optimization]] · [[social-network-analysis]] · [[reinforcement-learning]] · [[observed-versus-optimal-decisions]]
- [[ralf-herbrich]] · [[tom-minka]] · [[thore-graepel]] · [[atom-scott]] · [[microsoft-research]]
- [[bayesian-true-skill-rating|TrueSkill Summary]] · [[ai-football-reinforcement-learning|Scott et al. Summary]]
