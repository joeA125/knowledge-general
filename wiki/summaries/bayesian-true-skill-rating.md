---
title: "TrueSkill: A Bayesian Skill Rating System — Source Summary"
type: summary
tags: [bayesian, statistics, ranking-system, factor-graph, message-passing, matchmaking, gaming]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# TrueSkill: A Bayesian Skill Rating System

**Authors:** [[ralf-herbrich]], [[tom-minka]], [[thore-graepel]]
**Affiliation:** [[microsoft-research]]
**Published:** 2006 (NeurIPS)

## Key Contribution

This paper introduces [[trueskill]], a Bayesian skill rating system that generalises the [[elo-rating-system]]. TrueSkill tracks uncertainty about player skills as Gaussian distributions, explicitly models draws, handles any number of competing entities, and infers individual skills from team results. Inference is performed via [[approximate-message-passing]] on a [[factor-graph]] representation of the model.

## Model

- Each player $i$ has a skill $s_i$ modelled as a Gaussian $\mathcal{N}(\mu_i, \sigma_i^2)$.
- In a game, each player exhibits a performance $p_i \sim \mathcal{N}(s_i, \beta^2)$ drawn around their skill.
- Team performance is the sum of individual performances: $t_j = \sum_{i \in A_j} p_i$.
- Game outcomes are modelled as the ordering of team performances, with an explicit draw margin $\epsilon$.
- The posterior over skills given outcomes is computed via [[bayes-theorem]]: $p(\mathbf{s}|\mathbf{r}, A) \propto P(\mathbf{r}|\mathbf{s}, A) p(\mathbf{s})$.

## Inference

The model is represented as a [[factor-graph]] with four variable types: skills ($s_i$), performances ($p_i$), team performances ($t_j$), and team performance differences ($d_j$). Inference uses [[approximate-message-passing]] based on [[expectation-propagation]]:

- Messages from comparison factors are non-Gaussian, so marginals are approximated via moment matching (minimising KL divergence).
- The factor graph is acyclic; messages are iterated between approximate marginals until convergence.
- The algorithm is extremely efficient — within twice the runtime of a simple Elo update.

## Relation to Prior Work

- **[[elo-rating-system]]:** TrueSkill generalises Elo by tracking uncertainty ($\sigma$), modelling draws, handling teams, and handling multi-player games.
- **[[glicko-rating-system]]:** Glickman's Glicko introduced Gaussian belief over skill. TrueSkill extends this to teams and multi-party games.
- **Thurstone Case V / Bradley-Terry models:** The Gaussian and logistic variants of paired comparison models that underpin Elo.

## Experimental Results (Halo 2 Beta)

Tested on game outcome data from Bungie Studios' Halo 2 beta testing across four game modes (Free for All, Small Teams, Head to Head, Large Teams):

- TrueSkill outperformed Elo in prediction accuracy for most game modes.
- Significantly better at identifying tight matches (the "challenge" test).
- Better match quality: higher concentration of draws among top-ranked matches.
- Fairer win probabilities, especially for players with few games.
- Faster convergence: TrueSkill approaches target skill in ~10 games (near information-theoretic limit of ~5 for 8-player games), while Elo requires hundreds.

## Deployment: Xbox Live

TrueSkill was deployed on Xbox 360 Live (2+ million subscribers, hundreds of thousands of games/day), making it one of the largest applications of [[bayesian-inference]] at the time.

Key parameters: prior $\mu_0 = 25$, $\sigma_0 = 25/3$, $\beta = \sigma_0/2$, $\gamma = \sigma_0/100$. Player skill displayed as conservative estimate $\mu_i - 3\sigma_i$.

Matchmaking uses a draw probability quality criterion that aligns finding fair matches with finding the most informative matches (sequential experimental design).

## Observations from Deployment

1. Games differ in effective skill levels (chance vs. skill).
2. Matchmaking creates feedback loops — players game the system to protect ratings.
3. Skill distribution shifts below prior if new players consistently lose early; tighter matchmaking fixes this.

## See Also

- [[trueskill]]
- [[elo-rating-system]]
- [[factor-graph]]
- [[approximate-message-passing]]
- [[expectation-propagation]]
- [[bayesian-inference]]
