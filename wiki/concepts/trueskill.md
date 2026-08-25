---
title: "TrueSkill"
type: concept
tags: [bayesian, ranking-system, matchmaking, gaming, statistics, evaluation, uncertainty-quantification, message-passing]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.9
provenance:
  extracted: 74%
  inferred: 16%
  generated: 6%
  imported: 2%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-08-14
---

# TrueSkill

A Bayesian skill rating system developed by [[microsoft-research]] (Herbrich, Minka & Graepel, 2006), generalising the [[elo-rating-system|Elo]] system. Deployed for matchmaking on Xbox Live at 2+ million subscribers — one of the largest applications of [[bayesian-inference]] at the time.

## The Model

- Each player's skill is a Gaussian belief $s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, **not a point estimate**.
- In a game they exhibit a performance $p_i \sim \mathcal{N}(s_i, \beta^2)$ around that skill.
- **Team performance is the sum** of individual performances.
- Outcomes are the *ordering* of team performances, with an explicit draw margin $\epsilon$.

Inference runs as [[message-passing]] on a graphical model, with [[expectation-propagation]] approximating the non-Gaussian comparison factors by moment matching. Each game's posterior becomes the next game's prior.

Priors are $\mu_0 = 25$, $\sigma_0 = 25/3$; the displayed rating is the conservative $\mu_i - 3\sigma_i$, so leaderboard tops hold only players who are both strong **and** well-measured.

## What It Generalises

| | Uncertainty | Draws | Teams | Multi-player |
|---|---|---|---|---|
| [[elo-rating-system\|Elo]] | No | No | No | No |
| [[glicko-rating-system\|Glicko]] | **Yes** | No | No | No |
| **TrueSkill** | **Yes** | **Yes** | **Yes** | **Yes** |

Each added column costs tractability. Elo needs no inference at all; Glicko's updates stay near closed form; **TrueSkill needs approximate inference on a graphical model.** That is the price of the last three columns, and it is why the three systems coexist rather than one superseding the others.

## Convergence

On Halo 2 beta data, TrueSkill beat Elo on prediction accuracy across most modes and converged in **~10 games against Elo's hundreds** — near the information-theoretic limit of about 5 for an 8-player match.

The reason is the uncertainty: a new player's wide $\sigma$ produces large updates, and a well-measured player's narrow $\sigma$ produces small ones. Elo has no per-player measure of what it already knows, so it must apply one global step size to everyone. See [[elo-rating-system]].

## Rating Non-Human Competitors

The system's assumptions were written for people, and they hold **differently** for frozen model checkpoints:

| TrueSkill assumes | Human leagues | Model checkpoints |
|---|---|---|
| Skill is latent and unobservable | Yes | Yes |
| Matches are expensive | Yes | **No — arbitrarily many** |
| Skill drifts over time | Yes | **No — a checkpoint is frozen** |

> ### `latent-skill-models-suit-frozen-agents-better-than-people`
> **Skill-rating systems built for humans transfer unusually cleanly to model checkpoints, because a checkpoint satisfies the stationarity assumption that humans violate. The cost is that the uncertainty apparatus — the reason to prefer such systems over a win rate — becomes redundant once matches are cheap.**
> ^[generated: drawn from the assumption set of the held paper. rests-on: source:herbrich-trueskill-assumptions]

Where matches are cheap and competitors frozen, $\sigma$ collapses and the ranking approaches a well-regularised round-robin win rate. What survives is the useful part: **a principled single number on a common scale**, which is what any correlation analysis against a competitiveness axis requires.

⚠️ **A closed pool is a closed system.** A rating estimated from competitors who only ever played each other is defined relative to that pool, and nothing anchors it to external quality. TrueSkill's own framing warns about this and it is easily forgotten when the leaderboard looks decisive.

## Two Deployment Observations

**Matchmaking creates feedback loops.** Players game the system to protect ratings — declining unfavourable matches, playing at particular times. **Any rating fed back into the process that generates its own data changes the behaviour it measures.**

**The skill distribution shifts below the prior** if new entrants consistently lose early. Relevant wherever a population's composition changes over time, since the prior encodes an assumption about the incoming distribution that may stop holding.

## See Also

- [[elo-rating-system]] · [[glicko-rating-system]] · [[bradley-terry-model]] · [[bayesian-inference]] · [[expectation-propagation]] · [[message-passing]]
- [[uncertainty-quantification]] · [[probabilistic-classification]] · [[predictive-validity]] · [[model-selection]] · [[selection-bias]]
- [[microsoft-research]] · [[ralf-herbrich]] · [[tom-minka]] · [[thore-graepel]]
- [[bayesian-true-skill-rating|Source Summary]]
