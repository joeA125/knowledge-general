---
title: "Glicko Rating System"
type: concept
tags: [ranking-system, bayesian, statistics, uncertainty-quantification, paired-comparison, matchmaking, volatility]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.7
provenance:
  extracted: 18%
  inferred: 28%
  generated: 6%
  imported: 46%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Glicko Rating System

Mark Glickman's Bayesian generalisation of [[elo-rating-system|Elo]]. A competitor's skill is a **Gaussian belief** $\mathcal{N}(\mu, \sigma^2)$ rather than a point, and the rating deviation $\sigma$ does real work.

Glicko-2 adds a third quantity, **volatility**, capturing how erratically a competitor's true skill fluctuates — distinct from how uncertain we are about it.

## What the Uncertainty Buys

Three things, each of which Elo cannot do:

**The update size becomes automatic.** A player with wide uncertainty moves a lot; a well-measured one barely moves. Elo's global $K$ constant is replaced by something derived from what is already known — which is exactly the gap recorded as `a-point-estimate-forces-a-global-learning-rate` on [[elo-rating-system]].

**Uncertainty grows during inactivity.** A competitor absent for a year is genuinely less well known, and Glicko widens $\sigma$ over time to reflect it. Elo has no mechanism for this; a rating from 2019 and one from last week are treated identically.

**Conservative display becomes possible.** Reporting $\mu - k\sigma$ rather than $\mu$ means a leaderboard shows only competitors who are both strong *and* well-measured. See [[trueskill]], which uses $\mu - 3\sigma$.

## Volatility Is Not Uncertainty

^[imported: the Glicko-2 addition]

The distinction Glicko-2 introduces, and it is easy to blur:

| | Means | Falls with |
|---|---|---|
| **$\sigma$ — rating deviation** | How little we know | More observations |
| **Volatility** | How much the true value moves | Nothing — it is a property of the competitor |

A consistent competitor observed rarely has high $\sigma$ and low volatility. An erratic one observed constantly has the reverse. **Treating them as one quantity means a player who is genuinely inconsistent looks like a player we have not measured enough**, and more data will not resolve it.

> ### `measurement-noise-and-real-variation-need-separate-parameters`
> **A single uncertainty term conflates not knowing a value with the value moving. The two imply opposite responses — gather more data, or stop expecting stability — and no amount of data distinguishes them without a parameter for each.**
> ^[generated. rests-on: imported:glicko2-volatility]

The same conflation appears in measurement theory as the reliability-versus-real-change problem. See [[split-half-reliability]].

## The Computed-Then-Discarded Problem

Glicko produces a distribution and most downstream uses take the mean.

That is a real loss. A competitor rated 1700 on thin evidence and one rated 1700 on hundreds of games are different inputs to any decision, and passing a point estimate onward erases the distinction the system exists to maintain.

**The uncertainty is usually computed correctly and then thrown away at the interface**, which is a recurring pattern rather than a Glicko-specific flaw. See [[uncertainty-quantification]].

## Where It Sits

| | Uncertainty | Draws | Teams | Multiplayer |
|---|---|---|---|---|
| [[elo-rating-system\|Elo]] | No | No | No | No |
| **Glicko / Glicko-2** | **Yes** | No | No | No |
| [[trueskill\|TrueSkill]] | **Yes** | **Yes** | **Yes** | **Yes** |

TrueSkill's extensions require approximate inference — [[expectation-propagation]] on a graphical model — where Glicko's updates remain close to closed form. **That is the price of the last three columns.**

## See Also

- [[elo-rating-system]] · [[trueskill]] · [[bradley-terry-model]] · [[bayesian-inference]] · [[expectation-propagation]]
- [[uncertainty-quantification]] · [[split-half-reliability]] · [[predictive-validity]] · [[probabilistic-classification]]
- [[bayesian-true-skill-rating|TrueSkill Summary]]
