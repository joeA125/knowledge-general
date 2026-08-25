---
title: "Bradley-Terry Model"
type: concept
tags: [paired-comparison, ranking-system, statistics, probabilistic-classification, evaluation, bayesian]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.7
provenance:
  extracted: 15%
  inferred: 28%
  generated: 8%
  imported: 47%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Bradley-Terry Model

A statistical model for **pairwise contest outcomes**, from which a latent strength per competitor is inferred. Given that $i$ and $j$ compete:

$$P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j}$$

Equivalently, on a log scale, the outcome is a logistic function of the strength *difference* — which is why fitting Bradley-Terry is logistic regression with one indicator per competitor.

## Why It Is Foundational Rather Than Applied

Most rating systems in use are Bradley-Terry with something added:

| | Adds |
|---|---|
| **Bradley-Terry** | — |
| [[elo-rating-system\|Elo]] | An online update rule instead of a batch fit |
| [[glicko-rating-system\|Glicko]] | Uncertainty on each strength |
| [[trueskill\|TrueSkill]] | Uncertainty, draws, teams, multiplayer |

Recognising this collapses a confusing family into one model plus a list of extensions, and makes the extensions comparable: each addresses a specific thing the base model cannot express.

## The Thurstone Alternative

^[imported]

Thurstone's model makes the same move with a **Gaussian** rather than logistic link — performance is normally distributed around latent skill, and the winner is whoever performs higher.

The two are close numerically and differ in tractability. Logistic gives a closed-form likelihood; Gaussian does not, once draws or teams are involved, which is why [[trueskill|TrueSkill]] needs approximate inference where Elo needs none.

**The choice of link function is where the computational cost is decided**, not where the modelling is.

## What It Cannot Express

- **Intransitivity.** A single strength per competitor forces transitive preference: if $A$ beats $B$ and $B$ beats $C$, then $A$ is rated above $C$. Genuine rock-paper-scissors structure is inexpressible.
- **Context.** Home advantage, surface, format — all require extra terms.
- **Change over time.** The base model treats all games as exchangeable.

The first is the interesting limitation. **Where intransitivity is real, the model does not fail loudly — it produces a plausible ranking that averages over the cycle**, and nothing in the fit reveals that the ordering it reports does not exist.^[generated]

## Identifiability

Strengths are determined only up to a common scale factor, so a constraint is needed — fix one competitor at zero, or fix the sum. This is why rating scales are arbitrary and why cross-system comparisons are meaningless without a shared anchor.

## See Also

- [[elo-rating-system]] · [[glicko-rating-system]] · [[trueskill]] · [[probabilistic-classification]] · [[model-selection]]
- [[bayesian-inference]] · [[uncertainty-quantification]] · [[predictive-validity]]
- [[bayesian-true-skill-rating|TrueSkill Summary]]
