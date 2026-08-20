---
title: "Non-negative Matrix Factorization (NMF)"
type: concept
tags: [linear-algebra, dimensionality-reduction, machine-learning, representation-learning, statistics]
sources: [raw/papers/multiresolution-stochastic-process-model-nba-possessions.md]
confidence: 0.85
provenance:
  extracted: 60%
  inferred: 32%
  ambiguous: 8%
lifecycle: reviewed
created: 2026-07-23
updated: 2026-07-23
---

# Non-negative Matrix Factorization (NMF)

NMF approximates a non-negative matrix $\mathbf{G}$ as a product of two low-rank non-negative factors:

$$\mathbf{G} \approx \mathbf{U}\mathbf{V}, \qquad \mathbf{U} \in \mathbb{R}_{\ge 0}^{L \times r}, \; \mathbf{V} \in \mathbb{R}_{\ge 0}^{r \times p}$$

found by minimising a divergence $D(\mathbf{G}, \mathbf{U}\mathbf{V})$ subject to $U_{ij}, V_{ij} \ge 0$. [[multiresolution-stochastic-process-nba-possessions|Cervone et al. (2016)]] use a Kullback–Leibler-type divergence:

$$D(\mathbf{G}, \mathbf{G}^*) = \sum_{i,j} G_{ij}\log(G_{ij}/G^*_{ij}) - G_{ij} + G^*_{ij}$$

## Why Non-negativity Matters

The non-negativity constraint is what distinguishes NMF from other factorisations and gives it its characteristic behaviour. Because components can only *add*, never cancel, NMF tends to learn **parts-based, additive representations**: the rows of $\mathbf{V}$ behave like interpretable building blocks, and the rows of $\mathbf{U}$ say how much of each block a given unit uses.

Contrast with [[eigenvector]]-based methods:

| | PCA / SVD | NMF |
|---|---|---|
| Constraint | Orthogonality | Non-negativity |
| Components | Can be negative; often cancel | Purely additive |
| Interpretability | Directions of variance | Parts / motifs |
| Uniqueness | Essentially unique | Not unique |

For spatial intensity data — where a "negative amount of time spent in the corner" is meaningless — the non-negative factorisation is the semantically correct one.

## Two Uses in the Basketball EPV Model

[[martingale-epv|Cervone et al.'s model]] deploys NMF twice, for quite different purposes.

### 1. Learning player position from court occupancy
A $461 \times 575$ matrix of players × court bins is factorised at rank $r=5$. The rows of $\mathbf{V}$ become basis distributions over the court; the rows of $\mathbf{U}$ give each player a 5-dimensional "position" learned from behaviour rather than from the league's positional labels. Distances in this space then define the neighbourhood structure for the [[car-prior]].

### 2. Compressing spatial effect bases
The [[gaussian-process]] spatial effects are initially represented over a 383-vertex mesh. Simplified per-player hazard models are fit, coefficients exponentiated to put them on a natural scale, and NMF applied to the resulting $461 \times 383$ matrix at rank $d = 10$. The resulting bases are interpretable as shot-type motifs — the paper displays them ordered roughly from close-range to long-range.

Exponentiation before factorisation is deliberate: the coefficients inform a *log* hazard, so exponentiating puts them on the hazard scale and pushes strong negative signals toward zero, where they exert little influence on the factorisation.

## General Applications

Topic modelling (documents × words), audio source separation (spectrograms), recommender systems, and hyperspectral unmixing — all cases where data is inherently non-negative and an additive parts decomposition is meaningful.

## See Also

- [[eigenvector]]
- [[gaussian-process]]
- [[car-prior]]
- [[martingale-epv]]
- [[multiresolution-stochastic-process-nba-possessions|Source Summary]]
