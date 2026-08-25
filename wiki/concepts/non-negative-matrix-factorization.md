---
title: "Non-negative Matrix Factorization (NMF)"
type: concept
tags: [linear-algebra, dimensionality-reduction, machine-learning, representation-learning, statistics, clustering, interpretability]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 30%
  generated: 6%
  imported: 62%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-23
updated: 2026-08-14
---

# Non-negative Matrix Factorization (NMF)

> ⚠️ **No held source.** Background knowledge, marked `imported:`. Lee & Seung (1999) is the primary source.

Approximating a non-negative matrix $\mathbf{G}$ as a product of two low-rank non-negative factors:

$$\mathbf{G} \approx \mathbf{U}\mathbf{V}, \qquad \mathbf{U} \in \mathbb{R}_{\ge 0}^{L \times r}, \; \mathbf{V} \in \mathbb{R}_{\ge 0}^{r \times p}$$

found by minimising a divergence $D(\mathbf{G}, \mathbf{U}\mathbf{V})$ subject to non-negativity. The two standard choices are squared Frobenius error and a [[kl-divergence|KL]]-type divergence:

$$D(\mathbf{G}, \mathbf{G}^*) = \sum_{i,j} G_{ij}\log(G_{ij}/G^*_{ij}) - G_{ij} + G^*_{ij}$$

The KL form is preferred for count or intensity data, where the Frobenius error implicitly assumes Gaussian noise that count data does not have.

## Why Non-negativity Matters

The constraint is what distinguishes NMF and gives it its characteristic behaviour. Because components can only **add**, never cancel, NMF tends to learn **parts-based, additive representations**: the rows of $\mathbf{V}$ behave like interpretable building blocks, and the rows of $\mathbf{U}$ say how much of each a given unit uses.

| | PCA / SVD | NMF |
|---|---|---|
| Constraint | Orthogonality | **Non-negativity** |
| Components | Can be negative; often cancel | **Purely additive** |
| Reads as | Directions of variance | **Parts or motifs** |
| Uniqueness | Essentially unique | **Not unique** |

> ### `the-constraint-buys-interpretability-and-costs-uniqueness`
> **Non-negativity produces components that can be read as parts, because subtraction is unavailable and every component must contribute something present. The same constraint removes the orthogonality that makes [[eigenvector\|spectral]] decompositions unique — so an NMF result is one valid factorisation among many, and re-running with a different initialisation may give a different story.**
> ^[generated. rests-on: imported:nmf-properties]

That non-uniqueness is frequently forgotten when NMF components are interpreted substantively. **A parts decomposition that changes under reinitialisation is a description of one solution, not of the data.** See [[model-selection]].

## Where It Fits

Wherever data is inherently non-negative and an additive decomposition is meaningful:

- **Topic modelling** — documents × words, with topics as additive word distributions
- **Audio source separation** — spectrograms, with sources as additive spectral patterns
- **Recommender systems** — users × items
- **Hyperspectral unmixing** — pixels × wavelengths, with materials as endmembers
- **Spatial intensity data** — where a "negative amount of occupancy" is meaningless

The last is the clearest case for the constraint being *semantically* correct rather than merely convenient: a factorisation permitting negative occupancy would fit better and mean less.

## Two Distinct Uses

Worth separating because they are often conflated:

**As dimensionality reduction** — replace a wide matrix with a compact one, and use the low-dimensional rows as coordinates. Here $\mathbf{U}$ is the output and the bases are incidental.

**As basis discovery** — learn the components themselves, because the parts are the object of interest. Here $\mathbf{V}$ is the output.

The rank $r$ is an asserted parameter in both cases and is rarely swept. See [[model-selection]].

## A Practical Note on Scaling

Where a matrix holds coefficients on a log scale, exponentiating **before** factorisation is often the right move: it puts values on their natural scale and pushes strong negative signals toward zero, where they exert little influence on an additive decomposition. Factorising log-scale values directly asks a non-negative method to explain a quantity that is not non-negative.^[imported]

## See Also

- [[eigenvector]] · [[representation-learning]] · [[model-selection]] · [[feature-engineering]]
- [[kl-divergence]] · [[interpretability]] · [[probabilistic-classification]] · [[uncertainty-quantification]]