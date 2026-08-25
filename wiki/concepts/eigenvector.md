---
title: "Eigenvector"
type: concept
tags: [statistics, linear-algebra, machine-learning, representation-learning]
sources: [raw/articles/eigenvectors-explained.md]
confidence: 0.9
provenance:
  extracted: 70%
  inferred: 25%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-08
updated: 2026-07-08
---

# Eigenvector

An eigenvector of a square matrix $A$ is a non-zero vector $\mathbf{v}$ whose direction is preserved when $A$ is applied as a linear transformation:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The scalar $\lambda$ is the corresponding eigenvalue — it quantifies how much the eigenvector is stretched ($|\lambda| > 1$), compressed ($|\lambda| < 1$), or flipped ($\lambda < 0$).

## Geometric Interpretation

A matrix $A$ represents a linear transformation that generally both stretches and rotates vectors. Eigenvectors are the special directions along which the transformation acts purely as scaling — no rotation. They reveal the "skeleton" of the transformation: the natural axes along which the system behaves most simply.

## Finding Eigenvectors

1. **Eigenvalue equation:** Rearrange $A\mathbf{v} = \lambda\mathbf{v}$ to $(A - \lambda I)\mathbf{v} = \mathbf{0}$.
2. **Characteristic polynomial:** For non-trivial solutions ($\mathbf{v} \neq \mathbf{0}$), the matrix $(A - \lambda I)$ must be singular: $\det(A - \lambda I) = 0$. This yields the eigenvalues $\lambda_1, \lambda_2, \ldots$
3. **Null space:** For each $\lambda_i$, solve $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$ to find the corresponding eigenvectors.

## Key Properties

- Eigenvectors are defined up to scalar multiples: if $\mathbf{v}$ is an eigenvector, so is $c\mathbf{v}$ for any $c \neq 0$.
- A symmetric matrix always has real eigenvalues and orthogonal eigenvectors.
- The eigenvalues of a positive semi-definite matrix (e.g., a covariance matrix) are all $\geq 0$.
- An $n \times n$ matrix has at most $n$ linearly independent eigenvectors.

## Where It Appears

### Singular Value Decomposition
SVD decomposes any matrix as $A = U\Sigma V^T$, where $U$ and $V$ contain eigenvectors of $AA^T$ and $A^T A$ respectively. It underpins the **Direct Linear Transform**, the standard least-squares method for estimating a [[homography]] from point correspondences — the smallest singular vector gives the solution to a homogeneous system.

### Principal Component Analysis
PCA finds the eigenvectors of a data covariance matrix. The eigenvector with the largest eigenvalue is the direction of maximum variance.

Worth contrasting with [[non-negative-matrix-factorization|NMF]], which decomposes the same kind of matrix under a different constraint. **PCA's orthogonality gives uniqueness and components that may cancel; NMF's non-negativity gives interpretable parts and loses uniqueness.** The choice is about what the components are meant to mean, not about fit quality.

### Covariance and Bayesian Methods
[[trueskill]] and [[expectation-propagation]] work with Gaussians parameterised by mean vectors and covariance matrices. **The eigenvectors of a covariance matrix are the principal axes of the uncertainty ellipsoid** — the directions along which uncertainty is greatest and smallest.

That geometry is what a scalar summary of uncertainty discards: two distributions with the same total variance can be uncertain in entirely different directions. See [[uncertainty-quantification]].

### Spectral Methods More Broadly
Graph Laplacian eigenvectors underpin spectral clustering and spectral embeddings; the leading eigenvector of a transition matrix gives a stationary distribution. **Wherever a problem reduces to "find the directions in which this operator acts simply", eigendecomposition is the tool.**^[imported]

## See Also

- [[non-negative-matrix-factorization]] · [[representation-learning]] · [[homography]] · [[camera-calibration]]
- [[trueskill]] · [[expectation-propagation]] · [[uncertainty-quantification]] · [[bayesian-inference]] · [[attention-mechanism]]
- [[eigenvectors-explained|Source Summary]]
