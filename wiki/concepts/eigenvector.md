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

## Connections to Vault Concepts

### Singular Value Decomposition (SVD)
SVD decomposes any matrix as $A = U\Sigma V^T$, where $U$ and $V$ contain eigenvectors of $AA^T$ and $A^T A$ respectively. SVD is used in the **Direct Linear Transform (DLT)** for [[homography]] estimation — solving for the [[camera-calibration|camera projection]] that best maps field correspondences.

### Principal Component Analysis (PCA)
PCA finds the eigenvectors of the data's covariance matrix. The eigenvector with the largest eigenvalue is the first principal component — the direction of maximum variance. This is used implicitly in dimensionality reduction techniques (e.g., the UMAP step in the [[detection-tracking-football-broadcast-footage|Tshiani detection/tracking pipeline]]).

### Covariance and Bayesian Methods
The [[trueskill]] rating system and [[expectation-propagation]] work with Gaussian distributions parameterised by mean vectors and covariance matrices. The eigenvectors of a covariance matrix define the principal axes of the uncertainty ellipsoid — the directions along which uncertainty is greatest or smallest.

### Attention as Weighted Combination
The [[attention-mechanism]] computes weighted combinations of value vectors: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$. While not an eigendecomposition, this shares the linear-algebraic flavour of projecting data along informative directions — analogous to how PCA projects data along eigenvectors.

## See Also

- [[eigenvectors-explained|Source Summary]]
- [[homography]]
- [[trueskill]]
- [[attention-mechanism]]
