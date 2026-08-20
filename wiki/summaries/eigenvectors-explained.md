---
title: "Eigenvectors Explained — Source Summary"
type: summary
tags: [statistics, linear-algebra, machine-learning, representation-learning]
sources: [raw/articles/eigenvectors-explained.md]
confidence: 0.9
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-08
updated: 2026-07-08
---

# Eigenvectors Explained

**Source type:** Tutorial article (not peer-reviewed)
**Published:** 2026

## Key Content

A beginner-oriented explainer of [[eigenvector]]s and eigenvalues, progressing from physical analogy (fabric stretching) through formal definition to a complete worked example with geometric interpretation.

## Core Definition

Given a matrix $A$ (representing a linear transformation) and a vector $\mathbf{v}$, an eigenvector is a vector whose direction is preserved under the transformation:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda$ is the eigenvalue — the scalar factor by which the eigenvector is stretched or compressed. Eigenvectors are the "magic directions" that only get scaled, never rotated, under the transformation.

## Worked Example

For the matrix $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$:

1. **Set up:** $(A - \lambda I)\mathbf{v} = \mathbf{0}$
2. **Find eigenvalues:** $\det(A - \lambda I) = (3-\lambda)(2-\lambda) = 0$ → $\lambda_1 = 3$, $\lambda_2 = 2$
3. **Find eigenvectors:** $\mathbf{v}_1 = [1, 0]$ (for $\lambda_1 = 3$), $\mathbf{v}_2 = [1, -1]$ (for $\lambda_2 = 2$)
4. **Verify:** $A\mathbf{v}_1 = [3, 0] = 3 \times [1, 0]$ ✓; $A\mathbf{v}_2 = [2, -2] = 2 \times [1, -1]$ ✓

## Why Determinant Must Be Zero

If $(A - \lambda I)$ had an inverse, multiplying both sides of $(A - \lambda I)\mathbf{v} = \mathbf{0}$ by that inverse forces $\mathbf{v} = \mathbf{0}$ — the trivial solution. A zero determinant (singular matrix) means no inverse exists, allowing non-trivial solutions: the eigenvectors.

## Applications Mentioned

- **Principal Component Analysis (PCA):** Eigenvectors of the covariance matrix identify the directions of maximum variance in data.
- **Image compression:** Eigendecomposition identifies key visual patterns.
- **Quantum mechanics:** Eigenstates describe particle measurement outcomes.
- **Structural engineering:** Eigenfrequencies predict natural vibration modes.

## Geometric Interpretation

Eigenvectors reveal the fundamental "stretching directions" of a linear transformation. Every other vector gets both stretched and rotated, but eigenvectors only get stretched. They are the skeleton or backbone of the transformation — analogous to the grain in wood.

## See Also

- [[eigenvector]]
