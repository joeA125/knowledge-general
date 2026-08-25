---
title: "Homography"
type: concept
tags: [projective-geometry, computer-vision, camera-calibration, image-alignment, linear-algebra, approximation]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 26%
  generated: 6%
  imported: 66%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Homography

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

A projective transformation between two planes, represented by a $3\times3$ matrix acting on homogeneous coordinates:

$$\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = \mathbf{H} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}, \qquad \mathbf{H} \in \mathbb{R}^{3\times3}$$

with image coordinates recovered as $(x'/w', y'/w')$. Defined **up to scale** — $\mathbf{H}$ and $c\mathbf{H}$ give the same mapping — so it has **8 degrees of freedom**, not 9.

## When a Homography Is the Right Model

Two conditions, either of which suffices:

- **The scene is planar.** Any two images of a plane are related by a homography, whatever the camera motion.
- **The camera rotates about its optical centre.** Any two views from a common centre are related by a homography, whatever the scene.

**Where neither holds, a homography is the wrong model** and no amount of fitting fixes it — a scene with depth viewed from two different positions requires the full epipolar geometry, because parallax cannot be expressed by a plane-to-plane map.

That failure is worth recognising because it is silent: a homography fit to a non-planar scene will produce a plausible-looking matrix with high residuals concentrated in the out-of-plane regions.

## Estimation

^[imported]

**The Direct Linear Transform** is the standard method. Each point correspondence supplies two linear constraints on $\mathbf{H}$, so **four correspondences in general position** determine it exactly. With more, the system is overdetermined and solved in least squares.

The DLT is a homogeneous system $\mathbf{A}\mathbf{h} = \mathbf{0}$, solved by taking the singular vector corresponding to the smallest singular value — which is why [[eigenvector|SVD]] is the computational core of homography estimation.

Two refinements matter in practice:

- **Normalisation.** Translating and scaling the points to zero mean and unit average distance before the DLT dramatically improves conditioning. This is not optional; the unnormalised DLT is numerically poor.
- **Robust fitting.** RANSAC or a similar scheme handles outlier correspondences, which are near-universal in real matching.

## The Difference From Calibration

A homography maps **image plane to image plane** or **plane to plane**. It says nothing about where the camera is or what its focal length is.

[[camera-calibration]] recovers the camera's intrinsic and extrinsic parameters — a stronger claim requiring stronger assumptions. **A homography is sufficient for warping one view onto another and insufficient for reasoning about 3D position.**

Where the scene contains a known planar structure with measurable geometry, a homography can be *decomposed* into intrinsics and pose, but the decomposition is not unique without additional constraints.

## Related Transformations

| | DOF | Preserves |
|---|---|---|
| Translation | 2 | Everything but position |
| Euclidean | 3 | Lengths, angles |
| Similarity | 4 | Angles, ratios of lengths |
| Affine | 6 | Parallelism |
| **Homography** | **8** | **Straight lines only** |

The progression is a hierarchy — each row is a special case of the one below. **A homography preserves collinearity and nothing else**, which is exactly what makes it right for perspective and wrong when a simpler model would do: fitting 8 parameters where 4 suffice invites overfitting to correspondence noise.

## See Also

- [[camera-calibration]] · [[eigenvector]] · [[semantic-segmentation]] · [[siamese-network]]
- [[model-selection]] · [[uncertainty-quantification]] · [[feature-engineering]]