---
title: "Camera Calibration"
type: concept
tags: [camera-calibration, projective-geometry, computer-vision, radial-distortion, model-selection, uncertainty-quantification]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 26%
  generated: 8%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Camera Calibration

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Recovering the parameters that map 3D world coordinates to 2D image coordinates. The prerequisite for any measurement made from an image — distance, size, position, speed.

## Two Kinds of Parameter

| | Describes | Typical parameters |
|---|---|---|
| **Intrinsic** | The camera itself | Focal length, principal point, skew, lens distortion |
| **Extrinsic** | Where the camera is | Rotation (3 DOF), translation (3 DOF) |

The pinhole model composes them into a projection matrix $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$, where $\mathbf{K}$ holds the intrinsics and $[\mathbf{R} \mid \mathbf{t}]$ the pose.

**Intrinsics are usually fixed for a given camera and lens; extrinsics change whenever it moves.** That asymmetry is why calibration is often split into an offline stage estimating $\mathbf{K}$ once, and an online stage tracking pose per frame.

## Why Distortion Breaks the Linear Model

The pinhole projection is linear in homogeneous coordinates — which is what makes [[homography|homographies]] and the DLT work. Real lenses are not.

**Radial distortion** bends straight lines, most visibly toward image edges and most severely on wide-angle lenses. It is modelled by a polynomial in radial distance and is **non-linear**, so it cannot be folded into $\mathbf{P}$.

The practical consequence: **calibration is a two-stage problem.** Estimate a linear model, undistort, then refine — usually by non-linear least squares over reprojection error. Skipping the undistortion step produces a projection that fits the image centre well and degrades toward the edges, which is a characteristic and easily-missed failure.

## Estimation Strategies

^[imported]

| Approach | Requires | Note |
|---|---|---|
| **Calibration target** | Physical access, a known pattern | Most accurate; often impossible after the fact |
| **Known scene geometry** | Identifiable structures of known dimension | Common where the scene is standardised |
| **Self-calibration** | Multiple views, rigidity | No prior geometry; weaker constraints, less stable |
| **Learned regression** | Training data with known parameters | Fast; generalises only as far as the training distribution |

The learned approach has an evaluation trap worth naming. **Synthetic training data can be generated in unlimited quantity with perfect ground truth, and differs systematically from real imagery** — so a model can report excellent accuracy on held-out synthetic data and degrade badly in deployment. That is [[domain-adaptation]] in its Sim-to-Real form, and the standard mitigation is domain randomisation.

## Evaluating a Calibration

The natural metric is **reprojection error** — project known 3D points and measure pixel distance to their observed positions.

Two limitations:

- **It is measured where correspondences exist.** A calibration fitted to points clustered in one region can be arbitrarily wrong elsewhere, and the error metric will not say so.
- **Pixel error is not world error.** The same reprojection error corresponds to very different physical distances depending on depth and viewing angle — near the horizon, a pixel is metres.

> ### `reprojection-error-understates-error-where-it-matters-most`
> **Reprojection error is uniform in image space and highly non-uniform in world space. A calibration with low mean pixel error can carry large positional error precisely in the regions of greatest depth, and the metric provides no signal about it.**
> ^[generated. rests-on: imported:reprojection-error-properties]

A world-space evaluation — projecting into a metric ground plane and measuring there — is more informative and less commonly reported.

## See Also

- [[homography]] · [[eigenvector]] · [[siamese-network]] · [[semantic-segmentation]] · [[conditional-gan]]
- [[domain-adaptation]] · [[model-selection]] · [[uncertainty-quantification]] · [[selection-bias]] · [[feature-engineering]]