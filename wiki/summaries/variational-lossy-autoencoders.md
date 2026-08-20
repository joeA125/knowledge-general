---
title: "Variational Lossy Autoencoder — Source Summary"
type: summary
tags: [deep-learning, generative-model, vae, autoregressive-model, representation-learning, density-estimation]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Variational Lossy Autoencoder

**Authors:** Xi Chen, [[diederik-kingma]], Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, [[ilya-sutskever]], Pieter Abbeel
**Affiliations:** UC Berkeley, [[openai]]
**Published:** 2017 (ICLR 2017); arXiv:1611.02731

## Key Contribution

This paper introduces the [[variational-lossy-autoencoder]] (VLAE), which combines a VAE with autoregressive models (PixelCNN) in a principled way to learn controllable lossy representations. The key insight is the **information preference property**: when a VAE decoder is sufficiently powerful (e.g., autoregressive), information that can be modelled locally by the decoder will be encoded locally, and only the remainder is captured by the global latent code $\mathbf{z}$.

## The Problem: VAEs with Powerful Decoders Ignore Latent Codes

When a VAE uses an autoregressive decoder like an RNN, the latent code $\mathbf{z}$ is typically ignored — the model collapses to an unconditional autoregressive model. Previous work attributed this to "optimization challenges," but this paper shows it is a fundamental consequence of the Bits-Back Coding interpretation: using the latent code incurs an unavoidable cost $D_{KL}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x}))$ from imperfect posterior approximation. When the decoder can model the data without $\mathbf{z}$, the model avoids this cost by ignoring the latent code.

## Solution: Explicit Information Placement

By constraining the decoder's receptive field (e.g., using a small-window PixelCNN), information that requires long-range dependencies cannot be modelled locally and must flow through $\mathbf{z}$. This turns the VAE into a lossy compressor where:
- **Local statistics** (texture, noise) are captured by the autoregressive decoder.
- **Global structure** (shapes, object identity) is encoded in the latent code.

The type of information in $\mathbf{z}$ is controllable by designing the decoder's receptive field.

## Autoregressive Flow Prior

VLAE also uses an autoregressive flow (AF) as a learned prior $p(\mathbf{z})$. This is shown to be equivalent to using an Inverse Autoregressive Flow (IAF) posterior but with a deeper, more expressive generative model at no additional training cost.

## Results

### Binary Image Density Estimation
| Dataset | VLAE NLL (test) | Previous SOTA |
|---|---|---|
| Static MNIST | **79.03** | 79.20 (PixelRNN) |
| Dynamic MNIST | **78.53** | 79.10 (IAF VAE) |
| OMNIGLOT | **89.83** (tuned) | ~91.00 (Conv DRAW) |
| Caltech-101 | **77.36** | 88.48 (SpARN) |

### CIFAR-10
DenseNet VLAE achieves **2.95 bits/dim**, outperforming all other variationally-trained latent variable models and most tractable likelihood models (behind only PixelCNN++ at 2.92).

### Lossy Compression
On MNIST, VLAE uses 19.2 bits per image (vs 37.3 bits for a standard VAE), encoding only global structure. Decompressed images preserve digit identity but differ in local texture/style.

On CIFAR-10, larger decoder receptive fields progressively move more structural information into local modelling, making the latent code increasingly abstract.

## See Also

- [[variational-lossy-autoencoder]]
- [[variational-autoencoder]]
- [[autoregressive-model]]
