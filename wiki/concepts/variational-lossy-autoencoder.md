---
title: "Variational Lossy Autoencoder"
type: concept
tags: [deep-learning, generative-model, vae, autoregressive-model, representation-learning]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Variational Lossy Autoencoder

The Variational Lossy Autoencoder (VLAE; Chen et al., 2017) is a [[variational-autoencoder]] that combines a global latent code with a local [[autoregressive-model]] decoder (PixelCNN) to learn controllable lossy representations.

## Key Insight: Information Preference

When a VAE has a powerful autoregressive decoder, information that can be modelled locally (without $\mathbf{z}$) will be encoded locally, because using the latent code incurs an unavoidable Bits-Back Coding cost $D_{KL}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x}))$. By constraining the decoder's receptive field, VLAE forces global information into $\mathbf{z}$ while letting local details be handled autoregressively.

## Architecture

- **Encoder:** ResNet-based inference network producing $q(\mathbf{z}|\mathbf{x})$.
- **Decoder:** Small-receptive-field PixelCNN that models local pixel dependencies $p(\mathbf{x}_i|\mathbf{z}, \mathbf{x}_{WindowAround(i)})$.
- **Prior:** Autoregressive flow (AF) transforming simple Gaussian noise, equivalent to IAF posterior but with a deeper generative path at no extra training cost.

## Controllable Representation

The receptive field size determines what is "local" vs "global":
- Small window → latent code captures detailed shapes and colour.
- Large window → latent code captures only coarse structure.
- Grayscale window → latent code additionally preserves colour information.

## See Also

- [[variational-autoencoder]]
- [[autoregressive-model]]
- [[variational-lossy-autoencoders|Source Summary]]
