---
title: "Conditional GAN (Pix2Pix)"
type: concept
tags: [deep-learning, generative-model, gan, computer-vision, semantic-segmentation]
sources: [raw/papers/sports-camera_calibration-synthetic_data.md, raw/papers/amateur_footbal_analytics_computer_vision.md]
confidence: 0.9
provenance:
  extracted: 75%
  inferred: 20%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-18
updated: 2026-06-18
---

# Conditional GAN (Pix2Pix)

A conditional GAN (cGAN; Mirza & Osindero, 2014) extends the standard GAN by conditioning both the generator and discriminator on additional input information. Pix2Pix (Isola et al., 2017) applies this to image-to-image translation: given a paired training set $\{(x_i, y_i)\}$, it learns the mapping $G : \{x, z\} \rightarrow y$ from observed image $x$ and noise $z$ to output $y$.

## Architecture

- **Generator:** U-Net encoder-decoder with skip connections that concatenate features at layer $i$ with those at layer $N-i$, preserving spatial detail across the bottleneck.
- **Discriminator:** PatchGAN — classifies each $N \times N$ image patch as real or fake, modelling high-frequency structure while an $L_1$ loss handles low-frequency correctness.

## Loss Function

$$G^* = \arg\min_G \max_D \; \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L_1}(G)$$

where the conditional adversarial loss is:

$$\mathcal{L}_{cGAN} = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x,z)))]$$

## Use in Sports Court Detection

Both [[sports-camera-calibration-synthetic-data|Chen & Little (2019)]] and [[amateur-football-analytics-computer-vision|Mavrogiannis (2021)]] use a **dual-cGAN** (two chained Pix2Pix models) for court detection:

1. **Segmentation GAN:** Input broadcast frame → output grass/non-grass binary mask (removes crowds, ad boards, stadium structures).
2. **Detection GAN:** Input segmented foreground → output edge map of field markings (lines and circles).

Soft alpha-blending at segmentation boundaries prevents the detection GAN from learning segmentation edges as field markings. The grass mask is also applied to the edge map output to suppress false-positive lines outside the court (e.g., behind goal posts).

### Training
Both models trained independently for 200 epochs, batch size 1, $lr = 2 \times 10^{-4}$ (linearly decayed after epoch 100). Input: $3 \times 256 \times 256$ RGB, output: $1 \times 256 \times 256$ binary mask. Data augmentation via random crop ($286 \rightarrow 256$), horizontal flip, and normalisation. Trained on the World Cup 2014 dataset (Homayounfar et al., 2017).

## Relation to Other Generative Models

Unlike [[variational-autoencoder|VAEs]], cGANs produce sharp outputs but lack a principled latent space. Unlike [[autoregressive-model|autoregressive models]], they generate the full output in a single forward pass. The adversarial training replaces hand-designed loss functions with a learned loss, making Pix2Pix applicable to any image-to-image task (segmentation, edge detection, colourisation, style transfer).

## See Also

- [[camera-calibration]]
- [[semantic-segmentation]]
- [[variational-autoencoder]]
