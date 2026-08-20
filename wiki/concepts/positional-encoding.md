---
title: "Positional Encoding"
type: concept
tags: [transformer, encoding, deep-learning]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.9
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-05-07
---

# Positional Encoding

Positional encodings inject sequence order information into the [[transformer]], which otherwise has no built-in notion of position (unlike RNNs or CNNs).

## Sinusoidal Encoding (Original Transformer)

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

Each dimension corresponds to a sinusoid with wavelengths forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. This design allows the model to attend by relative position, since $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$.

## Learned vs Sinusoidal

The [[attention-is-all-you-need|original paper]] found that learned positional embeddings and sinusoidal encodings produce nearly identical results. The sinusoidal version was preferred because it may generalise to longer sequences than those seen during training.

## Application

Positional encodings are added (summed) to input embeddings at the bottom of both the encoder and decoder stacks, and have the same dimension $d_{\text{model}}$ as the embeddings.

## See Also

- [[transformer]]
- [[attention-mechanism]]
