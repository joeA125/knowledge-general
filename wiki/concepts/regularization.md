---
title: "Regularization"
type: concept
tags: [regularization, deep-learning, machine-learning]
sources: [raw/papers/rnn-regularisation.md, raw/papers/attention-is-all-you-need.md]
confidence: 0.85
provenance:
  extracted: 45%
  inferred: 50%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Regularization

Regularization refers to any technique that improves a model's generalisation to unseen data, typically by discouraging it from fitting noise in the training set. Regularisation trades a small increase in training error for a larger reduction in the gap between training and test performance.

## Common Techniques

- **[[dropout]]** — randomly zeroing unit activations during training to prevent co-adaptation of features.
- **[[label-smoothing]]** — softening one-hot targets so the model is less over-confident.
- **Weight decay / $L_2$** — penalising large weights.
- **Early stopping, data augmentation, and noise injection** — further widely used approaches.^[inferred: general catalogue beyond the ingested sources]

## In the Vault's Sources

Regularisation is essential to training the deep models discussed here. The [[transformer]] relies on both [[dropout]] ($P_{drop}=0.1$) and [[label-smoothing]] ($\epsilon_{ls}=0.1$); the [[attention-is-all-you-need|paper]] notes dropout is critical to avoid overfitting. For recurrent models, [[dropout-for-rnns|Zaremba et al. (2014)]] showed that *where* dropout is applied matters — applying it only to non-recurrent connections regularises [[lstm|LSTMs]] effectively without destroying their memory.

## See Also

- [[dropout]]
- [[dropout-for-rnns]]
- [[label-smoothing]]
- [[batch-normalization]]
