---
title: "Scaling Laws"
type: concept
tags: [deep-learning, scaling-laws, language-modelling, training-technique]
sources: [raw/papers/scaling-neural-language-models.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Scaling Laws

Scaling laws describe the empirical power-law relationships between neural network performance and three key factors: model size ($N$), dataset size ($D$), and training compute ($C$).

## Power-Law Relationships

[[scaling-neural-language-models|Kaplan et al. (2020)]] showed that for [[transformer]] language models:

- $L(N) \propto N^{-0.076}$ — loss decreases as a power of model parameters
- $L(D) \propto D^{-0.095}$ — loss decreases as a power of dataset tokens
- $L(C_{min}) \propto C_{min}^{-0.050}$ — loss decreases as a power of optimally-allocated compute

These trends span 6–8 orders of magnitude with no signs of deviation.

## Key Implications

1. **Scale dominates architecture:** Within a wide range, depth vs width, head count, and FF ratio have minimal effect on performance at fixed $N$.
2. **Optimal allocation:** Given a fixed compute budget, most should go to larger models rather than more training steps.
3. **Sample efficiency:** Larger models reach the same loss with fewer data points.
4. **Sub-linear data scaling:** Dataset size need only grow as $D \propto N^{0.74}$ to avoid overfitting.
5. **Diminishing returns:** All scaling is power-law (not exponential), so each doubling yields a fixed percentage improvement.

## Later Work

Hoffmann et al. (2022, "Chinchilla") revised the optimal compute allocation, arguing for more balanced scaling of $N$ and $D$ than Kaplan et al. suggested.

## See Also

- [[transformer]]
- [[scaling-neural-language-models|Source Summary]]
