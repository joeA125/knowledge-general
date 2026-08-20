---
title: "Scaling Laws for Neural Language Models — Source Summary"
type: summary
tags: [deep-learning, transformer, language-modelling, scaling-laws, training-technique]
sources: [raw/papers/scaling-neural-language-models.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Scaling Laws for Neural Language Models

**Authors:** [[jared-kaplan]], [[sam-mccandlish]], Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, [[dario-amodei]]
**Affiliations:** Johns Hopkins University, [[openai]]
**Published:** 2020; arXiv:2001.08361

## Key Contribution

This paper establishes empirical [[scaling-laws]] for [[transformer]] language models, showing that cross-entropy loss follows precise power-law relationships with model size $N$, dataset size $D$, and training compute $C$, spanning over seven orders of magnitude. Performance depends very weakly on architectural details (depth, width, heads) within a wide range.

## Core Scaling Laws

1. **Model size:** $L(N) = (N_c/N)^{\alpha_N}$, where $\alpha_N \approx 0.076$
2. **Dataset size:** $L(D) = (D_c/D)^{\alpha_D}$, where $\alpha_D \approx 0.095$
3. **Compute (optimally allocated):** $L(C_{min}) = (C_c^{min}/C_{min})^{\alpha_C^{min}}$, where $\alpha_C^{min} \approx 0.050$
4. **Joint model-data:** $L(N,D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$
5. **Training curve:** $L(N,S) = (N_c/N)^{\alpha_N} + (S_c/S_{min})^{\alpha_S}$, where $\alpha_S \approx 0.76$

## Key Findings

- **Shape independence:** Performance depends very weakly on depth vs width, number of heads, or feed-forward ratio when total non-embedding parameters $N$ are held fixed.
- **Sample efficiency:** Larger models are significantly more sample-efficient, reaching the same loss with fewer data points.
- **Optimal compute allocation:** Given a fixed compute budget, most should go to increasing model size, with relatively modest increases in data and negligible increases in training steps. Optimal: $N \propto C^{0.73}$, $S \propto C^{0.03}$.
- **Convergence is inefficient:** Compute-efficient training trains very large models and stops significantly before convergence (~10% above converged loss).
- **Overfitting universality:** Overfitting depends on the ratio $N^{0.74}/D$; scaling data sub-linearly with model size avoids overfitting ($D \gtrsim 5000 \cdot N^{0.74}$).
- **Transfer:** Generalisation to other distributions improves with training loss with a roughly constant offset.
- **Transformers vs LSTMs:** Transformers outperform LSTMs asymptotically due to better use of long contexts. LSTMs plateau after ~100 tokens; Transformers improve throughout the full 1024-token context.
- **Critical batch size:** $B_{crit}(L) = B_*/L^{1/\alpha_B}$ depends only on loss, not model size.

## Implications

The scaling relations suggest that larger models will continue to improve and be increasingly sample-efficient. The authors conjecture an eventual intersection between compute-efficient scaling and data-limited scaling at $C^* \sim 10^4$ PF-days, $N^* \sim 10^{12}$ parameters, providing a rough estimate of when Transformer language models reach maximal performance on natural language.

## See Also

- [[scaling-laws]]
- [[transformer]]
