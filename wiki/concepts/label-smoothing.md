---
title: "Label Smoothing"
type: concept
tags: [regularization, training-technique, deep-learning]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.85
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-07
updated: 2026-05-07
---

# Label Smoothing

Label smoothing is a regularisation technique (Szegedy et al., 2015) that replaces hard one-hot target distributions with a mixture of the one-hot target and a uniform distribution over all classes, controlled by a parameter $\epsilon_{ls}$.

## In the Transformer

The [[transformer]] uses $\epsilon_{ls} = 0.1$. This hurts perplexity (the model becomes less confident) but improves accuracy and BLEU score, as it prevents the model from becoming overly certain about its predictions.

## See Also

- [[transformer]]
- [[dropout]]
