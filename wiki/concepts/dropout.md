---
title: "Dropout"
type: concept
tags: [deep-learning, regularization]
sources: [raw/papers/rnn-regularisation.md, raw/papers/attention-is-all-you-need.md]
confidence: 0.9
provenance:
  extracted: 50%
  inferred: 40%
  ambiguous: 10%
lifecycle: draft
created: 2026-05-08
updated: 2026-05-08
---

# Dropout

Dropout (Srivastava, 2013) is a regularisation technique that randomly sets a fraction of unit activations to zero during training. This forces the network to learn redundant representations and prevents co-adaptation of features, substantially reducing overfitting.

## Mechanism

During training, each unit is independently "dropped" (set to zero) with probability $p$. At test time, all units are active but outputs are scaled by $(1 - p)$ (or equivalently, inverted dropout scales during training).

## In the Transformer

The [[transformer]] applies dropout ($P_{drop} = 0.1$) to the output of each sub-layer before it is added to the residual, and to attention weights and positional-encoding sums.

## In RNNs

Standard dropout applied to recurrent connections hurts RNNs by amplifying noise across time steps. [[dropout-for-rnns|Zaremba et al. (2014)]] showed that applying dropout only to non-recurrent connections resolves this issue.

## See Also

- [[dropout-for-rnns]]
- [[label-smoothing]]
- [[regularization]]
