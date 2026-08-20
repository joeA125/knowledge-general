---
title: "Adam Optimizer"
type: concept
tags: [deep-learning, training-technique]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.8
provenance:
  extracted: 40%
  inferred: 55%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-07
updated: 2026-07-07
---

# Adam Optimizer

Adam (Adaptive Moment Estimation; Kingma & Ba, 2015) is a first-order gradient-based optimisation algorithm that adapts the learning rate per parameter using running estimates of the first and second moments of the gradients. It is one of the most widely used optimisers for training deep networks.^[inferred: Adam's definition is general background; the vault has not ingested the original Adam paper]

## Mechanism

Adam maintains exponentially decaying averages of past gradients ($m_t$, the first moment) and past squared gradients ($v_t$, the second moment), applies bias correction, and updates parameters by:

$$\theta_t = \theta_{t-1} - \frac{\eta \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The per-parameter scaling makes it robust to sparse gradients and to differing gradient magnitudes across parameters, typically requiring less learning-rate tuning than plain SGD.

## Use in the Transformer

The [[transformer]] is trained with Adam using $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, together with a custom warmup schedule: the learning rate rises linearly for the first $\text{warmup\_steps} = 4000$ steps, then decays proportionally to the inverse square root of the step number. This schedule is important for stable Transformer training and is extracted from [[attention-is-all-you-need|Attention Is All You Need]].

## See Also

- [[transformer]]
- [[dropout]]
- [[label-smoothing]]
- [[attention-is-all-you-need|Source Summary]]
