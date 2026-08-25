---
title: "KL Divergence"
type: concept
tags: [information-theory, statistics, machine-learning, vae, alignment, probabilistic-classification, approximation]
sources: [raw/papers/variational-lossy-autoencoders.md, raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.8
provenance:
  extracted: 38%
  inferred: 30%
  generated: 6%
  imported: 24%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Kullback-Leibler Divergence

A measure of how one probability distribution differs from another:

$$D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Read as the **expected extra information cost** of coding samples from $P$ using a code built for $Q$. Zero when the distributions match, positive otherwise.

## It Is Not a Distance

Two properties that matter more than the definition:

**Asymmetric.** $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$, and the difference is behavioural rather than cosmetic:

| | Penalises | Resulting fit |
|---|---|---|
| **Forward** $D_{KL}(P \parallel Q)$ | $Q$ small where $P$ is large | **Mass-covering** — $Q$ spreads to cover all of $P$ |
| **Reverse** $D_{KL}(Q \parallel P)$ | $Q$ large where $P$ is small | **Mode-seeking** — $Q$ collapses onto one mode |

Variational methods minimise the **reverse** direction, which is why approximate posteriors are characteristically over-confident and under-dispersed. That is a property of the objective, not a failure of optimisation.

**Unbounded.** If $Q(x) = 0$ where $P(x) > 0$, the divergence is infinite. In practice this forces smoothing or support constraints.

## Three Roles in This Vault

**1. The variational bound.** The ELBO decomposes as reconstruction minus $D_{KL}(q(z \mid x) \parallel p(z))$, so the KL term is what regularises the latent toward the prior. See [[variational-autoencoder]].

**2. The information-preference cost.** [[variational-lossy-autoencoders|VLAE]] identifies an unavoidable penalty $D_{KL}(q(z \mid x) \parallel p(z \mid x))$ from imperfect posterior approximation — and shows that **a sufficiently powerful decoder avoids that cost by ignoring the latent entirely.** The divergence is not incidental there; it explains a failure mode.

**3. The alignment anchor.** [[rlhf|RLHF]] adds a KL penalty between the optimised policy and its starting point, keeping the model near the behaviour it began with while reward is maximised. See [[proximal-policy-optimization]].

The third is the most transferable. **A KL term between a model and a reference model is an imitation constraint written in information-theoretic terms**, and its coefficient controls how much the model is permitted to depart from what it was.

> ### `a-kl-anchor-makes-the-imitation-weight-explicit`
> **Where a model is regularised toward a reference by KL, the coefficient states exactly how strongly observed behaviour constrains the result. That the quantity is nameable and reportable is why RLHF sweeps it as standard — and why anchoring terms written as bespoke auxiliary losses are so often left unreported.**
> ^[generated. rests-on: source:instructgpt-kl-penalty, imported:auxiliary-loss-practice]

## Related Quantities

^[imported]

| | Note |
|---|---|
| **Cross-entropy** | $H(P) + D_{KL}(P \parallel Q)$ — minimising it over $Q$ minimises KL |
| **Jensen–Shannon** | Symmetric, bounded; the divergence implicitly minimised by a GAN discriminator |
| **Mutual information** | $D_{KL}$ between a joint and the product of its marginals |

The first explains why KL rarely appears by name in supervised learning: minimising cross-entropy loss *is* minimising the KL to the empirical distribution, with $H(P)$ constant.

## See Also

- [[variational-autoencoder]] · [[generative-model]] · [[rlhf]] · [[proximal-policy-optimization]] · [[imitation-learning]]
- [[probabilistic-classification]] · [[probability-calibration]] · [[bayesian-inference]] · [[expectation-propagation]]
- [[variational-lossy-autoencoders|VLAE Summary]] · [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
