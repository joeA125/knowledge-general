---
title: "Variational Autoencoder"
type: concept
tags: [deep-learning, generative-model, vae, bayesian, inference, representation-learning, trajectory-prediction, rnn]
sources: [raw/papers/variational-lossy-autoencoders.md, raw/papers/evaluation_creating_scoring_opportunities_trajectory_prediction.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 50%
  ambiguous: 10%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-07-27
---

# Variational Autoencoder

The Variational Autoencoder (VAE; Kingma & Welling, 2013; Rezende et al., 2014) is a [[generative-model|generative model]] that learns a latent representation by jointly training an encoder $q(\mathbf{z}|\mathbf{x})$ and decoder $p(\mathbf{x}|\mathbf{z})$ to maximise the variational lower bound (ELBO):

$$\mathcal{L}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The first term encourages reconstruction; the [[kl-divergence|KL]] term regularises the latent code toward the prior.

## Where It Sits Among Generative Families

The VAE's defining trade is **a bounded likelihood in exchange for a usable latent space**. Autoregressive models give exact likelihoods but no compact representation and slow sequential sampling; GANs give sharp samples but no likelihood at all.

| | Likelihood | Latent space | Sampling |
|---|---|---|---|
| [[autoregressive-model\|Autoregressive]] | Exact | None | Sequential, slow |
| **VAE** | **Bounded (ELBO)** | **Yes, structured** | One shot, fast |
| GAN | Implicit | Yes, unstructured | One shot, fast |

The latent space is what most applications actually want — see [[generative-model]] for the derived capabilities that follow from it.

## When Does a VAE Autoencode?

The [[variational-lossy-autoencoder|VLAE paper]] showed that VAEs do not always autoencode: when the decoder is powerful enough (e.g. autoregressive), it can model data without using $\mathbf{z}$, causing the latent code to be ignored.

This is usually described as a failure — *posterior collapse* — but VLAE's contribution was to turn it into a design lever. If the decoder will absorb whatever it can model locally, then **restricting the decoder's receptive field controls what the latent must encode.** Local texture goes to the decoder; global structure is forced into $\mathbf{z}$. Lossiness becomes a specification rather than a defect.

## Sequential Variants

The VAE composes with recurrence, and the composition is what makes it useful for [[trajectory-prediction]].

**VRNN** (Chung et al., 2015) conditions the prior on an RNN hidden state and injects a fresh latent at each timestep:

$$p_\theta(z_t | x_{<t}, z_{<t}) = \varphi_{\text{prior}}(h_{t-1}), \qquad h_t = f(x_t, z_t, h_{t-1})$$

Trained by maximising a sequential ELBO — a sum of per-timestep VAE bounds.

**GVRNN** (Yeh et al., 2019) replaces VRNN's per-agent networks with [[graph-neural-network|graph neural networks]], so each agent's latent is conditioned on all others through message passing.

**Why the latent matters here** is worth being precise about. A deterministic sequence model asked to predict several plausible futures returns their *average*, which is often physically implausible — the midpoint between "runs left" and "runs right" is "stands still". The stochastic latent lets the model represent modes separately rather than blend them. The empirical gap is large: 0.608 m endpoint error for GVRNN against VRNN's 5.952 m at four seconds, though that comparison also confounds graph structure with centralised optimisation.

## An Unusual Use: The Model as a Measuring Instrument

In [[c-obso]], a GVRNN trained on opponent data supplies not a forecast but a **[[counterfactual-baseline|reference]]** — what an average player would have done — against which a specific player's movement is measured.

This inverts the usual objective. A generative model is normally wanted for sample quality or likelihood; here it is wanted for a well-calibrated notion of *normal*. The metric is identically zero under a perfect model, so accuracy and usefulness pull against each other. See [[imitation-learning]].

## See Also

- [[generative-model]] · [[variational-lossy-autoencoder]] · [[autoregressive-model]]
- [[trajectory-prediction]] · [[graph-neural-network]] · [[c-obso]] · [[counterfactual-baseline]]
- [[kl-divergence]] · [[bayesian-inference]] · [[recurrence]] · [[lstm]]
- [[variational-lossy-autoencoders|VLAE Summary]] · [[creating-scoring-opportunities-trajectory-prediction|C-OBSO Summary]]
