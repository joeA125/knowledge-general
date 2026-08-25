---
title: "Variational Autoencoder"
type: concept
tags: [deep-learning, generative-model, vae, bayesian, inference, representation-learning, rnn]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 34%
  generated: 6%
  imported: 18%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-08-14
---

# Variational Autoencoder

A [[generative-model|generative model]] (Kingma & Welling, 2013; Rezende et al., 2014) that learns a latent representation by jointly training an encoder $q(\mathbf{z} \mid \mathbf{x})$ and decoder $p(\mathbf{x} \mid \mathbf{z})$ to maximise the evidence lower bound:

$$\mathcal{L}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x} \mid \mathbf{z})] - D_{KL}(q(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))$$

The first term encourages reconstruction; the [[kl-divergence|KL]] term regularises the latent toward the prior.

## Where It Sits Among Generative Families

The defining trade is **a bounded likelihood in exchange for a usable latent space.**

| | Likelihood | Latent space | Sampling |
|---|---|---|---|
| [[autoregressive-model\|Autoregressive]] | Exact | None | Sequential, slow |
| **VAE** | **Bounded (ELBO)** | **Yes, structured** | One shot, fast |
| [[conditional-gan\|GAN]] | Implicit | Yes, unstructured | One shot, fast |

The latent space is what most applications actually want — see [[generative-model]] for what follows from having one.

## When Does a VAE Autoencode?

[[variational-lossy-autoencoders|VLAE]] showed that VAEs do not always autoencode. When the decoder is powerful enough — autoregressive, say — it can model the data **without using $\mathbf{z}$ at all**, and the latent is ignored.

Usually called *posterior collapse* and treated as a failure. VLAE's contribution was to show it is structural rather than an optimisation problem — using the latent incurs an unavoidable cost from imperfect posterior approximation, so a decoder that can avoid that cost will — and then to turn it into a **design lever**.

**Restricting the decoder's receptive field controls what the latent must encode.** Local texture goes to the decoder; global structure has nowhere to go but $\mathbf{z}$. Lossiness becomes a specification rather than a defect.

That generalises well beyond VAEs: see `a-representation-learns-what-it-is-not-given-for-free` on [[representation-learning]].

## Sequential Variants

The VAE composes with recurrence, which is what makes it useful for modelling sequences of interacting entities.

**VRNN** conditions the prior on a recurrent hidden state and injects a fresh latent at each timestep:

$$p_\theta(z_t \mid x_{<t}, z_{<t}) = \varphi_{\text{prior}}(h_{t-1}), \qquad h_t = f(x_t, z_t, h_{t-1})$$

Trained by maximising a sequential ELBO — a sum of per-timestep bounds.^[imported: Chung et al. 2015; not held]

**GVRNN** replaces the per-entity networks with [[graph-neural-network|graph neural networks]], so each entity's latent is conditioned on all others through [[message-passing]].^[imported: Yeh et al. 2019; not held]

**Why the latent matters here** is the multimodality argument. A deterministic sequence model asked for several plausible futures returns their *average*, which is frequently not a valid outcome. The stochastic latent lets modes be represented separately rather than blended. See `averaging-over-modes-produces-invalid-outputs` on [[generative-model]].

## The Model as a Measuring Instrument

A use worth separating, because it inverts the usual objective.

A sequential VAE trained on a population produces what a *typical* instance would do — which can serve as a **[[counterfactual-baseline|reference]]** against which a specific instance's deviation is measured.

The model is then wanted not for sample quality or likelihood but for a well-calibrated notion of *normal*. **Accuracy and usefulness pull against each other**: under a perfect model the measured deviation is identically zero. See [[imitation-learning]] and [[counterfactual-baseline]].

## See Also

- [[generative-model]] · [[autoregressive-model]] · [[conditional-gan]] · [[kl-divergence]] · [[representation-learning]]
- [[graph-neural-network]] · [[message-passing]] · [[counterfactual-baseline]] · [[counterfactual-simulation]] · [[imitation-learning]]
- [[bayesian-inference]] · [[lstm]] · [[gated-recurrent-unit]] · [[recurrence]] · [[trajectory-prediction]]
- [[variational-lossy-autoencoders|VLAE Summary]]
