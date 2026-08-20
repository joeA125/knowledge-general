---
title: "Generative Model"
type: concept
tags: [generative-model, machine-learning, deep-learning, density-estimation, vae, gan, autoregressive-model, counterfactual, representation-learning]
sources: [raw/papers/variational-lossy-autoencoders.md, raw/papers/scoutgpt-generative-transformer-football-player-valuation.md, raw/papers/evaluation_creating_scoring_opportunities_trajectory_prediction.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 55%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-27
updated: 2026-07-27
---

# Generative Model

A model of the data distribution $p(x)$ itself, rather than only of a conditional $p(y|x)$. It can be *sampled from* — asked to produce new data — which is what distinguishes it from a discriminative model and what makes a whole class of downstream uses possible.

## The Families

The vault holds instances of each, and they differ mainly in how they make the likelihood tractable.

| Family | Mechanism | Likelihood | In this vault |
|---|---|---|---|
| **[[autoregressive-model\|Autoregressive]]** | Chain rule: $p(x) = \prod_t p(x_t \mid x_{<t})$ | Exact | [[gpt]], [[large-event-model]], [[eventgpt]], [[scoutgpt]] |
| **[[variational-autoencoder\|Latent variable / VAE]]** | Latent $z$, optimise an evidence lower bound | Bounded | [[variational-lossy-autoencoder]], VRNN, GVRNN |
| **[[conditional-gan\|Adversarial]]** | Generator versus discriminator | Implicit | Pix2Pix, CycleGAN |

Autoregressive models give exact likelihoods and strong samples but generate sequentially, so long outputs are slow and errors compound — see [[teacher-forcing]]. VAEs give a usable latent space and fast sampling but blur detail. GANs give sharp samples and no likelihood at all, which makes them hard to evaluate and to compose.

## What Generativity Buys

Being able to sample is not the point in most vault applications. Three derived capabilities are.

**Counterfactuals.** A generative model conditioned on an intervenable entity can be asked what *would* have happened. [[scoutgpt]] substitutes a player into a lineup and regenerates the sequence; [[eventgpt]] substitutes and re-scores. Neither is possible with a discriminative model, which can only score what exists. See [[counterfactual-simulation]].

**Reference behaviour.** A generative trajectory model trained on league-wide data produces what a *typical* player would have done, which becomes a baseline to measure deviation against. This is [[c-obso]]'s mechanism — and note it inverts the usual objective, since the model is wanted for its notion of *normal* rather than its accuracy. See [[counterfactual-baseline]] and [[imitation-learning]].

**Multimodality.** A deterministic model asked to predict several plausible futures returns their average, which is often implausible — the midpoint between "runs left" and "runs right" is "stands still". Injecting a stochastic latent lets a model represent the modes separately. This is the specific failure that motivates VRNN over RNN in [[trajectory-prediction]], and the endpoint-error gap there is nearly an order of magnitude.

## The Evaluation Problem

Generative models are hard to evaluate, and the difficulty runs deeper than metric choice.

Likelihood is available for autoregressive models and bounded for latent-variable ones, absent for GANs. But high likelihood does not imply good samples, and good samples do not imply high likelihood — the two can be traded against each other, which [[variational-lossy-autoencoder|the VLAE work]] exploits deliberately by discarding local detail into the decoder.

For the vault's sports applications this bites hard, because the model is a *means*: [[scoutgpt]]'s value depends on its counterfactuals being right, not its perplexity being low. Held-out likelihood cannot tell you whether substituting a player produces a realistic alternative world. The substitute checks used — self-to-self reconstruction, out-of-sample transfer prediction — are weaker, and both are discussed on [[counterfactual-simulation]].

## The Causal Caveat

Worth stating plainly because the language invites the error: a generative model trained on observational data learns the **observational** distribution. Intervening and regenerating gives the correct causal answer only if the model captured the right dependency structure and nothing important is unmeasured.

In football, observed performance is confounded with team quality, tactics and opposition. Conditioning on a lineup absorbs some of that; nothing guarantees the learned association is a causal effect. **"Generative" is not "causal"**, and treating the two as equivalent is the most common error in this literature.

## See Also

- [[autoregressive-model]] · [[variational-autoencoder]] · [[variational-lossy-autoencoder]] · [[conditional-gan]]
- [[counterfactual-simulation]] · [[counterfactual-baseline]] · [[imitation-learning]] · [[trajectory-prediction]]
- [[scoutgpt]] · [[eventgpt]] · [[large-event-model]] · [[c-obso]]
- [[teacher-forcing]] · [[event-prediction]] · [[space-creation]]
