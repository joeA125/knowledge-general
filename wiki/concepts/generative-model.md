---
title: "Generative Model"
type: concept
tags: [generative-model, machine-learning, deep-learning, density-estimation, vae, gan, autoregressive-model, counterfactual, representation-learning]
sources: [raw/papers/variational-lossy-autoencoders.md]
confidence: 0.8
provenance:
  extracted: 30%
  inferred: 40%
  generated: 8%
  imported: 20%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Generative Model

A model of the data distribution $p(x)$ itself, rather than only a conditional $p(y \mid x)$. It can be **sampled from**, which distinguishes it from a discriminative model and enables a class of downstream uses.

## The Families

They differ mainly in how they make the likelihood tractable, and each pays for it somewhere.

| Family | Mechanism | Likelihood | Cost |
|---|---|---|---|
| **[[autoregressive-model\|Autoregressive]]** | Chain rule: $p(x) = \prod_t p(x_t \mid x_{<t})$ | **Exact** | Sequential generation; errors compound |
| **[[variational-autoencoder\|Latent variable]]** | Latent $z$, optimise a lower bound | **Bounded** | Blurred detail |
| **[[conditional-gan\|Adversarial]]** | Generator versus discriminator | **Implicit** | No likelihood — hard to evaluate or compose |

The third row's cost is easy to underrate. Without a likelihood a model cannot be compared to another on held-out data, cannot be used as a component in a larger probabilistic model, and cannot report how surprised it is by an input.

## What Generativity Buys

Sampling is rarely the point. Three derived capabilities usually are.

**Counterfactuals.** A model conditioned on an intervenable entity can be asked what *would* have happened under a substitution. A discriminative model can only score what exists. See [[counterfactual-simulation]].

**Reference behaviour.** A model trained on a population produces what a *typical* instance would look like, which becomes a baseline to measure deviation against. Note that this **inverts the usual objective** — the model is wanted for its notion of *normal* rather than its accuracy, and under a perfect model the deviation is identically zero. See [[counterfactual-baseline]] and [[imitation-learning]].

**Multimodality.** A deterministic model asked for several plausible futures returns their **average**, which is frequently implausible — the midpoint of two valid options is often not a valid option. A stochastic latent lets the modes be represented separately rather than blended.

> ### `averaging-over-modes-produces-invalid-outputs`
> **Where a distribution is genuinely multimodal, a deterministic model minimising expected error returns a point between the modes. That point may be not merely unlikely but impossible, and nothing in the loss registers the difference.**
> ^[generated. rests-on: imported:multimodal-regression]

## The Evaluation Problem

Harder than metric choice, and the difficulty is structural.

**High likelihood does not imply good samples, and good samples do not imply high likelihood.** The two can be traded deliberately — [[variational-lossy-autoencoders|VLAE]] exploits exactly this, pushing local detail into an autoregressive decoder so the latent carries global structure and the likelihood is spent elsewhere.

That matters most where the model is a **means rather than an end**. If a generative model exists to produce counterfactuals, its value depends on those counterfactuals being right, not on its perplexity being low — and held-out likelihood cannot tell you whether an intervened-upon sample describes a realistic alternative world.

The available substitutes — self-to-self reconstruction, out-of-sample intervention — are weaker and are discussed on [[counterfactual-simulation]].

## The Causal Caveat

Worth stating plainly because the vocabulary invites the error.

A generative model trained on observational data learns the **observational** distribution. Intervening on it and regenerating gives the correct causal answer only if the model captured the right dependency structure and nothing important is unmeasured — neither of which is typically checked.

**"Generative" is not "causal."** Conditioning on an observed variable absorbs some confounding and guarantees nothing about the rest. See [[selection-bias]] for the version of this that arises before fitting, and [[domain-adaptation]] for the version that arises at deployment.

## See Also

- [[autoregressive-model]] · [[variational-autoencoder]] · [[conditional-gan]] · [[transformer]] · [[gpt]]
- [[counterfactual-simulation]] · [[counterfactual-baseline]] · [[imitation-learning]] · [[agent-based-simulation]]
- [[kl-divergence]] · [[representation-learning]] · [[event-prediction]] · [[selection-bias]] · [[domain-adaptation]] · [[uncertainty-quantification]]
- [[variational-lossy-autoencoders|VLAE Summary]]
