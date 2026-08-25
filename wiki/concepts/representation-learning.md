---
title: "Representation Learning"
type: concept
tags: [representation-learning, machine-learning, deep-learning, feature-engineering, entity-embedding, dimensionality-reduction, pre-training, tokenization, theory-based-modelling]
sources: [raw/papers/variational-lossy-autoencoders.md, raw/papers/bert-bidirectional-transformers.md]
confidence: 0.8
provenance:
  extracted: 30%
  inferred: 38%
  generated: 12%
  imported: 18%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Representation Learning

Learning what to *feed* a model rather than hand-specifying it. The premise is that most of a model's performance is determined by how its inputs are encoded, and that the encoding can itself be learned.

## Three Routes

| Route | Mechanism | Examples |
|---|---|---|
| **Mathematical** | A principled transform with known properties | [[non-negative-matrix-factorization\|NMF]], [[eigenvector\|spectral methods]] |
| **Architectural** | Structure that makes the right thing expressible | [[graph-neural-network\|GNN]] permutation equivariance, [[convolution\|convolutional]] weight sharing |
| **Learned end-to-end** | Train on a proxy task, keep the internals | [[pre-train-then-fine-tune\|Pre-training]], embeddings |

The first two are underrated relative to the third. **They do not require scale** — an architectural prior that makes the right function expressible costs nothing at inference and no data at all, whereas learning the same structure end-to-end costs however many examples the discovery requires.

[[fully-convolutional-network|Weight sharing]] is the clearest case: it is what makes dense prediction learnable from sparse supervision, and no amount of data substitutes for it.

## When Handcrafting Still Wins

The received view is that learned representations dominate given enough data. True, and the qualifier is where the interesting cases live.

> ### `handcrafted-features-rule`
> **Encode structure the representation cannot recover *and* the data cannot support learning. Encode nothing else.**
> ^[generated: a heuristic, not a finding. rests-on: imported:feature-engineering-debate]

**The two clauses are not independent**, which is the weakness worth stating. "The representation cannot recover it" is a property of the *architecture*; "the data cannot support learning it" is a property of *where on its learning curve* a flexible model sits. That gives four cases:

| | Representation **can** recover | Representation **cannot** |
|---|---|---|
| **Enough data** | Redundant → often harmful | Helpful |
| **Not enough data** | Unclear | **Essential** |

Adding a feature the architecture already expresses is not neutral — it adds parameters and correlated inputs, and can degrade performance outright.

**What would falsify the rule:** it predicts a locatable crossover, where a given engineered feature helps at small $N$ and stops helping as $N$ grows. If the curves never converge at any reachable $N$, the first clause is doing all the work and the second is decorative. A data-scaling sweep would settle it, and the rule should be treated as a working heuristic until one is run. See [[theory-based-modelling]].

## Removing Shortcuts Improves Representations

A recurring and counter-intuitive pattern: **withholding an easy signal produces a better representation of it.**

[[bert|Masked language modelling]] is the canonical case — hiding tokens forces the model to infer them from bidirectional context, and the resulting representations transfer better than those from a model given the tokens directly.

[[variational-lossy-autoencoders|VLAE]] does the same by architecture rather than by masking: restricting the decoder's receptive field means local detail *cannot* be modelled locally, so global structure is forced through the latent. The lossiness becomes a specification.

> ### `a-representation-learns-what-it-is-not-given-for-free`
> **Where a model can satisfy its objective using a shortcut feature, it will, and the representation of the underlying structure stays weak. Removing the shortcut is often a cheaper improvement than adding capacity or data.**
> ^[generated. rests-on: source:vlae-receptive-field, source:bert-masking]

The practical form: **if a feature lets the model avoid learning what you want, remove it.**

## What Counts as Good

Rarely defined explicitly, and the candidates conflict:

- **Downstream performance** — the default, and circular if there is only one downstream task
- **[[transfer-learning|Transfer]]** — does it help elsewhere? The foundation-model ambition
- **Structure** — does the geometry mean something? Do related entities sit near each other?
- **Compression** — how much can be discarded? VLAE's framing

[[interpretability]] is a fifth and is usually traded against the rest, though not always: a representation chosen to align with an existing human vocabulary can perform identically to a raw one while producing outputs in terms a practitioner already uses. **Where that happens the trade is illusory**, and it is worth checking for before assuming it.

## The Cost That Moved Rather Than Vanished

Pre-trained representations removed the need for handcrafting, and relocated the assumptions rather than eliminating them.

[[feature-engineering]] made assumptions **visible in the feature list**. Transfer makes them **invisible in the weights**, encoded by a corpus and an objective that downstream users do not inspect. See `transfer-imports-the-pretraining-corpus-assumptions` on [[transfer-learning]].

## See Also

- [[feature-engineering]] · [[theory-based-modelling]] · [[tokenization]] · [[transfer-learning]] · [[pre-train-then-fine-tune]]
- [[graph-neural-network]] · [[fully-convolutional-network]] · [[convolution]] · [[transformer]] · [[siamese-network]]
- [[variational-autoencoder]] · [[generative-model]] · [[masked-language-model]] · [[bert]] · [[interpretability]] · [[model-selection]]
- [[variational-lossy-autoencoders|VLAE Summary]] · [[bert-bidirectional-transformers|BERT Summary]]
