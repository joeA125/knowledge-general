---
title: "Transfer Learning"
type: concept
tags: [transfer-learning, representation-learning, pre-training, machine-learning, deep-learning, domain-adaptation, evaluation]
sources: [raw/papers/bert-bidirectional-transformers.md, raw/papers/language_understanding_gpt.md]
confidence: 0.8
provenance:
  extracted: 42%
  inferred: 28%
  generated: 6%
  imported: 22%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Transfer Learning

Reusing representations learned on one task or dataset for another. The dominant paradigm in modern deep learning, and the reason most practitioners no longer train from scratch.

## The Two-Stage Structure

^[extracted from the held GPT and BERT summaries]

1. **Pre-train** on a large corpus with a self-supervised objective — next-token prediction, masked-token prediction — where labels come free from the data itself.
2. **Adapt** to the target task with far less labelled data.

The economics are what make it dominant: pre-training is expensive and done once; adaptation is cheap and done many times. **The cost of the expensive stage amortises across every downstream use**, which is why a handful of organisations pre-train and everyone else adapts.

## Modes of Adaptation

| Mode | Changes | Data needed |
|---|---|---|
| **Feature extraction** | Nothing — the encoder is frozen, a head is trained | Least |
| **Fine-tuning** | All parameters, at a low learning rate | Moderate |
| **Prompting / in-context** | Nothing at all | A few examples, at inference |

The third only became viable at scale and represents a genuine discontinuity: adaptation without gradient updates. See [[gpt]].

## What Transfers, and What Does Not

The useful question is not *whether* to transfer but *what*.

**Lower layers transfer better than upper layers.** Early representations capture generic structure — edges, syntax — while later ones specialise to the pre-training objective. This is why feature extraction from an intermediate layer often beats using the final one.^[imported]

**Transfer helps most where target data is scarce.** With abundant target data the advantage shrinks and can invert, since the pre-trained representation carries inductive biases the target task may not want.

> ### `transfer-imports-the-pretraining-corpus-assumptions`
> **A pre-trained representation encodes what its corpus contained and what its objective rewarded. Adapting it moves the modelling assumptions upstream rather than removing them — and the assumptions are now in a dataset nobody downstream inspects.**
> ^[generated. rests-on: source:bert-pretraining, source:gpt-pretraining]

That is the honest cost of the paradigm. [[feature-engineering]] made assumptions visible in the feature list; transfer makes them invisible in the weights.

## Negative Transfer

^[imported]

Transfer can hurt. Where source and target are unrelated, the pre-trained initialisation is worse than random — the model must first unlearn, and may not fully succeed.

The condition is not similarity of *task* but of *distribution and structure*. Two tasks can look alike and share little useful structure; two that look unrelated can share a great deal.

**Nothing reliably predicts negative transfer in advance**, which means the standard practice is to try it and measure — with all the [[model-selection|selection-on-validation]] hazards that implies.

## Relation to Domain Adaptation

| | Changes | Assumes |
|---|---|---|
| **Transfer learning** | The **task** | Some shared structure |
| [[domain-adaptation]] | The **distribution**, task fixed | The task is the same |

The distinction is often blurred and is worth keeping: transfer asks "can this representation serve a different job", domain adaptation asks "can this model survive different data doing the same job".

## See Also

- [[pre-train-then-fine-tune]] · [[representation-learning]] · [[feature-engineering]] · [[domain-adaptation]] · [[masked-language-model]]
- [[bert]] · [[gpt]] · [[transformer]] · [[model-selection]] · [[predictive-validity]] · [[siamese-network]]
- [[bert-bidirectional-transformers|BERT Summary]] · [[language-understanding-gpt|GPT Summary]]
