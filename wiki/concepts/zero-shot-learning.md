---
title: "Zero-Shot Learning"
type: concept
tags: [zero-shot-learning, transfer-learning, language-modelling, evaluation, prompt-engineering, representation-learning, pre-training]
sources: [raw/papers/language_understanding_gpt.md]
confidence: 0.75
provenance:
  extracted: 30%
  inferred: 30%
  generated: 8%
  imported: 30%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Zero-Shot Learning

Performing a task with **no task-specific training examples**. The model is asked to do something it was never explicitly trained to do, using only what it acquired from a different objective.

## Where the Capability Comes From

The mechanism, as the held [[language-understanding-gpt|GPT]] work shows it: **task-relevant capability accumulates as a by-product of a broader objective.**

Zero-shot performance on sentiment, Winograd schemas, linguistic acceptability and question answering improves steadily *as language-model pre-training progresses* — with no supervised fine-tuning at any point. The model was optimised to predict the next token; the ability to classify sentiment came along with it.

That is only surprising if next-token prediction is taken to be a narrow objective. It is not: predicting text well **requires** modelling whatever the text is about, and the tasks were latent in the corpus all along.

## The Terminology Is Contested

Three regimes are routinely conflated, and the distinction matters for what a result means:

| | Task examples at inference | Parameter updates |
|---|---|---|
| **Zero-shot** | None | None |
| **Few-shot / in-context** | A handful, in the prompt | **None** |
| **Fine-tuned** | Many, in training | Yes |

Few-shot learning is often reported alongside zero-shot as though they differed in degree. **They differ in kind on one axis and not at all on another**: neither updates parameters, but only one supplies task examples.

> ### `zero-shot-claims-depend-on-what-the-pretraining-corpus-contained`
> **A task is zero-shot only relative to a corpus nobody has inspected. Where the pre-training data contains the task, or close analogues of it, the result measures retrieval rather than generalisation — and at web scale, assuming absence is a strong claim.**
> ^[generated. rests-on: source:gpt-zero-shot-progression, imported:contamination-concerns]

This is the central evaluation problem in the area, and it is not fully solvable: verifying that a benchmark does not appear in a trillion-token corpus is hard, and near-duplicates are harder still.

## The Older Sense

^[imported]

Before language models, zero-shot learning meant something more specific: recognising classes unseen in training, by mapping inputs and class labels into a **shared semantic space** — attribute vectors, word embeddings — so a new class could be described rather than exemplified.

That framing still describes what makes it work. **A model can only do zero-shot inference on a task expressible in a representation it already has.** The language-model case is the same idea with the shared space learned implicitly rather than specified.

See [[representation-learning]] and [[siamese-network]], where the same open-set property is achieved by learning a metric rather than a decision boundary.

## Why It Matters

Practically: it removes the labelled-data requirement for the long tail of tasks too small to justify collecting data for.

Methodologically: it changed what a "model" is taken to be. A fine-tuned model does one task; a model with zero-shot capability is a **general function whose task is specified at inference time**, which is what makes prompting a modelling activity rather than an interface detail.

## Limitations

- **Unreliable and hard to predict.** Zero-shot performance varies sharply with phrasing, which is a poor property for a capability claim.
- **No calibration guarantee.** A model doing a task zero-shot has no reason to be [[probability-calibration|calibrated]] on it.
- **Evaluation is contaminated by construction**, per the claim above.
- **It sets a floor, not a ceiling.** Zero-shot results are usually beaten by fine-tuning where data exists, so the interesting claim is about *breadth* rather than peak performance.

## See Also

- [[transfer-learning]] · [[pre-train-then-fine-tune]] · [[gpt]] · [[bert]] · [[representation-learning]] · [[tokenization]]
- [[scaling-laws]] · [[chain-of-thought]] · [[siamese-network]] · [[probability-calibration]] · [[predictive-validity]] · [[selection-bias]]
- [[language-understanding-gpt|GPT Summary]]
