---
title: "Tokenization"
type: concept
tags: [tokenization, representation-learning, language-modelling, sequence-modelling, machine-learning, entity-embedding]
sources: [raw/papers/language_understanding_gpt.md, raw/papers/bert-bidirectional-transformers.md]
confidence: 0.8
provenance:
  extracted: 45%
  inferred: 25%
  generated: 8%
  imported: 20%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Tokenization

Converting raw input into the discrete units a sequence model operates on. The first modelling decision in any language pipeline, and one that constrains everything after it.

## The Granularity Trade

| Unit | Vocabulary | Sequence length | Unseen input |
|---|---|---|---|
| Character | Tiny | **Very long** | Always representable |
| **Subword** | Moderate | Moderate | **Decomposed into known pieces** |
| Word | Huge | Short | **Out-of-vocabulary** |

Subword schemes dominate because they resolve the trade rather than picking a side: a fixed vocabulary that never fails on unseen input, because anything unknown decomposes into known parts.

## The Two Subword Schemes

^[extracted from the held GPT and BERT summaries]

| | Merges by | Used by |
|---|---|---|
| **Byte-pair encoding** | Raw frequency of adjacent pairs | [[gpt]], 40K merges |
| **WordPiece** | Likelihood gain from merging | [[bert]], 30K vocabulary |

Both start from characters and merge iteratively; they differ only in the merge criterion. WordPiece's likelihood objective favours merges that improve the language model rather than merges that are simply common — a subtler criterion with similar results in practice.

## Why It Constrains What Follows

Three consequences, all easy to overlook:

**Sequence length is a tokenizer output.** Attention cost is quadratic in tokens, so tokenizer granularity directly sets compute. A tokenizer producing 30% more tokens costs ~70% more attention.

**The vocabulary is a fixed embedding table.** Every token gets a learned vector; anything outside the vocabulary has none. The tokenizer decides what the model can have an opinion about.

**Tokenization is not neutral across inputs.** Text in languages under-represented in the training corpus fragments into more tokens, costing more compute and more context for the same content.^[imported: widely documented; not established by the held sources]

> ### `tokenization-decisions-are-invisible-downstream`
> **The tokenizer is fixed before training and rarely revisited, so its choices appear later as properties of the model — context limits, per-language cost, brittleness on rare strings — with nothing in the model's behaviour indicating that the tokenizer is the cause.**
> ^[generated. rests-on: source:gpt-bpe, source:bert-wordpiece]

## Beyond Text

The tokenizer's job — turn continuous or structured input into a discrete vocabulary — generalises to any sequence a Transformer might consume. Images become patch tokens; audio becomes frame tokens; structured event streams become event tokens.

The transferable question is the same in every case: **what is the unit, and what does choosing it make inexpressible?** A vocabulary that cannot distinguish two situations guarantees the model cannot either.

## Limitations

- **Fixed at training time.** Changing the tokenizer means retraining.
- **Merge tables encode a corpus.** A tokenizer built on one distribution is inefficient on another.
- **Token boundaries are not semantic boundaries**, which is the source of a class of brittleness on numbers, code and rare strings.

## See Also

- [[representation-learning]] · [[feature-engineering]] · [[transformer]] · [[gpt]] · [[bert]] · [[masked-language-model]]
- [[language-understanding-gpt|GPT Summary]] · [[bert-bidirectional-transformers|BERT Summary]]
