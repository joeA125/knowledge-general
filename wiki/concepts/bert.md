---
title: "BERT"
type: concept
tags: [transformer, language-modelling, deep-learning, masked-language-model, pre-training, transfer-learning, representation-learning]
sources: [raw/papers/bert-bidirectional-transformers.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-07
updated: 2026-07-07
---

# BERT

BERT (Bidirectional Encoder Representations from Transformers; Devlin et al., 2019) is a pretrained language representation model built on the [[transformer]] encoder. Unlike left-to-right models like [[gpt]], BERT jointly conditions on both left and right context in all layers, producing deeply bidirectional representations via the [[masked-language-model]] objective.

## Architecture

BERT uses a multi-layer bidirectional [[transformer]] encoder:
- **BERT$_\text{BASE}$:** 12 layers, 768 hidden, 12 heads, 110M params
- **BERT$_\text{LARGE}$:** 24 layers, 1024 hidden, 16 heads, 340M params

Input representation sums three embeddings: WordPiece token embeddings (30K vocab), segment embeddings (sentence A/B), and learned [[positional-encoding|position embeddings]]. Special tokens `[CLS]` (classification output) and `[SEP]` (sentence separator) structure the input.

## Pre-training

Two self-supervised tasks on BooksCorpus + English Wikipedia (3.3B words total):

1. **[[masked-language-model|Masked Language Model (MLM)]]:** 15% of tokens masked (80% `[MASK]`, 10% random, 10% unchanged). Predicts original tokens from bidirectional context.
2. **Next Sentence Prediction (NSP):** Binary classification — is sentence B the true continuation of sentence A? Trained using the `[CLS]` representation.

## Fine-tuning

The [[pre-train-then-fine-tune]] paradigm: add one linear output layer, fine-tune all parameters end-to-end. The same architecture handles single-sentence classification (`[CLS]` output), sentence-pair tasks (packed with `[SEP]`), span extraction (start/end vectors dot-producted with token representations), and sequence tagging.

## Key Results

- **GLUE:** 80.5 overall (+7.7 over prior SOTA), MNLI 86.7%
- **SQuAD v1.1:** 93.2 F1 (ensemble), surpassing human (91.2)
- **SQuAD v2.0:** 83.1 F1 (+5.1 over prior best)
- **SWAG:** 86.3% (vs GPT's 78.0%, human expert 85.0%)

## Why Bidirectionality Matters

Ablation shows removing MLM (using left-to-right like [[gpt]]) drops MRPC by 9.2 points and SQuAD F1 by 10.7 points. Adding a BiLSTM on top of LTR partially recovers SQuAD but still far underperforms bidirectional pre-training.

## Impact

BERT became the dominant paradigm for NLU (2019–2021), spawning variants (RoBERTa, ALBERT, DeBERTa, DistilBERT) and demonstrating that bidirectional pre-training is superior to unidirectional for understanding tasks.

## See Also

- [[bert-bidirectional-transformers|Source Summary]]
- [[masked-language-model]]
- [[pre-train-then-fine-tune]]
- [[gpt]]
- [[transformer]]
- [[scaling-laws]]
