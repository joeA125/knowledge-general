---
title: "Masked Language Model"
type: concept
tags: [deep-learning, transformer, language-modelling, masked-language-model, pre-training, representation-learning]
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

# Masked Language Model

A masked language model (MLM) is a self-supervised pre-training objective that randomly masks tokens in the input and trains the model to predict the original tokens from their bidirectional context. It was introduced by [[bert-bidirectional-transformers|BERT (Devlin et al., 2019)]], inspired by the Cloze task (Taylor, 1953).

## Why Masking Is Necessary

Standard (autoregressive) language models factorise left-to-right: $p(x_1, \ldots, x_n) = \prod_i p(x_i | x_{<i})$. This is inherently unidirectional — each token only sees its left context. Naively conditioning on both left and right context would let the model trivially "see itself" through the multi-layer [[transformer]]'s self-attention. Masking breaks this information leak, enabling deep bidirectional representations.

## BERT's Masking Procedure

For each input sequence, 15% of tokens are selected for prediction. Of these:
- **80%** are replaced with a special `[MASK]` token
- **10%** are replaced with a random token from the vocabulary
- **10%** are left unchanged

The mixed strategy reduces the pre-train/fine-tune mismatch (since `[MASK]` never appears during fine-tuning). The model predicts the original token via cross-entropy loss on the masked positions only.

## Trade-offs vs Autoregressive Pre-training

| Dimension | MLM (BERT) | Autoregressive LM ([[language-understanding-gpt|GPT]]) |
|---|---|---|
| Context | Bidirectional (full) | Unidirectional (left-only) |
| Training signal | 15% of tokens per sequence | 100% of tokens |
| Convergence | Slower (fewer predictions per batch) | Faster |
| Fine-tuning tasks | Understanding (NLI, QA, NER) | Generation (text, code, dialogue) |
| Architecture | Encoder-only | Decoder-only |

BERT's ablation shows MLM converges marginally slower but achieves substantially higher accuracy than LTR pre-training on understanding tasks (e.g., +10.7 F1 on SQuAD, +9.2% on MRPC).

## Variants

- **RoBERTa** (Liu et al., 2019): Dynamic masking (new mask each epoch), removes NSP, trains longer.
- **SpanBERT** (Joshi et al., 2020): Masks contiguous spans rather than random tokens.
- **ALBERT** (Lan et al., 2020): Adds sentence-order prediction (SOP) instead of NSP.
- **DeBERTa** (He et al., 2021): Disentangled attention with separate content and position embeddings.

## See Also

- [[bert-bidirectional-transformers|BERT Source Summary]]
- [[pre-train-then-fine-tune]]
- [[transformer]]
- [[autoregressive-model]]
