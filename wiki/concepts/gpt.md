---
title: "GPT"
type: concept
tags: [transformer, language-modelling, autoregressive-model, deep-learning, pre-training, transfer-learning, representation-learning, tokenization]
sources: [raw/papers/language_understanding_gpt.md]
confidence: 0.95
provenance:
  extracted: 80%
  inferred: 15%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-07
updated: 2026-08-14
---

# GPT

GPT (Generative Pre-trained Transformer; Radford et al., 2018) is a decoder-only [[transformer]] language model that established the [[pre-train-then-fine-tune]] paradigm: pre-train a left-to-right [[autoregressive-model|autoregressive LM]] on a large corpus, then fine-tune with minimal architectural changes on diverse downstream tasks.

## Architecture

A 12-layer decoder-only [[transformer]] with masked [[multi-head-attention]] (768-d states, 12 heads, 3072-d FFN inner dimension). Learned [[positional-encoding|position embeddings]], BPE [[tokenization|tokenization]] (40K merges), GELU activation, and [[layer-normalization]]. 117M parameters. Pre-trained on BooksCorpus (800M words, ~7K books).

## Pre-training

Standard left-to-right language modelling: $L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1})$. Each token attends only to its left context via masked self-attention.

## Fine-tuning

Task-specific input transformations convert structured inputs into token sequences with delimiter tokens, avoiding task-specific architectures:
- **Classification:** `[Start] Text [Extract]`
- **Entailment:** `[Start] Premise [Delim] Hypothesis [Extract]`
- **Similarity:** Both orderings processed, representations added element-wise
- **Multiple choice:** Each answer concatenated with context separately

Only one linear output layer $W_y$ is added per task. An auxiliary LM loss ($\lambda = 0.5$) during fine-tuning improves generalisation — an early instance of an auxiliary objective shaping the primary one.

## Key Results

SOTA on 9/12 benchmarks: Story Cloze +8.9% (86.5%), RACE +5.7% (59.0%), MultiNLI +1.5% (82.1%), GLUE 72.8 (+3.9).

## Key Ablation Findings

- Pre-training provides +14.8% average improvement over training from scratch.
- Transformer > LSTM: a 5.6-point average drop when replacing the Transformer with a single-layer 2048-unit LSTM.
- Each additional layer transferred improves performance, up to the full 12.
- Zero-shot performance improves steadily *during* pre-training across sentiment, Winograd, acceptability and QA — capability emerging without supervision.

## The Architecture Travels Beyond Language

The decoder-only GPT design has become a general-purpose sequence architecture, applied wherever data can be [[tokenization|tokenised]] into a discrete sequence — event streams, symbolic music, biological sequences, action traces.

The substantive point is that **the backbone typically needs no modification.** What changes is entirely outside it:

| Component | Language | A non-language domain |
|---|---|---|
| Tokens | BPE subwords | Domain-specific atomic units |
| Context | Preceding text | A conditioning block |
| Objective | Next-token cross-entropy | Often plus auxiliary heads |
| Decoding | Free sampling | Often masked to enforce validity |

The Transformer contributes a generic mechanism for modelling dependencies in token sequences; **domain adaptation happens in tokenization, conditioning, objective and decoding**, not in the architecture. That is why the same implementation serves domains with nothing else in common, and why [[tokenization]] is the consequential choice rather than an implementation detail.

The limit is data. [[scaling-laws]] describe how far the paradigm reaches for a given corpus size, and domains with corpora orders of magnitude smaller than text should not expect the same returns.^[imported]

## Impact

GPT established that left-to-right Transformer LMs could transfer effectively across diverse NLP tasks. The paradigm scaled to GPT-2, GPT-3 and GPT-4, leading to [[chain-of-thought|CoT prompting]], [[react|agentic reasoning]] and [[rlhf|RLHF alignment]]. [[bert]] later showed bidirectional pre-training via [[masked-language-model]] was superior for understanding tasks, creating the encoder-vs-decoder split.

## See Also

- [[transformer]] · [[bert]] · [[pre-train-then-fine-tune]] · [[transfer-learning]] · [[tokenization]]
- [[autoregressive-model]] · [[masked-language-model]] · [[scaling-laws]] · [[rlhf]] · [[generative-model]]
- [[representation-learning]] · [[chain-of-thought]] · [[react]] · [[language-understanding-gpt|Source Summary]]
