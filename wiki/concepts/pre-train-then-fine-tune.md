---
title: "Pre-train then Fine-tune"
type: concept
tags: [deep-learning, transfer-learning, pre-training, language-modelling, representation-learning]
sources: [raw/papers/language_understanding_gpt.md, raw/papers/bert-bidirectional-transformers.md, raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.95
provenance:
  extracted: 80%
  inferred: 15%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-07
updated: 2026-07-07
---

# Pre-train then Fine-tune

The pre-train then fine-tune paradigm is a two-stage training recipe for neural language models: (1) pre-train a large model on a vast unlabelled corpus using a self-supervised objective (language modelling or masked language modelling), then (2) fine-tune all parameters on a smaller labelled dataset for a specific downstream task, adding only a minimal task-specific output layer.

## Historical Development

1. **[[language-understanding-gpt|GPT (Radford et al., 2018)]]:** Established the paradigm for [[transformer]]s. A 12-layer decoder-only Transformer pre-trained as a left-to-right LM on BooksCorpus, then fine-tuned with task-specific input transformations. SOTA on 9/12 benchmarks. Showed pre-training provides +14.8% average improvement over training from scratch.

2. **[[bert-bidirectional-transformers|BERT (Devlin et al., 2019)]]:** Refined the paradigm with bidirectional pre-training via [[masked-language-model]]. Used encoder-only Transformer. SOTA on 11 NLP tasks. Showed bidirectional > unidirectional for understanding tasks.

3. **[[training-lm-follow-instructions-with-human-feedback|InstructGPT (Ouyang et al., 2022)]]:** Extended the paradigm to three stages: pre-train → supervised fine-tune (SFT) → [[rlhf|RLHF]]. Aligned LLMs with human intent.

## Why It Works

- **Data efficiency:** Pre-training on billions of tokens captures syntactic, semantic, and world knowledge that would be impossible to learn from small labelled datasets alone.
- **Transfer:** GPT showed each Transformer layer learns increasingly task-relevant representations — transferring all layers gives up to 9% improvement over embeddings alone (MultiNLI).
- **Regularisation:** Pre-training acts as a regulariser (Erhan et al., 2010), enabling better generalisation even on small datasets. BERT's 340M-param model improves on tasks with only 3.6K training examples.
- **Minimal architecture changes:** Fine-tuning requires only one additional output layer, avoiding task-specific engineering.

## Two Paradigms

| Approach | Architecture | Pre-training | Strengths |
|---|---|---|---|
| GPT-style | Decoder-only | Autoregressive LM | Generation, few-shot, scaling |
| BERT-style | Encoder-only | Masked LM + NSP | Understanding, classification, extraction |

The GPT line scaled to GPT-2 → GPT-3 → GPT-4, eventually leading to [[chain-of-thought|CoT prompting]] and [[react|agentic reasoning]] at scale. The BERT line focused on efficient understanding (RoBERTa, ALBERT, DeBERTa). Both converge in modern instruction-tuned models that combine pre-training, fine-tuning, and [[rlhf|alignment]].

## See Also

- [[language-understanding-gpt|GPT Source Summary]]
- [[bert-bidirectional-transformers|BERT Source Summary]]
- [[masked-language-model]]
- [[rlhf]]
- [[scaling-laws]]
- [[transformer]]
