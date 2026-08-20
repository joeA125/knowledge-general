---
title: "Shall We Pretrain Autoregressive Language Models with Retrieval? — Source Summary"
type: summary
tags: [deep-learning, RAG, language-modelling, transformer, knowledge-intensive]
sources: [raw/papers/autogressive-language-model-retrieval.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Shall We Pretrain Autoregressive Language Models with Retrieval?

**Authors:** Boxin Wang, Wei Ping, Peng Xu, et al.
**Affiliations:** University of Illinois, NVIDIA
**Published:** 2023

## Key Contribution

A comprehensive study reproducing and evaluating RETRO (Borgeaud et al., 2022) up to 9.5B parameters with a 330B-token retrieval corpus. Compares RETRO to standard GPT and retrieval-augmented GPT on text generation, factuality, toxicity, downstream tasks, and open-domain QA. Introduces RETRO++, a variant that feeds the top retrieved document into the decoder context.

## Key Findings

- **Text generation:** RETRO reduces repetition by ~21% vs GPT; comparable fluency and coherence.
- **Factuality:** Moderately higher factual accuracy on FactualityPrompts and TruthfulQA.
- **Toxicity:** RETRO with pretraining corpus slightly increases toxicity; switching to Wikipedia corpus reduces it below GPT.
- **Zero-shot benchmarks:** RETRO outperforms GPT on knowledge-intensive tasks (HellaSwag, BoolQ); on par for others.
- **Open-domain QA:** RETRO++ achieves 54.1 EM on Natural Questions (+8.6 over original RETRO, outperforming RAG-GPT at 50.9).
- **Training cost:** ~25% additional GPU hours over standard GPT pretraining.

## See Also

- [[retrieval-augmented-generation]]
- [[transformer]]
