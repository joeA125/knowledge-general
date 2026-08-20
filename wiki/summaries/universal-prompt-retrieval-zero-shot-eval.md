---
title: "UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation — Source Summary"
type: summary
tags: [deep-learning, RAG, prompt-engineering, language-modelling, zero-shot-learning]
sources: [raw/papers/universal-prompt-retrieval-zero-shot-eval.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation

**Authors:** Daixuan Cheng, Shaohan Huang, Junyu Bi, et al.
**Affiliation:** Microsoft
**Published:** 2023; ACL 2023

## Key Contribution

Introduces UPRISE, a lightweight prompt retriever that automatically retrieves natural language prompts from a pre-constructed pool to improve zero-shot LLM performance. The retriever is tuned on diverse tasks with a small LLM (GPT-Neo-2.7B) but generalises across both unseen task types (cross-task) and much larger LLMs (cross-model: BLOOM-7.1B, OPT-66B, GPT3-175B).

## Method

1. **Prompt pool:** Training demonstrations from diverse task clusters.
2. **Prompt scoring:** A frozen LLM scores prompt-input pairs by task metrics; best/worst become positive/negative labels.
3. **Retriever tuning:** A bi-encoder (initialised from BERT) trained with contrastive loss (InfoNCE) to maximise similarity between inputs and effective prompts.
4. **Inference:** Top-K retrieved prompts are prepended to the task input for a frozen LLM.

## Key Results

- Cross-task gains on Reading Comprehension (+8.5%), Paraphrase Detection (+14.6%), Closed-book QA, and NLI.
- Cross-model: improvements transfer from 2.7B to 175B parameter LLMs.
- Mitigates ChatGPT hallucination on fact-checking tasks (FEVER2.0: +5%, Covid-19: +36%).
- Outperforms BM25 and SBERT retrieval baselines.

## See Also

- [[retrieval-augmented-generation]]
- [[scaling-laws]]
