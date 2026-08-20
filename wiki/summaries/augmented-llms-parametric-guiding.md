---
title: "PKG: Augmented LLMs with Parametric Knowledge Guiding — Source Summary"
type: summary
tags: [deep-learning, RAG, language-modelling, knowledge-intensive, instruction-tuning]
sources: [raw/papers/augmented-llms-parametric-guiding.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# PKG: Augmented Large Language Models with Parametric Knowledge Guiding

**Authors:** Ziyang Luo, Can Xu, Pu Zhao, et al.
**Affiliations:** Hong Kong Baptist University, Microsoft
**Published:** 2023

## Key Contribution

Proposes Parametric Knowledge Guiding (PKG), which uses an instruction-fine-tuned open-source "white-box" LM (LLaMA-7B) to generate domain-specific background knowledge that is then fed as context to a "black-box" LLM (InstructGPT-175B). Unlike retrieval-based methods, PKG generates knowledge parametrically, avoiding the need for an external knowledge database.

## Method

1. **Knowledge alignment:** Fine-tune LLaMA-7B on (instruction, input, background) triples from the target domain.
2. **Background-augmented prompting:** For a given input, the PKG module generates relevant background knowledge, which is prepended to the input for the black-box LLM.

## Key Results

- Factual knowledge (FM2): +7.9%
- Tabular knowledge (NQ-Table): +11.9%
- Medical knowledge (MedMC-QA): +3.0%
- Multimodal science (ScienceQA): +8.1%
- Outperforms both retrieval-based (BM25, REPLUG) and self-guiding (CoT, GenRead) baselines.
- Larger PKG modules and larger target LLMs both improve performance.

## See Also

- [[retrieval-augmented-generation]]
- [[scaling-laws]]
