---
title: "ITER-RETGEN: Iterative Retrieval-Generation Synergy — Source Summary"
type: summary
tags: [deep-learning, RAG, language-modelling, multi-hop-reasoning, chain-of-thought]
sources: [raw/papers/autogressive-language-model-retrieval-iterative.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# ITER-RETGEN: Enhancing Retrieval-Augmented LLMs with Iterative Retrieval-Generation Synergy

**Authors:** Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, et al.
**Affiliations:** Tsinghua University, Microsoft Research Asia, Microsoft Azure AI
**Published:** 2023

## Key Contribution

Proposes ITER-RETGEN, which iterates retrieval-augmented generation and generation-augmented retrieval: each iteration's LLM output provides context for retrieving more relevant knowledge in the next iteration. Simpler than interleaved approaches (ReAct, Self-Ask) while processing all retrieved knowledge as a whole.

## Method

For $T$ iterations: (1) retrieve top-$k$ paragraphs using previous generation $y_{t-1}$ concatenated with question $q$; (2) generate output $y_t$ via Chain-of-Thought prompting conditioned on all retrieved paragraphs.

## Key Results

- Outperforms ReAct, Self-Ask, and DSP on 4 of 6 datasets (HotPotQA, 2WikiMultiHopQA, Bamboogle, StrategyQA).
- 2 iterations give the greatest performance boost; fewer API calls and retrieved documents than baselines.
- Generation-augmented retrieval adaptation (distilling from a re-ranker) further improves performance.
- Better at preserving parametric knowledge when retrieval is unhelpful.

## See Also

- [[retrieval-augmented-generation]]
