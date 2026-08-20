---
title: "PCC: Fact-Checking with LLMs via Probabilistic Certainty and Consistency — Source Summary"
type: summary
tags: [deep-learning, language-modelling, RAG, knowledge-intensive, uncertainty-quantification, fact-checking]
sources: [raw/papers/llm_factcheck_consistency_certainty.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-12
updated: 2026-05-12
---

# PCC: Fact-Checking with LLMs via Probabilistic Certainty and Consistency

**Authors:** Haoran Wang, Maryam Khalid, Qiong Wu, Jian Gao, Cheng Cao
**Affiliations:** Emory University, Amazon
**Published:** 2026; arXiv:2601.02574

## Key Contribution

Introduces Probabilistic Certainty and Consistency (PCC), a framework for estimating an LLM's factual confidence along two complementary dimensions: **internal certainty** (log-probability margin of the verdict token) and **reasoning consistency** (NLI-based contradiction score between rationales generated under opposing assumptions). PCC enables an adaptive fact-checking pipeline that routes claims to different verification strategies based on the model's confidence profile.

## Method

### Internal Certainty ($\tau$)
Measures how strongly the output distribution favours one verdict (True/False) over the other. If the top two tokens agree on the same verdict class, certainty is maximal; otherwise, $\tau = |p_T - p_F|$.

### Reasoning Consistency ($\gamma$)
Elicits rationales under both "assume true" and "assume false" framings, then measures cross-assumption contradiction using an NLI model. $\gamma = 1 - \bar{\phi}_{contr}$, where higher values indicate more coherent reasoning.

### Adaptive Verification (Four Quadrants)
1. **High $\tau$, High $\gamma$:** Direct answer (no retrieval needed).
2. **High $\tau$, Low $\gamma$:** Targeted search using contradiction-derived queries (potential hallucination).
3. **Low $\tau$, High $\gamma$:** Reflection-guided retrieval (incomplete knowledge).
4. **Low $\tau$, Low $\gamma$:** Deep search with iterative retrieval and self-reflection (high risk).

## Key Results

- PCC consistently achieves lower Expected Calibration Error (ECE) than verbal confidence across GPT-4o, GPT-4o-mini, Gemini-2.5, and Mistral-7B.
- On HoVER (multi-hop fact-checking), PCC improves F1 for False claims by 15.2% over the strongest baseline (FIRE).
- PCC-guided GPT-3.5-turbo nearly matches GPT-4.1 on FeLMWk False claims (0.66 vs 0.68), while FIRE shows a much larger gap (0.23 vs 0.65).
- Targeted search outperforms deep search in the high-certainty/low-consistency quadrant; deep search is better in the low-certainty/high-consistency quadrant — confirming PCC guides not just *whether* to retrieve but *how*.

## See Also

- [[retrieval-augmented-generation]]
- [[scaling-laws]]
