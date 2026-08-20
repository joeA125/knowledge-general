---
title: "ReAct: Synergizing Reasoning and Acting in Language Models — Source Summary"
type: summary
tags: [deep-learning, language-modelling, chain-of-thought, reasoning, RAG, prompt-engineering]
sources: [raw/papers/synergising-reasoning-acting-llms.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-06-10
updated: 2026-06-18
---

# ReAct: Synergizing Reasoning and Acting in Language Models

**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
**Affiliations:** Princeton University, [[google-research]], [[google-brain]]
**Published:** 2023 (ICLR 2023)

## Key Contribution

Introduces [[react]], a prompting paradigm that interleaves reasoning traces (thoughts) and task-specific actions, allowing LLMs to dynamically reason about action plans while interacting with external environments (e.g., Wikipedia API). Reasoning traces help the model track progress and handle exceptions; actions allow it to gather information and ground its reasoning.

## Method

The agent's action space is augmented: $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$, where $\mathcal{L}$ is the space of language. "Thoughts" are actions in language space that don't affect the environment but update the reasoning context. Few-shot exemplars contain interleaved Thought–Action–Observation triples.

## Key Results

- **HotpotQA:** ReAct → CoT-SC achieves **35.1** EM (best prompting method), vs 29.4 for [[chain-of-thought|CoT]] alone.
- **FEVER:** CoT-SC → ReAct achieves **64.6** accuracy, vs 60.4 for CoT-SC alone.
- **ALFWorld:** ReAct achieves **71%** success rate (vs 45% Act-only, 37% BUTLER imitation learning), with only 1–2 in-context examples.
- **WebShop:** ReAct achieves **40%** success rate (vs 30.1% Act-only, 28.7% IL+RL).

## Key Insights

- CoT hallucinates facts 56% of the time on HotpotQA failures; ReAct has **0% hallucination** in failure cases (grounded by external retrieval).
- ReAct's main failure mode is reasoning errors (47%), often from repetitive loops.
- Combining ReAct + CoT-SC outperforms either alone: internal knowledge (CoT) and external knowledge (ReAct) are complementary.
- Finetuning with just 3K examples makes ReAct the best method, with PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods.

## See Also

- [[react]]
- [[chain-of-thought]]
- [[retrieval-augmented-generation]]
- [[scaling-laws]]
