---
title: "ReAct"
type: concept
tags: [deep-learning, language-modelling, reasoning, RAG, prompt-engineering, chain-of-thought]
sources: [raw/papers/synergising-reasoning-acting-llms.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-13
updated: 2026-06-13
---

# ReAct

ReAct (Yao et al., 2023) is a prompting paradigm that synergises reasoning and acting in large language models by interleaving verbal reasoning traces ("thoughts") with task-specific actions in an agentic loop. The model reasons to plan actions, and acts to gather information that informs further reasoning.

## Method

The agent's action space is augmented with a language space: $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$. Actions in $\mathcal{L}$ (thoughts) do not affect the environment but update the reasoning context. Few-shot exemplars contain interleaved Thought–Action–Observation triples, and the model generates both reasoning and actions autoregressively.

## How It Differs from Chain-of-Thought

[[chain-of-thought|CoT]] generates reasoning traces in a single pass with no external interaction — it is a "static black box" grounded only in the model's internal knowledge. ReAct grounds reasoning in external observations (e.g., Wikipedia API results), enabling the model to correct itself, handle exceptions, and access up-to-date information.

## Key Properties

1. **Grounded reasoning:** ReAct eliminates hallucination in failure cases (0% vs 56% for CoT on HotpotQA), because reasoning is anchored to retrieved facts.
2. **Flexible thought types:** Thoughts can decompose goals, track progress, inject commonsense, handle exceptions, reformulate searches, and synthesise answers.
3. **Sparse or dense:** For reasoning-heavy tasks (QA), thoughts alternate with every action; for decision-making tasks (ALFWorld), thoughts appear sparsely at key decision points.
4. **Human-editable:** Unlike RL policies, ReAct trajectories can be corrected on the fly by editing thoughts, enabling human-in-the-loop control.

## Key Results

- **HotpotQA:** ReAct → CoT-SC achieves 35.1 EM (best prompting method).
- **FEVER:** CoT-SC → ReAct achieves 64.6 accuracy.
- **ALFWorld:** 71% success with 1–2 examples (vs 37% for imitation learning trained on 10⁵ trajectories).
- **WebShop:** 40% success rate (vs 28.7% for IL+RL).
- **Finetuning:** With just 3K examples, PaLM-8B finetuned with ReAct outperforms all PaLM-62B prompting methods.

## Complementarity with CoT

ReAct and CoT have complementary failure modes: CoT hallucinates facts but reasons fluently; ReAct is grounded but can get stuck in repetitive loops. Combining them (ReAct ↔ CoT-SC fallback) outperforms either alone across tasks.

## See Also

- [[synergising-reasoning-acting-llms|Source Summary]]
- [[chain-of-thought]]
- [[retrieval-augmented-generation]]
