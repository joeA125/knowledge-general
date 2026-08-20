---
title: "Chain-of-Thought Prompting"
type: concept
tags: [deep-learning, language-modelling, chain-of-thought, prompt-engineering, reasoning]
sources: [raw/papers/chain-of-thought-reasoning-llms.md, raw/papers/synergising-reasoning-acting-llms.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-13
updated: 2026-06-13
---

# Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting (Wei et al., 2022) is a technique for eliciting multi-step reasoning from large language models by augmenting few-shot exemplars with intermediate natural language reasoning steps — a "chain of thought" — before the final answer.

## Method

Standard few-shot prompting provides exemplars as ⟨input, output⟩ pairs. CoT extends these to ⟨input, chain of thought, output⟩ triples, where the chain of thought is a series of intermediate reasoning steps written in natural language. No fine-tuning is required; the technique works with frozen, off-the-shelf LLMs.

## Emergent Ability of Scale

CoT is an emergent property of model scale: it does not improve (and often hurts) performance for models smaller than ~10B parameters. Gains only appear at ~100B+ parameters. Small models produce fluent but illogical chains of thought.

## Key Properties

1. **Decomposition:** Allows models to break multi-step problems into intermediate steps, allocating more computation to harder problems.
2. **Interpretability:** The chain of thought provides a window into the model's reasoning process, enabling debugging of incorrect reasoning paths.
3. **Generality:** Applicable to any task humans solve via step-by-step reasoning — arithmetic, commonsense, symbolic reasoning, and beyond.
4. **Robustness:** Outperforms standard prompting across different annotators, exemplar sets, exemplar orderings, and language models (GPT-3, LaMDA, PaLM).

## Why It Works (Ablation Evidence)

- **Equation-only prompting:** Helps on simple problems but fails on semantically complex ones — natural language reasoning steps are necessary, not just equations.
- **Variable compute only:** Outputting filler tokens (dots) to match CoT length does not help — the semantic content of the reasoning matters.
- **Reasoning after answer:** Placing the chain of thought after the answer performs at baseline — the sequential reasoning must precede the answer.

## Limitations

- CoT does not guarantee correct reasoning; models can arrive at right answers via wrong reasoning (especially on multiple-choice tasks).
- Generated chains of thought are not always factually correct (hallucination).
- Only effective at large model scales, making it costly to deploy.
- Error analysis shows ~46% of failures are "almost correct" (calculator/symbol mapping/one-step errors), while ~54% involve deeper semantic misunderstanding.

## Extensions

- **Self-consistency (CoT-SC):** Sampling multiple CoT trajectories and taking the majority answer (Wang et al., 2022) consistently boosts performance over single-sample CoT.
- **[[synergising-reasoning-acting-llms|ReAct]]:** Interleaves CoT reasoning with external actions (e.g., Wikipedia retrieval), grounding reasoning in facts and eliminating hallucination. ReAct + CoT-SC achieves the best prompting results on HotpotQA and FEVER.
- **Zero-shot CoT:** Simply appending "Let's think step by step" to the prompt (Kojima et al., 2022).

## See Also

- [[chain-of-thought-reasoning-llms|Source Summary]]
- [[synergising-reasoning-acting-llms|ReAct Source Summary]]
- [[scaling-laws]]
- [[retrieval-augmented-generation]]
