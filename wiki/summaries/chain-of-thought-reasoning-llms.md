---
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — Source Summary"
type: summary
tags: [deep-learning, language-modelling, chain-of-thought, prompt-engineering, reasoning, scaling-laws]
sources: [raw/papers/chain-of-thought-reasoning-llms.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-06-10
updated: 2026-06-18
---

# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou
**Affiliation:** [[google-brain]], [[google-research]]
**Published:** 2022 (NeurIPS 2022)

## Key Contribution

Introduces [[chain-of-thought]] (CoT) prompting — augmenting few-shot exemplars with intermediate natural language reasoning steps — which dramatically improves LLM performance on arithmetic, commonsense, and symbolic reasoning tasks. CoT is an emergent ability that only appears at ~100B+ parameters.

## Key Results

- PaLM 540B + CoT achieves **56.9%** on GSM8K (vs 17.9% standard prompting), new SOTA surpassing finetuned GPT-3 with verifier (55%).
- PaLM 540B + CoT achieves **95.4%** on sports understanding (vs 80.5% standard), surpassing unaided human (84%).
- CoT facilitates length generalisation on symbolic tasks (coin flip, letter concatenation) to sequences longer than training exemplars.

## Three Key Findings

1. **Emergent ability of scale:** CoT hurts performance for models <10B; only yields gains at ~100B+. Related to [[scaling-laws]].
2. **Larger gains on harder problems:** GSM8K (multi-step) sees 3× improvement; single-step MAWPS subsets see minimal gains.
3. **Robust to prompt variation:** Different annotators, exemplar sets, and exemplar orderings all outperform standard prompting by a large margin.

## Ablation: Why CoT Works

- Equation-only prompting helps on simple problems but not on semantically complex ones (GSM8K).
- Variable compute only (outputting dots) does not help — the natural language reasoning content matters, not just extra tokens.
- Chain of thought after the answer does not help — the sequential reasoning must occur before the answer.

## See Also

- [[chain-of-thought]]
- [[react]]
- [[rlhf]]
- [[scaling-laws]]
- [[retrieval-augmented-generation]]
