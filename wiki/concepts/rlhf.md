---
title: "Reinforcement Learning from Human Feedback"
type: concept
tags: [deep-learning, language-modelling, reinforcement-learning, alignment, instruction-tuning]
sources: [raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-13
updated: 2026-06-13
---

# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is an alignment technique that trains language models to follow human intent by optimising against a learned reward model derived from human preference judgments.

## Three-Step Pipeline

[[training-lm-follow-instructions-with-human-feedback|Ouyang et al. (2022)]] introduced the canonical RLHF pipeline (InstructGPT):

1. **Supervised Fine-Tuning (SFT):** Fine-tune a pretrained LM on human-written demonstrations of desired behaviour (~13K prompt–response pairs).
2. **Reward Model (RM):** Train a separate model on human preference rankings — labelers compare multiple model outputs and rank them, producing ~33K comparisons. The RM learns to predict which outputs humans prefer.
3. **Policy Optimisation (PPO):** Optimise the SFT model to maximise the reward model's score using Proximal Policy Optimization, with a KL-divergence penalty to prevent the policy from diverging too far from the SFT model.

## Key Results

- InstructGPT 1.3B is preferred by human labelers over GPT-3 175B despite being **100× smaller** — demonstrating that alignment can substitute for raw scale.
- Outputs are more helpful, less harmful, and more truthful (fewer hallucinations and toxic outputs).
- Small performance regressions on public NLP benchmarks ("alignment tax") can be mitigated by mixing pretraining data into PPO training.

## Significance

RLHF became the standard alignment technique for production LLMs, directly informing ChatGPT and subsequent instruction-following models. It addresses the gap between what LMs learn during pretraining (predicting next tokens) and what users actually want (helpful, harmless, and honest responses).

## Limitations

- Expensive: requires large-scale human annotation for both demonstrations and comparisons.
- Reward model can be gamed (reward hacking) if the policy finds outputs that score highly but don't genuinely satisfy intent.
- Human labeler agreement is imperfect, introducing noise into the reward signal.

## See Also

- [[training-lm-follow-instructions-with-human-feedback|Source Summary]]
- [[chain-of-thought]]
- [[scaling-laws]]
