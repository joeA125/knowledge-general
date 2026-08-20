---
title: "Training Language Models to Follow Instructions with Human Feedback — Source Summary"
type: summary
tags: [deep-learning, language-modelling, instruction-tuning, reinforcement-learning, alignment]
sources: [raw/papers/training-lm-follow-instructions-with-human-feedback.md]
confidence: 0.9
provenance:
  extracted: 80%
  inferred: 15%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-10
updated: 2026-06-18
---

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
**Affiliation:** [[openai]]
**Published:** 2022 (NeurIPS 2022)

## Key Contribution

Introduces InstructGPT, which aligns GPT-3 with human intent using the [[rlhf]] (Reinforcement Learning from Human Feedback) pipeline: (1) supervised fine-tuning (SFT) on human-written demonstrations, (2) training a reward model (RM) on human preference rankings, and (3) optimising the SFT model against the RM using Proximal Policy Optimization (PPO).

## Method (Three Steps)

1. **SFT:** Fine-tune GPT-3 on ~13K prompt–demonstration pairs written by human labelers.
2. **Reward Model:** Train a 6B model on ~33K comparisons of model outputs ranked by humans.
3. **RLHF (PPO):** Optimise the SFT policy to maximise the reward model's score, with a KL penalty to prevent divergence from the SFT model.

## Key Results

- InstructGPT 1.3B is preferred by human labelers over GPT-3 175B despite being 100× smaller.
- InstructGPT produces fewer hallucinations, fewer toxic outputs, and better follows user intent.
- On public NLP benchmarks, InstructGPT shows small performance regressions ("alignment tax") which can be mitigated by mixing pretraining data into PPO training.

## Impact

InstructGPT / RLHF became the standard alignment technique for LLMs, directly informing ChatGPT and subsequent instruction-following models across the industry.

## See Also

- [[rlhf]]
- [[chain-of-thought]]
- [[react]]
- [[scaling-laws]]
- [[retrieval-augmented-generation]]
