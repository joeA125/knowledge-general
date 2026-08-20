---
title: "Retrieval-Augmented Generation"
type: concept
tags: [deep-learning, RAG, language-modelling, knowledge-intensive]
sources: [raw/papers/rag-intense-nlp-tasks.md, raw/papers/universal-prompt-retrieval-zero-shot-eval.md, raw/papers/autogressive-language-model-retrieval.md, raw/papers/autogressive-language-model-retrieval-iterative.md, raw/papers/augmented-llms-parametric-guiding.md]
confidence: 0.9
provenance:
  extracted: 60%
  inferred: 30%
  ambiguous: 10%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-06-10
---

# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) refers to a family of methods that enhance language models by incorporating external knowledge retrieved at inference time (or during training), rather than relying solely on knowledge stored in model parameters.

## Origin

[[rag-intense-nlp-tasks|Lewis et al. (2020)]] introduced the RAG framework, combining a pre-trained BART generator with Dense Passage Retrieval (DPR) over Wikipedia, jointly fine-tuned end-to-end. Two variants — RAG-Sequence (same documents for whole output) and RAG-Token (different documents per token) — set new SOTA on open-domain QA (Natural Questions, TriviaQA, CuratedTrec).

## Stages of Retrieval Integration

- **Inference time:** kNN-LM (Khandelwal et al., 2020) interpolates LM predictions with nearest-neighbour lookups.
- **Fine-tuning:** DPR, RAG, FiD train a retriever and reader jointly on downstream tasks.
- **Pretraining:** RETRO (Borgeaud et al., 2022) pretrains an autoregressive LM with chunk-wise cross-attention over retrieved passages from a trillion-token corpus. [[autogressive-language-model-retrieval|Wang et al. (2023)]] show RETRO outperforms GPT on text quality, factuality, and knowledge-intensive tasks.

## Prompt Retrieval

Rather than retrieving passages, [[universal-prompt-retrieval-zero-shot-eval|UPRISE]] retrieves task demonstrations as prompts, tuning a lightweight retriever that generalises across tasks and model scales.

## Iterative Retrieval-Generation

[[autogressive-language-model-retrieval-iterative|ITER-RETGEN]] iterates: the LLM's output from one round serves as context for retrieving better knowledge in the next round, enabling multi-hop reasoning without complex structured workflows.

## Parametric Knowledge Generation

[[augmented-llms-parametric-guiding|PKG]] replaces retrieval with a fine-tuned "white-box" LM that generates relevant background knowledge, avoiding external databases entirely.

## Reasoning + Retrieval

[[synergising-reasoning-acting-llms|ReAct]] interleaves reasoning traces with retrieval actions, grounding chain-of-thought in external knowledge. This eliminates hallucination in failure cases (0% vs 56% for CoT alone on HotpotQA).

## Key Trade-offs

- Retrieval quality depends on the corpus and retriever; noisy retrieval can hurt performance.
- Pretraining with retrieval (RETRO) costs ~25% more compute but yields consistent improvements.
- Larger retrieval corpora improve factuality but can amplify toxicity if the corpus contains toxic content.

## See Also

- [[transformer]]
- [[scaling-laws]]
- [[attention-mechanism]]
