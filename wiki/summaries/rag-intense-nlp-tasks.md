---
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Source Summary"
type: summary
tags: [deep-learning, language-modelling, RAG, knowledge-intensive, transformer]
sources: [raw/papers/rag-intense-nlp-tasks.md]
confidence: 0.9
provenance:
  extracted: 80%
  inferred: 15%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-06-10
updated: 2026-06-18
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
**Affiliations:** Facebook AI Research, University College London, New York University
**Published:** 2020 (NeurIPS 2020)

## Key Contribution

Introduces RAG ([[retrieval-augmented-generation]]), the first general-purpose fine-tuning approach that combines a pre-trained parametric model (BART) with a non-parametric memory (Dense Passage Retrieval over Wikipedia) for knowledge-intensive NLP tasks. The retriever and generator are jointly fine-tuned end-to-end.

## Two RAG Variants

- **RAG-Sequence:** Same retrieved documents condition the entire generated sequence. Marginalises over documents per sequence: $p(y|x) \approx \sum_z p(z|x) p(y|x,z)$.
- **RAG-Token:** Different documents can be retrieved per generated token. Marginalises over documents per token: $p(y_i|x, y_{1:i-1}) \approx \sum_z p(z|x) p(y_i|x, z, y_{1:i-1})$.

## Key Results

- **Open-domain QA:** RAG sets new SOTA on Natural Questions (44.5 EM), TriviaQA (56.1 EM), and CuratedTrec (52.5 EM) — outperforming extractive models and closed-book generators.
- **Abstractive QA (MSMARCO):** RAG generates more factual, specific, and diverse answers than BART alone.
- **Jeopardy question generation:** RAG-Token generates more factual questions (highest human eval) while maintaining specificity.
- **Fact verification (FEVER):** RAG achieves 4.75% state-of-the-art within 4.3% of pipeline models that use intermediate IR steps.
- **Knowledge updating:** Swapping the retrieval index updates RAG's knowledge without retraining.

## Impact

The RAG paper established the paradigm of combining retrieval with generation that became foundational to modern [[retrieval-augmented-generation]] systems, influencing RETRO, REALM, and the broader RAG ecosystem. Later work extended RAG with [[chain-of-thought|chain-of-thought reasoning]] ([[react|ReAct]]) and iterative retrieval-generation loops ([[autogressive-language-model-retrieval-iterative|ITER-RETGEN]]).

## See Also

- [[retrieval-augmented-generation]]
- [[react]]
- [[chain-of-thought]]
- [[transformer]]
- [[scaling-laws]]
