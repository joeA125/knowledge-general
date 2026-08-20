---
title: "A Definition of AGI — Source Summary"
type: summary
tags: [AI, deep-learning, language-modelling, reasoning, cognitive-science, evaluation]
sources: [raw/papers/agi_definition.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-12
updated: 2026-05-12
---

# A Definition of AGI

**Authors:** Dan Hendrycks, Dawn Song, Christian Szegedy, Honglak Lee, Yarin Gal, Erik Brynjolfsson, Sharon Li, Andy Zou, Yoshua Bengio, et al. (30 authors)
**Affiliations:** Center for AI Safety, UC Berkeley, various universities and labs
**Published:** 2025

## Key Contribution

Proposes a quantifiable framework for defining AGI as matching the cognitive versatility and proficiency of a well-educated adult. Grounded in Cattell-Horn-Carroll (CHC) theory — the most empirically validated model of human cognition — the framework decomposes general intelligence into ten core cognitive domains, each weighted at 10%, yielding a standardised AGI Score (0–100%).

## The Ten Cognitive Domains

1. **General Knowledge (K):** Commonsense, science, social science, history, culture.
2. **Reading & Writing (RW):** Letter-word decoding, comprehension, writing, English usage.
3. **Mathematical Ability (M):** Arithmetic, algebra, geometry, probability, calculus.
4. **On-the-Spot Reasoning (R):** Deduction, induction, theory of mind, planning, adaptation.
5. **Working Memory (WM):** Textual, auditory, visual, cross-modal.
6. **Long-Term Memory Storage (MS):** Associative, meaningful, verbatim.
7. **Long-Term Memory Retrieval (MR):** Retrieval fluency and hallucination avoidance.
8. **Visual Processing (V):** Perception, generation, reasoning, spatial scanning.
9. **Auditory Processing (A):** Phonetic coding, speech recognition, voice, rhythm, musical judgment.
10. **Speed (S):** Perceptual speed, reading/writing speed, reaction times, pointer fluency.

## Key Results

| Model | K | RW | M | R | WM | MS | MR | V | A | S | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4 | 8% | 6% | 4% | 0% | 2% | 0% | 4% | 0% | 0% | 3% | **27%** |
| GPT-5 | 9% | 10% | 10% | 7% | 4% | 0% | 4% | 4% | 6% | 3% | **57%** |

## Critical Finding: "Jagged" Cognitive Profile

Current AI systems exhibit a highly uneven profile — strong in knowledge-intensive domains (K, RW, M) but with critical deficits in foundational cognitive machinery. Long-term memory storage (MS) scores 0% for both GPT-4 and GPT-5, representing the most significant bottleneck. The paper argues that workarounds like massive context windows (for MS) and RAG (for MR) are "capability contortions" that mask underlying limitations.

## Related Definitions

The paper distinguishes AGI from: Pandemic AI, Cyberwarfare AI, Self-Sustaining AI, Recursive AI, Superintelligence, and Replacement AI.

## See Also

- [[scaling-laws]]
- [[retrieval-augmented-generation]]
- [[transformer]]
