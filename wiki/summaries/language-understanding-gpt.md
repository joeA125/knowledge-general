---
title: "Improving Language Understanding by Generative Pre-Training (GPT) — Source Summary"
type: summary
tags: [deep-learning, transformer, language-modelling, transfer-learning, pre-training, autoregressive-model, representation-learning, zero-shot-learning]
sources: [raw/papers/language_understanding_gpt.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  generated: 0%
  imported: 0%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-07-07
updated: 2026-07-27
---

# Improving Language Understanding by Generative Pre-Training (GPT)

**Authors:** [[alec-radford]], Karthik Narasimhan, Tim Salimans, [[ilya-sutskever]]
**Affiliation:** [[openai]]
**Published:** 2018

## Key Contribution

Introduces the [[pre-train-then-fine-tune]] paradigm for [[transformer]]-based language understanding: a 12-layer decoder-only Transformer is pre-trained as a left-to-right language model on BooksCorpus (800M words), then fine-tuned with minimal architectural changes on downstream tasks. SOTA on 9 of 12 benchmarks, demonstrating that generative pre-training transfers to *discriminative* tasks.

## Architecture

A 12-layer decoder-only [[transformer]] with masked [[multi-head-attention]] (768-d states, 12 heads, 3072-d FFN). Learned [[positional-encoding|position embeddings]] rather than sinusoidal, BPE [[tokenization|tokenisation]] (40K merges), GELU, [[layer-normalization]]. 117M parameters.

Pre-training objective: standard left-to-right language modelling, $L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1})$ — see [[autoregressive-model]].

## Task-Specific Input Transformations

Rather than task-specific architectures, GPT converts every task into a token sequence with delimiters:

- **Classification:** `[Start] Text [Extract]`
- **Entailment:** `[Start] Premise [Delim] Hypothesis [Extract]`
- **Similarity:** both orderings processed, representations added element-wise
- **Multiple choice:** each answer concatenated with context separately

Only one new linear output layer per task. An auxiliary LM loss ($\lambda = 0.5$) during fine-tuning improves generalisation and convergence.

**The framing move is the durable part**: reshaping the *input* rather than the *model*, so one architecture serves many tasks. The same manoeuvre recurs wherever a non-language domain is reformulated as a token sequence — see [[tokenization]] for the representational cost when the underlying data is structured tuples rather than words.

## Key Results

| Task | GPT | Previous SOTA | Improvement |
|---|---|---|---|
| Story Cloze (commonsense) | **86.5%** | 77.6% | +8.9% |
| RACE (QA) | **59.0%** | 53.3% | +5.7% |
| MultiNLI (entailment) | **82.1%** | 80.6% | +1.5% |
| GLUE (overall) | **72.8** | 68.9 | +3.9 |

## Key Ablation Findings

- **Pre-training is essential** — without it, average performance drops 14.8%.
- **Transformer beats LSTM** — swapping in a single-layer 2048-unit [[lstm]] costs 5.6 points, which the authors attribute to the Transformer's structured memory aiding transfer.
- **The auxiliary LM objective helps on large datasets** (NLI, QQP) and not on smaller ones — an addition that regularises at scale can overfit below it. See [[model-selection]].
- **Layer transfer is monotone** — each additional Transformer layer transferred improves performance, up to all 12.

## Zero-Shot Behaviour

Zero-shot performance on sentiment, Winograd schemas, linguistic acceptability and QA improves steadily *as pre-training progresses*, without any supervised fine-tuning — evidence that the LM acquires task-relevant capability as a by-product of language modelling.

This is a structural observation that generalises: **a model trained to forecast produces evaluative capability as a by-product**, needing no task labels. The same pattern appears wherever a predictive model is repurposed as a measuring instrument — see [[event-prediction]] and [[counterfactual-baseline]].

## Impact

GPT established that left-to-right Transformer LMs pre-trained on large corpora transfer to diverse NLP tasks with minimal architectural change. The [[pre-train-then-fine-tune]] paradigm became the foundation for GPT-2 and GPT-3, was refined by [[bert-bidirectional-transformers|BERT]] (bidirectional pre-training via [[masked-language-model]]), and later extended by [[training-lm-follow-instructions-with-human-feedback|InstructGPT]] and [[rlhf|RLHF]]. Its scaling behaviour is the subject of [[scaling-neural-language-models|Kaplan et al.]]

## See Also

- [[gpt]] · [[pre-train-then-fine-tune]] · [[transformer]] · [[autoregressive-model]] · [[masked-language-model]]
- [[tokenization]] · [[scaling-laws]] · [[representation-learning]] · [[transfer-learning]] · [[event-prediction]]
- [[alec-radford]] · [[ilya-sutskever]] · [[openai]]
- [[bert-bidirectional-transformers|BERT Summary]] · [[scaling-neural-language-models|Scaling Laws Summary]] · [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
