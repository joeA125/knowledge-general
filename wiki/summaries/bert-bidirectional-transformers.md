---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — Source Summary"
type: summary
tags: [deep-learning, transformer, language-modelling, transfer-learning, pre-training, masked-language-model, representation-learning, weak-supervision]
sources: [raw/papers/bert-bidirectional-transformers.md]
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

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Authors:** [[jacob-devlin]], Ming-Wei Chang, Kenton Lee, Kristina Toutanova
**Affiliation:** Google AI Language ([[google-research]])
**Published:** 2019 (NAACL); arXiv:1810.04805

## Key Contribution

Pre-trains deep bidirectional representations using a [[masked-language-model]] objective. Unlike [[language-understanding-gpt|GPT's]] left-to-right pre-training, BERT jointly conditions on left *and* right context in all layers. Fine-tuning with one additional output layer achieves SOTA on 11 NLP benchmarks.

## Architecture

A multi-layer bidirectional [[transformer]] **encoder**, not decoder:

- **BERT$_\text{BASE}$:** L=12, H=768, A=12, 110M params — deliberately matched to GPT for comparison
- **BERT$_\text{LARGE}$:** L=24, H=1024, A=16, 340M params

Input: WordPiece embeddings (30K vocab) + segment embeddings + learned [[positional-encoding|position embeddings]], summed. See [[tokenization]]. Special tokens `[CLS]` for classification output and `[SEP]` for separation.

## Pre-training Tasks

### Masked Language Model

Randomly masks 15% of tokens; of those, 80% become `[MASK]`, 10% a random token, 10% unchanged. The model predicts the original from bidirectional context.

This solves a real obstacle: standard LMs must run one direction, since bidirectional conditioning would let each word **see itself**. Masking removes the shortcut and forces the representation to be built from context.

That is the same mechanism the vault records as **a representation learns what it is not given for free** — see [[representation-learning]], where [[variational-lossy-autoencoders|VLAE's]] receptive-field restriction is the architectural version of the same idea.

The 80/10/10 split is itself a mitigation: `[MASK]` never appears at fine-tuning time, so training exclusively on it would create a train/test mismatch — a mild form of the [[teacher-forcing|exposure-bias]] problem, **anticipated and priced in** rather than discovered later.

### Next Sentence Prediction

Binary: does sentence B follow A (50%) or is it random? Trained through the `[CLS]` representation. Helps QA and NLI.

Pre-training data: BooksCorpus (800M words) + English Wikipedia (2,500M). 1M steps, 128K tokens per batch, 4 days on 16–64 TPU chips.

## Key Results

| Benchmark | BERT$_\text{LARGE}$ | GPT | Prior SOTA |
|---|---|---|---|
| GLUE (overall) | **80.5** | 72.8 | 74.0 |
| MNLI | **86.7%** | 82.1% | — |
| SQuAD v1.1 (F1) | **93.2** | — | 91.7 |
| SQuAD v2.0 (F1) | **83.1** | — | 78.0 |
| SWAG | **86.3%** | 78.0% | human expert 85.0% |

## Ablation Findings

1. **Bidirectionality is crucial.** Removing MLM (left-to-right only, like GPT) costs 9.2 points on MRPC and 10.7 F1 on SQuAD.
2. **NSP matters**, though less: QNLI −3.5%, MNLI −0.5%.
3. **Model size helps monotonically**, 3→6→12→24 layers, **even on tiny datasets** (3.6K examples). Notable because it runs against the usual sample-size logic — models trained from scratch typically degrade with added capacity on small data. The difference is that BERT's capacity is *pre-trained elsewhere* and only transferred, so the small dataset never has to constrain it. See [[transfer-learning]] and `architecture-choice-tracks-data-scale-more-than-task-difficulty` on [[gated-recurrent-unit]].
4. **Feature-based use is nearly as good.** Concatenating the top four hidden layers as fixed features reaches 96.1 F1 on CoNLL NER, 0.3 behind full fine-tuning.

## Impact

BERT established bidirectional pre-training as superior for *understanding* tasks, complementing GPT's generative strengths — and the two together fixed [[pre-train-then-fine-tune]] as the dominant paradigm. Spawned RoBERTa, ALBERT, DeBERTa and DistilBERT.

The division has held: **encoder-only models for discriminative work, decoder-only for generative**, with encoder-decoder architectures retained where a genuine sequence-to-sequence mapping is needed. Which branch a domain follows is determined by whether its questions are generative — simulate what happens next — or discriminative.

## See Also

- [[bert]] · [[masked-language-model]] · [[pre-train-then-fine-tune]] · [[transformer]] · [[tokenization]]
- [[representation-learning]] · [[transfer-learning]] · [[teacher-forcing]] · [[scaling-laws]] · [[gpt]]
- [[jacob-devlin]] · [[google-research]]
- [[language-understanding-gpt|GPT Summary]] · [[scaling-neural-language-models|Scaling Laws Summary]] · [[training-lm-follow-instructions-with-human-feedback|InstructGPT Summary]]
