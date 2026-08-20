---
title: "Neural Machine Translation by Jointly Learning to Align and Translate — Source Summary"
type: summary
tags: [deep-learning, machine-translation, attention, sequence-modelling, rnn, encoder-decoder-bottleneck, alignment]
sources: [raw/papers/neural-machine-translation.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Neural Machine Translation by Jointly Learning to Align and Translate

**Authors:** [[dzmitry-bahdanau]], [[kyunghyun-cho]], [[yoshua-bengio]]
**Affiliations:** [[jacobs-university-bremen]], [[universite-de-montreal]]
**Published:** 2014 (ICLR 2015); arXiv:1409.0473

## Key Contribution

This paper introduces [[additive-attention]] (commonly called "Bahdanau attention"), the first [[attention-mechanism]] for neural machine translation. It solves the [[encoder-decoder-bottleneck]] — the problem of compressing an entire source sentence into a single fixed-length vector — by allowing the decoder to dynamically attend to different parts of the source at each generation step.

## The Problem

Standard [[encoder-decoder]] NMT models (Sutskever et al., 2014; Cho et al., 2014) encode the full source into a fixed-length vector $c$. Performance degrades sharply on long sentences because the vector cannot retain all necessary information.

## Proposed Architecture (RNNsearch)

1. **Encoder:** A [[bidirectional-rnn]] (BiRNN) produces annotations $h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$ for each source position, capturing both left and right context.
2. **Alignment model:** A learned feedforward network $a(s_{i-1}, h_j)$ scores how well source position $j$ matches decoder state $s_{i-1}$.
3. **Context vector:** A position-specific weighted sum of annotations: $c_i = \sum_j \alpha_{ij} h_j$, where $\alpha_{ij} = \text{softmax}(e_{ij})$.
4. **Decoder:** A [[gated-recurrent-unit]] (GRU) RNN conditioned on the context vector $c_i$, the previous hidden state, and the previous output word.

This is a **soft alignment** — fully differentiable and jointly trained end-to-end with the translation model, unlike traditional hard alignments.

## Results (WMT '14 English-to-French)

| Model | BLEU (All) | BLEU (No UNK) |
|---|---|---|
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses (phrase-based) | 33.30 | 35.63 |

- RNNsearch-50 nearly matches Moses on known-word sentences — a major milestone for end-to-end NMT.
- RNNsearch shows no performance degradation on sentences of 50+ words, while RNNencdec collapses.
- RNNsearch-30 even outperforms RNNencdec-50.

## Qualitative Findings

- Alignment matrices show linguistically plausible soft alignments (e.g. correctly handling French-English adjective reordering).
- Soft alignment naturally handles phrase-length mismatches without needing NULL alignments.
- Long sentences are translated completely and accurately, unlike the fixed-vector encoder which omits or distorts content after ~30 words.

## Architecture Details

- Hidden units: 1000 (encoder and decoder)
- Word embeddings: 620 dimensions
- Alignment model: single-layer MLP with 1000 units
- Deep output with maxout (500 units)
- Trained with SGD + Adadelta, gradient clipping at norm 1, minibatch size 80
- Vocabulary: 30,000 words per language
- Training: ~5 days on a single GPU (TITAN BLACK / Quadro K-6000)

## Impact

This paper introduced the concept of attention to sequence-to-sequence models, directly inspiring the [[multi-head-attention]] mechanism in the [[transformer]]. It is one of the most influential papers in modern NLP, establishing attention as a fundamental building block.

## See Also

- [[additive-attention]]
- [[attention-mechanism]]
- [[encoder-decoder-bottleneck]]
- [[bidirectional-rnn]]
- [[gated-recurrent-unit]]
- [[encoder-decoder]]
- [[transformer]]
