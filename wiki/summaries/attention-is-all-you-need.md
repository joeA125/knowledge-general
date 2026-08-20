---
title: "Attention Is All You Need — Source Summary"
type: summary
tags: [transformer, attention, machine-translation, deep-learning, architecture]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-07
updated: 2026-06-18
---

# Attention Is All You Need

**Authors:** [[ashish-vaswani]], [[noam-shazeer]], [[niki-parmar]], [[jakob-uszkoreit]], [[llion-jones]], [[aidan-gomez]], [[lukasz-kaiser]], [[illia-polosukhin]]
**Affiliations:** [[google-brain]], [[google-research]], [[university-of-toronto]]
**Published:** 2017 (NeurIPS)

## Key Contribution

This paper introduces the [[transformer]], the first sequence transduction model built entirely on [[attention-mechanism]]s, eliminating recurrence and convolution altogether. The architecture relies on [[multi-head-attention]] and [[positional-encoding]] to process sequences in parallel, enabling dramatically faster training with superior results.

## Architecture Overview

The [[transformer]] follows an [[encoder-decoder]] structure:

- **Encoder:** 6 identical layers, each containing a [[multi-head-attention]] sub-layer and a position-wise feed-forward network (two linear transformations with ReLU), with [[residual-connections]] and [[layer-normalization]] around each.
- **Decoder:** 6 identical layers with an additional masked [[multi-head-attention]] sub-layer for [[autoregressive-model|autoregressive]] generation, plus cross-attention over encoder output.
- **Attention:** [[scaled-dot-product-attention]] computes $\text{softmax}(QK^T / \sqrt{d_k})V$. Multi-head attention runs $h=8$ parallel heads with $d_k = d_v = 64$.
- **Positional Encoding:** Sinusoidal functions inject sequence order information, allowing the model to generalise to unseen sequence lengths.
- **Embedding sharing:** Input/output embeddings and the pre-softmax linear layer share the same weight matrix.

Key hyperparameters for the base model: $d_{\text{model}} = 512$, $d_{ff} = 2048$, $N = 6$ layers, $h = 8$ heads, ~65M parameters.

## Why Self-Attention?

The paper argues self-attention is superior to recurrence and convolution on three axes:

1. **Computational complexity:** $O(n^2 \cdot d)$ per layer, faster than recurrent $O(n \cdot d^2)$ when $n < d$ (typical for NLP).
2. **Parallelisation:** $O(1)$ sequential operations vs $O(n)$ for RNNs.
3. **Maximum path length:** $O(1)$ between any two positions, making long-range dependencies easier to learn.

## Results

- **English-to-German (WMT 2014):** 28.4 BLEU (new SOTA, +2 BLEU over best ensemble).
- **English-to-French (WMT 2014):** 41.8 BLEU (new single-model SOTA), trained in 3.5 days on 8 P100 GPUs.
- **English constituency parsing:** 92.7 F1 (semi-supervised), competitive with task-specific models despite no task-specific tuning.

## Training Details

- Optimizer: Adam with custom warmup schedule ($\text{warmup\_steps} = 4000$).
- Regularisation: [[dropout]] ($P_{drop} = 0.1$), [[label-smoothing]] ($\epsilon_{ls} = 0.1$).
- Hardware: 8 NVIDIA P100 GPUs; base model trained in ~12 hours (100K steps).

## Model Variations (Table 3 Findings)

- Single attention head hurts performance (−0.9 BLEU); too many heads (32) also degrades slightly.
- Reducing $d_k$ hurts quality, suggesting dot-product compatibility needs sufficient dimensionality.
- Bigger models perform better; [[dropout]] is essential for avoiding overfitting.
- Learned positional embeddings perform nearly identically to sinusoidal encodings.

## Impact

This paper is one of the most cited in deep learning history. The [[transformer]] architecture became the foundation for BERT, GPT, and virtually all modern large language models, fundamentally reshaping NLP and eventually computer vision, speech, and other domains.

## See Also

- [[transformer]]
- [[attention-mechanism]]
- [[multi-head-attention]]
- [[scaled-dot-product-attention]]
- [[positional-encoding]]
- [[encoder-decoder]]
- [[residual-connections]]
- [[layer-normalization]]
- [[label-smoothing]]
- [[scaling-laws]]
