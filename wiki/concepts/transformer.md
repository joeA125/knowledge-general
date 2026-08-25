---
title: "Transformer"
type: concept
tags: [transformer, architecture, deep-learning, attention, sequence-modelling, machine-translation, language-modelling]
sources: [raw/papers/attention-is-all-you-need.md]
confidence: 0.9
provenance:
  extracted: 58%
  inferred: 22%
  generated: 5%
  imported: 13%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-08-14
updated: 2026-08-14
---

# Transformer

A sequence transduction architecture built entirely on [[attention-mechanism|attention]], dispensing with recurrence and convolution ([[attention-is-all-you-need|Vaswani et al., 2017]]).

It is the substrate for nearly everything else in this vault's language-model material, and the reason the [[attention-mechanism]] cluster exists.

## Architecture

An [[encoder-decoder]] stack, six identical layers each:

| Component | Encoder layer | Decoder layer |
|---|---|---|
| Self-attention | [[multi-head-attention]] | **Masked** multi-head attention |
| Cross-attention | — | Over encoder output |
| Position-wise | [[feed-forward-network]] | [[feed-forward-network]] |
| Wrapping | [[residual-connections]] + [[layer-normalization]] | Same |

Attention is [[scaled-dot-product-attention]]: $\text{softmax}(QK^T/\sqrt{d_k})V$, run across $h=8$ heads with $d_k = d_v = 64$. Order is supplied by [[positional-encoding]], since attention is otherwise permutation-invariant. Input and output embeddings share weights with the pre-softmax linear layer.

Base model: $d_{\text{model}} = 512$, $d_{ff} = 2048$, ~65M parameters.

The decoder mask is what makes generation [[autoregressive-model|autoregressive]] — position $i$ may attend only to positions $< i$, so the model cannot see the token it is predicting.

## The Argument for Self-Attention

Three axes, all from the paper:

| | Self-attention | [[recurrence\|Recurrent]] | [[convolution\|Convolutional]] |
|---|---|---|---|
| Complexity per layer | $O(n^2 \cdot d)$ | $O(n \cdot d^2)$ | $O(k \cdot n \cdot d^2)$ |
| Sequential operations | **$O(1)$** | $O(n)$ | $O(1)$ |
| Path length between positions | **$O(1)$** | $O(n)$ | $O(\log_k n)$ |

**Self-attention wins when $n < d$** — sequences shorter than the model dimension, which held for the sentence-level NLP of the time. The $O(n^2)$ term is what later made long-context work expensive and drove the efficient-attention literature.^[imported: post-2017 development, beyond the held source]

The path-length row is the one that matters most for learning. Any two positions connect in one operation, so a gradient between distant tokens does not decay through intermediate steps — the problem that [[lstm]] gating and [[encoder-decoder-bottleneck|attention over a bottleneck]] were both invented to address.

## Results

| Task | Result |
|---|---|
| WMT14 English–German | **28.4 BLEU** — +2 over the best prior ensemble |
| WMT14 English–French | **41.8 BLEU** single model, 3.5 days on 8 P100s |
| English constituency parsing | 92.7 F1 semi-supervised, with no task-specific tuning |

The parsing result is worth noting: a model designed for translation transferred to a structurally different task without architectural change, which prefigured the general-purpose direction the architecture took.

## Ablations Worth Retaining

- **A single attention head costs 0.9 BLEU.** Too many (32) also degrades slightly — the benefit is representational diversity, not head count.
- **Reducing $d_k$ hurts**, suggesting dot-product compatibility needs sufficient dimensionality to be discriminative.
- **Learned positional embeddings perform nearly identically to sinusoidal ones.** The sinusoidal choice was made for extrapolation to unseen lengths, not accuracy.
- [[dropout]] at $P_{drop} = 0.1$ is essential; [[label-smoothing]] at $\epsilon_{ls} = 0.1$ hurts perplexity while helping BLEU and accuracy.

Training used [[adam-optimizer|Adam]] with a warmup-then-decay schedule ($\text{warmup} = 4000$ steps) — a detail that turned out to matter more than its brief mention suggests, since Transformer training is notably sensitive to the schedule.^[imported: widely reported subsequently; not established by the held source]

## What Followed

The architecture split into three lineages, all held here:

| Lineage | Uses | Example |
|---|---|---|
| **Encoder-only** | Bidirectional representation | [[bert]] — [[masked-language-model|masked LM]] pretraining |
| **Decoder-only** | Autoregressive generation | [[gpt]] — next-token prediction |
| **Encoder-decoder** | Sequence-to-sequence | The original; translation, summarisation |

[[scaling-laws|Scaling behaviour]] proved unusually clean — performance follows power laws in parameters, data and compute across many orders of magnitude, which is what made the scale-up strategy rational rather than speculative.

## Permutation Invariance Is a Feature, Not Only a Bug

Positional encoding is usually presented as a patch: attention has no notion of order, so order must be injected.

The inverse reading matters. **Self-attention over a set with no positional encoding is [[message-passing]] on a fully-connected graph**, with attention-weighted rather than summed aggregation. That places the Transformer in the same family as graph neural networks, and explains why the two literatures keep arriving at similar constructions from opposite directions.

Wherever the input genuinely has no canonical order — sets, graphs, collections of interacting entities — the property that positional encoding exists to remove is the one you want to keep.

## Limitations

- **Quadratic attention cost** in sequence length, the central constraint on context windows.^[imported]
- **No inductive bias toward locality or order.** Everything must be learned, which is why Transformers are data-hungry relative to convolutional or recurrent models at small scale. See [[regularization]].
- **Fixed context.** The architecture has no persistent state across sequences; anything beyond the window must be supplied by [[retrieval-augmented-generation|retrieval]] or re-encoded.

## See Also

- [[attention-mechanism]] · [[multi-head-attention]] · [[scaled-dot-product-attention]] · [[positional-encoding]] · [[encoder-decoder]] · [[feed-forward-network]]
- [[residual-connections]] · [[layer-normalization]] · [[dropout]] · [[label-smoothing]] · [[adam-optimizer]] · [[regularization]]
- [[recurrence]] · [[lstm]] · [[gated-recurrent-unit]] · [[encoder-decoder-bottleneck]] · [[additive-attention]] · [[convolution]]
- [[bert]] · [[gpt]] · [[masked-language-model]] · [[autoregressive-model]] · [[pre-train-then-fine-tune]] · [[scaling-laws]] · [[message-passing]]
- [[ashish-vaswani]] · [[noam-shazeer]] · [[niki-parmar]] · [[jakob-uszkoreit]] · [[llion-jones]] · [[aidan-gomez]] · [[lukasz-kaiser]] · [[illia-polosukhin]]
- [[google-brain]] · [[google-research]] · [[university-of-toronto]]
- [[attention-is-all-you-need|Source Summary]] · [[bert-bidirectional-transformers|BERT Summary]] · [[language-understanding-gpt|GPT Summary]] · [[scaling-neural-language-models|Scaling Laws Summary]]
