---
title: "Order Matters: Sequence to Sequence for Sets — Source Summary"
type: summary
tags: [deep-learning, sequence-modelling, attention, pointer-mechanism, set-modelling, ordering, lstm, language-modelling]
sources: [raw/papers/sequence-to-sequence-sets.md]
confidence: 0.95
provenance:
  extracted: 90%
  inferred: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-08
---

# Order Matters: Sequence to Sequence for Sets

**Authors:** [[oriol-vinyals]], Samy Bengio, Manjunath Kudlur
**Affiliation:** [[google-brain]]
**Published:** 2016 (ICLR 2016); arXiv:1511.06391

## Key Contributions

This paper makes two central claims: (1) the order in which data is presented to or produced by seq2seq models matters significantly for performance, and (2) it introduces the [[read-process-write]] architecture for handling input sets and proposes an order-search training loss for output sets.

## Order Matters — Empirical Evidence

### Input Order
- Reversing English input sentences improved machine translation by 5.0 BLEU (Sutskever et al., 2014).
- Sorting input points by angle simplified convex hull computation from $O(n \log n)$ to $O(n)$, improving accuracy by up to 10%.
- For constituency parsing, reversing the English sentence improved F1 by 0.5%.

### Output Order
- **Language modelling (PTB):** Natural word order and reversed order both achieve 86 perplexity; a 3-word reversal scramble degrades to 96 perplexity.
- **Parsing:** Depth-first tree traversal achieves 89.5% F1 vs 81.5% for breadth-first.
- **Combinatorial problems:** Restricting output equivalence classes (e.g., starting from the lowest-indexed city, counter-clockwise) improves accuracy by 5%+ over arbitrary orderings.
- **Graphical models:** When the head variable of a star graph is output first, the LSTM learns the joint distribution more easily, especially with limited data.

## Read-Process-Write Architecture

A three-component model for handling input sets in a permutation-invariant way:

1. **Read block:** Embeds each input element $x_i$ into a memory vector $m_i$ via a shared neural network.
2. **Process block:** An [[lstm]] with no inputs or outputs performs $T$ attention-based processing steps over the memories, building up a permutation-invariant representation $q^*_T$. Uses content-based [[attention-mechanism|attention]] (Equations 3–7).
3. **Write block:** A [[pointer-network]] decoder that takes $q^*_T$ as context and points at input elements to produce the output sequence. Extended with "glimpses" — additional attention reads interleaved with pointer outputs.

This can be viewed as a special case of a Neural Turing Machine or Memory Network.

## Sorting Experiment Results

| $N$ | Ptr-Net (no glimpse) | Ptr-Net (glimpse) | RPW $P=5$ (no glimpse) | RPW $P=5$ (glimpse) |
|---|---|---|---|---|
| 5 | 81% | 90% | 88% | 94% |
| 10 | 8% | 28% | 17% | 57% |
| 15 | 0% | 4% | 2% | 4% |

Processing steps and glimpses both significantly improve performance over the baseline [[pointer-network]].

## Finding Optimal Output Orderings

For output sets where no natural order exists, the paper proposes maximising over orderings during training:

$$\theta^\star = \arg \max_\theta \sum_i \max_{\pi(X_i)} \log p(Y_{\pi(X_i)}|X_i; \theta)$$

Since exact search over $n!$ orderings is intractable, they use: (1) pretraining with uniform prior over orderings for 1000 steps, then (2) sampling $\pi(X)$ proportional to $p(Y_{\pi(X)}|X)$ via ancestral sampling ($O(1)$ cost). On 5-gram language modelling, this procedure automatically discovers natural/reversed orderings and achieves optimal perplexity of 225.

## See Also

- [[read-process-write]]
- [[pointer-network]]
- [[attention-mechanism]]
- [[lstm]]
