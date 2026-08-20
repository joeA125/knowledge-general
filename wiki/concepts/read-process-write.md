---
title: "Read-Process-Write"
type: concept
tags: [deep-learning, architecture, attention, set-modelling, sequence-modelling, pointer-mechanism, external-memory]
sources: [raw/papers/sequence-to-sequence-sets.md, raw/papers/neural-turing-machines.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-08
updated: 2026-05-10
---

# Read-Process-Write

Read-Process-Write (RPW; Vinyals, Bengio & Kudlur, 2016) is a neural architecture for handling unordered input sets in a permutation-invariant way, extending the seq2seq framework beyond sequences.

## Architecture

1. **Read block:** Each input element $x_i$ is embedded into a memory vector $m_i$ via a shared neural network.
2. **Process block:** An [[lstm]] with no external inputs performs $T$ processing steps. At each step, it reads the memories via content-based [[attention-mechanism|attention]]:
   - Query $q_t = \text{LSTM}(q^*_{t-1})$
   - Attention weights $a_{i,t} = \text{softmax}(f(m_i, q_t))$
   - Readout $r_t = \sum_i a_{i,t} m_i$
   - Updated state $q^*_t = [q_t; r_t]$
   The final state $q^*_T$ is a permutation-invariant embedding of the input set.
3. **Write block:** A [[pointer-network]] decoder that takes $q^*_T$ as context and points at elements of the input. Extended with "glimpses" — additional attention reads interleaved between each pointer output.

## Key Properties

- **Permutation invariance:** Swapping any $m_i$ and $m_j$ does not change the readout $r_t$, since attention is content-based.
- **Scalable memory:** Unlike bag-of-words, the representation scales with the number of processing steps $T$, allowing richer computation over larger sets.
- **Relation to other architectures:** Can be viewed as a special case of [[neural-turing-machine|Neural Turing Machines]] (Graves et al., 2014) or Memory Networks (Weston et al., 2015).

## Results

On sorting $N$ numbers, RPW with 5 processing steps and glimpses achieves 94% accuracy at $N=5$ and 57% at $N=10$, significantly outperforming standard [[pointer-network]] baselines (90% and 28% respectively).

## See Also

- [[pointer-network]]
- [[neural-turing-machine]]
- [[attention-mechanism]]
- [[lstm]]
- [[sequence-to-sequence-sets|Source Summary]]
