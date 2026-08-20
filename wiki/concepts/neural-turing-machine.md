---
title: "Neural Turing Machine"
type: concept
tags: [deep-learning, architecture, attention, external-memory, neural-computation]
sources: [raw/papers/neural-turing-machines.md]
confidence: 0.95
provenance:
  extracted: 85%
  inferred: 10%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-05-10
updated: 2026-05-10
---

# Neural Turing Machine

The Neural Turing Machine (NTM; Graves, Wayne & Danihelka, 2014) is a neural network architecture that couples a controller network to an external memory matrix via differentiable attention-based read and write heads. It is analogous to a Turing Machine but is fully differentiable and trainable with gradient descent.

## Architecture

- **Controller:** A neural network (feedforward or [[lstm]]) that receives external inputs, emits outputs, and parametrises read/write head operations.
- **Memory:** An $N \times M$ matrix. Unlike LSTM's internal state, the memory size is decoupled from the controller, scaling storage independently of computation.
- **Heads:** Read and write heads interact with memory via normalised attention weightings. Multiple heads can operate in parallel.

## Addressing

A hybrid system combining content-based and location-based addressing:

1. **Content-based:** Cosine similarity between a key vector and memory rows, scaled by key strength $\beta_t$ and normalised via softmax.
2. **Interpolation:** Gate $g_t$ blends content weighting with the previous time step's weighting.
3. **Shift:** Circular convolution with shift weighting $\mathbf{s}_t$ enables relative position changes (iteration, jumping).
4. **Sharpening:** Exponentiation by $\gamma_t \geq 1$ counteracts dispersion from repeated shifting.

## Reading and Writing

- **Read:** Weighted sum of memory rows (convex combination).
- **Write:** Erase-then-add, inspired by LSTM forget/input gates. Erase vector $\mathbf{e}_t$ and add vector $\mathbf{a}_t$ provide fine-grained control per memory element.

## Key Properties

- **Differentiable end-to-end:** All operations (reading, writing, addressing) are differentiable, enabling standard gradient-based training.
- **Algorithm learning:** NTMs learn simple algorithms (copy, sort, associative recall) from examples and generalise to inputs longer than those seen during training.
- **Working memory analogy:** The architecture mirrors cognitive models of working memory — a central executive (controller) manipulating a short-term buffer (memory) via attentional processes.

## Relation to Other Architectures

- The [[read-process-write]] architecture (Vinyals et al., 2016) can be viewed as a special case of an NTM or Memory Network.
- Memory Networks (Weston et al., 2015) use a similar external memory concept but with different addressing schemes.
- Differentiable Neural Computers (Graves et al., 2016) extended NTMs with dynamic memory allocation and temporal linking.

## See Also

- [[attention-mechanism]]
- [[lstm]]
- [[read-process-write]]
- [[neural-turing-machines|Source Summary]]
