---
title: "Message Passing"
type: concept
tags: [message-passing, inference, probabilistic-graphical-model, graph-neural-network, approximation, deep-learning, set-modelling]
sources: [raw/papers/bayesian-true-skill-rating.md]
confidence: 0.75
provenance:
  extracted: 25%
  inferred: 40%
  generated: 8%
  imported: 25%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-27
updated: 2026-08-14
---

# Message Passing

Computing something about a graph by having nodes and edges exchange local information iteratively, until the local computations collectively answer a global question.

**Two traditions share the name and are rarely discussed together.** They are the same computational pattern applied to different objects.

## Tradition 1: Inference on Graphical Models

Given a distribution factorised over a graph, compute marginals by passing messages between variable and factor nodes. Each message summarises everything one part of the graph knows about a variable; on a tree the algorithm is exact.

[[trueskill]] is the held instance: the graph encodes skills, per-game performances, team sums and outcome comparisons, and marginal skill beliefs fall out of the message schedule. Because the comparison factors are non-Gaussian, exact messages are intractable and [[expectation-propagation]] approximates them by moment matching.

| | |
|---|---|
| **Passed** | Probability distributions, or their sufficient statistics |
| **Computed** | Marginals of a **known** distribution |
| **Learned** | Nothing — structure and factors are specified |

## Tradition 2: Representation Learning on Graphs

Given a graph of entities, compute node representations by passing **learned** messages along edges. In a [[graph-neural-network|GNN]]:

$$e_{(k,j)} = f_e([v_k, v_j]), \qquad o_k = f_v\Big(\sum_{j \in N(k)} e_{(k,j)}\Big)$$

| | |
|---|---|
| **Passed** | Learned feature vectors |
| **Computed** | Representations for a downstream task |
| **Learned** | The message functions themselves |

## What They Share

**Locality.** Only neighbours communicate directly, so global structure emerges from repeated local operations. This is what makes both tractable on large graphs.

**Permutation invariance.** Messages aggregate by a symmetric operation, so relabelling nodes relabels outputs identically and changes nothing else. In the graphical-model case this follows from the factorisation; in the GNN case it is the architectural reason the approach suits sets of interacting entities.

**Approximation on loops.** Both are exact or well-behaved on trees and approximate on graphs with cycles. Loopy belief propagation may not converge; a deep GNN over-smooths, as repeated aggregation collapses node representations toward each other. See [[graph-neural-network]].

> ### `the-same-pattern-computes-inference-and-representation`
> **Passing distributions and passing learned features are the same algorithm over the same substrate, differing in whether the message function is specified or fitted. Treating them as unrelated methods obscures that the tractability constraints — locality, tree-exactness, loop approximation — apply identically to both.**
> ^[generated. rests-on: source:trueskill-message-passing, imported:gnn-formulation]

## Attention Is Message Passing

Self-attention over a set with **no positional encoding** is message passing on a fully-connected graph, with attention-weighted rather than summed aggregation.

That places the [[transformer]] in the same family as GNNs and explains why the set-modelling and graph literatures keep arriving at similar constructions from opposite directions. It also means the three standard responses to unordered input are one family rather than three:

| Approach | Mechanism |
|---|---|
| Graph message passing | Symmetric aggregation — equivariant by construction |
| Attention over a set | Learned, order-free weighting |
| Sorting by a key | Impose a canonical order and accept the loss |

The third is the cheap option, viable where a model cannot express equivariance at all — tree ensembles, for instance. Its cost is that the sort key must be meaningful and near-ties swap positions under noise.

## See Also

- [[graph-neural-network]] · [[transformer]] · [[attention-mechanism]] · [[trueskill]] · [[expectation-propagation]]
- [[bayesian-inference]] · [[uncertainty-quantification]] · [[representation-learning]] · [[multi-agent-reinforcement-learning]]
- [[bayesian-true-skill-rating|TrueSkill Summary]]
