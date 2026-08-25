---
title: "Graph Neural Network"
type: concept
tags: [graph-neural-network, message-passing, deep-learning, architecture, representation-learning, multi-agent, set-modelling]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 22%
  generated: 8%
  imported: 68%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Graph Neural Network

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

A neural architecture operating on graph-structured data by passing learned messages along edges. Node representations are computed from neighbours, iteratively:

$$e_{(k,j)} = f_e([v_k, v_j]), \qquad o_k = f_v\Big(\sum_{j \in N(k)} e_{(k,j)}\Big)$$

Both $f_e$ and $f_v$ are learned. See [[message-passing]] for what this shares with graphical-model inference, which uses the same pattern to pass distributions rather than features.

## Permutation Equivariance Is the Point

The aggregation is a **symmetric function** — a sum, mean or max over neighbours — so relabelling the nodes relabels the outputs identically and changes nothing else.

That is the property that makes GNNs the natural choice for **sets of interacting entities with no canonical order.** Feeding such a set to a sequence model requires imposing an order, and any imposed order is an arbitrary choice that the model then learns from.

Three ways to handle unordered input, all in this vault:

| Approach | Mechanism |
|---|---|
| **Graph message passing** | Symmetric aggregation — equivariant by construction |
| **[[transformer\|Attention over a set]]** | Learned order-free weighting |
| Sorting by a key | Impose a canonical order and accept the loss |

The second is closer to the first than it looks. **Self-attention over a set with no positional encoding *is* message passing on a fully-connected graph**, with attention-weighted rather than summed aggregation. The two literatures arrive at the same construction from opposite directions.

The third is the cheap option, viable where the model cannot express equivariance at all. Its cost is that the sort key must be meaningful and near-ties swap positions under noise.

## Over-Smoothing

^[imported]

Stacking many message-passing layers causes node representations to converge toward each other — each round mixes a node with its neighbours, and after enough rounds every node has mixed with everything.

The consequence is a hard practical limit: **GNNs are typically shallow**, two to four layers, which bounds how far information propagates. A property requiring evidence from six hops away is not reachable without over-smoothing everything else.

This is the graph analogue of the vanishing-gradient problem, and unlike that one it is not solved by gating or residual connections — the mixing is the mechanism, not a side effect.

## Variants

| | Aggregation |
|---|---|
| **GCN** | Degree-normalised mean |
| **GraphSAGE** | Sampled neighbourhood, learned aggregator |
| **GAT** | Attention-weighted — the explicit bridge to [[transformer|Transformers]] |

## Where the Graph Comes From

Often unremarked and frequently the most consequential modelling choice.

Some domains supply a graph — molecules, citation networks, social ties. Others do not, and the edges are **constructed**: fully-connected, $k$-nearest-neighbour, distance-thresholded.

> ### `a-constructed-graph-is-a-modelling-assumption`
> **Where edges are chosen rather than given, the graph encodes a hypothesis about which entities interact. A distance threshold asserts that interaction has a range; a fully-connected graph asserts everything interacts and pushes the whole burden onto the learned weights.**
> ^[generated. rests-on: imported:gnn-graph-construction]

Neither is neutral, and the choice is rarely swept.

## See Also

- [[message-passing]] · [[transformer]] · [[attention-mechanism]] · [[representation-learning]] · [[multi-agent-reinforcement-learning]]
- [[trajectory-prediction]] · [[variational-autoencoder]] · [[convolution]] · [[model-selection]]
