---
title: "Message Passing"
type: concept
tags: [message-passing, inference, probabilistic-graphical-model, factor-graph, graph-neural-network, approximation, deep-learning]
sources: [raw/papers/bayesian-true-skill-rating.md, raw/papers/evaluation_creating_scoring_opportunities_trajectory_prediction.md]
confidence: 0.8
provenance:
  extracted: 40%
  inferred: 55%
  ambiguous: 5%
lifecycle: draft
created: 2026-07-27
updated: 2026-07-27
---

# Message Passing

Computing something about a graph by having nodes and edges exchange local information iteratively, until the local computations collectively answer a global question.

The vault holds two traditions that share this name and are rarely discussed together. They turn out to be the same computational pattern applied to different objects, which is worth making explicit.

## Tradition 1: Inference on Graphical Models

Given a distribution factorised over a [[factor-graph]], compute marginals by passing messages between variable and factor nodes. Each message summarises everything one part of the graph knows about a variable; on a tree the algorithm is exact.

Used in [[trueskill]], where the factor graph encodes players' skills, per-game performances, team sums and outcome comparisons. Marginal skill beliefs fall out of the message schedule. Because the comparison factors are non-Gaussian, exact messages are intractable, so [[expectation-propagation]] approximates them by moment matching — see [[approximate-message-passing]] and [[gaussian-density-filtering]].

**What is passed:** probability distributions (or their sufficient statistics).
**What is computed:** marginals of a known distribution.
**Learned:** nothing — the graph structure and factors are specified.

## Tradition 2: Representation Learning on Graphs

Given a graph of entities, compute node representations by passing learned messages along edges. In a [[graph-neural-network|GNN]]:

$$e_{(k,j)} = f_e([v_k, v_j]), \qquad o_k = f_v\Big(\sum_{j \in N(k)} e_{(k,j)}\Big)$$

where $f_e$ and $f_v$ are neural networks. Used in the GVRNN of [[trajectory-prediction]], where nodes are players and each agent's latent distribution is conditioned on all others.

**What is passed:** learned feature vectors.
**What is computed:** representations for a downstream task.
**Learned:** the message functions themselves.

## What They Share

Three properties, and each has real consequences.

**Locality.** Only neighbours communicate directly, so global structure emerges from repeated local operations. This is what makes both tractable on large graphs.

**Permutation invariance.** Messages are aggregated by a symmetric operation — sum or product — so relabelling nodes relabels outputs identically and changes nothing else. In the graphical-model case this is a property of the factorisation; in the GNN case it is the architectural reason the approach suits multi-agent problems, where twenty-two players have no canonical ordering. See [[graph-neural-network]].

**Approximation on loops.** Both are exact or well-behaved on trees and approximate on graphs with cycles — loopy belief propagation may not converge; a deep GNN over-smooths as repeated aggregation makes node representations collapse toward each other.

## The Connection Worth Noting

Self-attention over a set with no positional encoding **is** message passing on a fully-connected graph, with attention-weighted rather than summed aggregation. That places the [[transformer]] in the same family as GNNs, and explains why [[read-process-write|Order Matters]] and the sports GNN literature keep arriving at similar constructions from opposite directions.

The vault's three answers to the same ordering problem line up accordingly:

| Approach | Mechanism | Example |
|---|---|---|
| Graph message passing | Symmetric aggregation | GVRNN in [[trajectory-prediction]] |
| Attention over a set | Learned order-free weighting | [[read-process-write]], [[transformer]] |
| Sorting by a meaningful key | Impose canonical order | [[vdep]] — players sorted by distance to ball |

The third is the cheap version, viable for models like tree ensembles that cannot express equivariance at all — and lossier, since the sort key must be meaningful and near-ties swap slots under noise.

## See Also

- [[factor-graph]] · [[approximate-message-passing]] · [[expectation-propagation]] · [[gaussian-density-filtering]]
- [[graph-neural-network]] · [[trajectory-prediction]] · [[transformer]] · [[read-process-write]]
- [[trueskill]] · [[inference]] · [[approximation]] · [[vdep]]
- [[bayesian-true-skill-rating|TrueSkill Summary]]
