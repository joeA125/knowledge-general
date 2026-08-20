---
title: "Agent Memory"
type: concept
tags: [AI, ai-agent, architecture, external-memory, representation-learning]
sources: [raw/articles/ai-agent-architecture-breakdown.md]
confidence: 0.85
provenance:
  extracted: 75%
  inferred: 20%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-08
updated: 2026-07-08
---

# Agent Memory

Agent memory refers to the state management systems that enable [[ai-agent]]s to maintain context across reasoning steps and interactions. Unlike stateless chatbots, agents require persistent, queryable memory to track goals, accumulate observations, and learn from past actions.

## Two-Layer Architecture

### Short-Term (Working) Memory
Current task context, recent actions taken, and active tool results. Stored in-memory (e.g., Redis, application state) for fast access. Analogous to the registers in a CPU — small, fast, and directly accessible by the reasoning engine.

- Scope: Single task execution
- Lifetime: Discarded on task completion
- Access pattern: Sequential (action history) + random (current state)

### Long-Term (Persistent) Memory
Conversation history, learned patterns, user preferences, and successful action templates. Stored across two systems:

- **Vector database** (Pinecone, Qdrant, Weaviate): Enables semantic similarity search over past experiences. Given a query, retrieves the $k$ most relevant historical interactions.
- **Relational database** (PostgreSQL): Structured storage for user profiles, task metadata, and explicit knowledge.

Retrieval combines both: recent context from short-term memory + semantically relevant past experiences from the vector store.

## Relation to Neural Memory Architectures

The agent memory architecture mirrors — at the systems level — the same controller + external memory pattern found in [[neural-turing-machine]]s (Graves et al., 2014):

| Dimension | Neural Turing Machine | AI Agent |
|---|---|---|
| Controller | LSTM/feedforward network | LLM brain |
| Memory | $N \times M$ differentiable matrix | Vector DB + relational DB |
| Read | Content-based + location-based attention | Semantic search + SQL queries |
| Write | Erase-then-add via attention weights | Insert/update via API calls |
| Training | End-to-end gradient descent | Not trained; engineered |
| Addressing | Differentiable soft attention | Embedding similarity + exact match |

The NTM learns its memory access patterns via gradient descent; the agent's memory patterns are engineered explicitly. The [[read-process-write]] architecture (Vinyals et al., 2016) represents an intermediate point — attention-based memory with a fixed read-process-write structure.

## Memory and RAG

[[retrieval-augmented-generation]] can be viewed as a specific memory pattern within the agent architecture: the vector database serves as the non-parametric memory, and retrieval is the read operation. The key insight from the agent memory framing is that retrieval is just one of several memory operations — agents also write (store new experiences), update (refine existing knowledge), and forget (expire stale context).

## See Also

- [[ai-agent]]
- [[neural-turing-machine]]
- [[read-process-write]]
- [[retrieval-augmented-generation]]
- [[ai-agent-architecture-breakdown|Source Summary]]
