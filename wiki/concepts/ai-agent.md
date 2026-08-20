---
title: "AI Agent"
type: concept
tags: [AI, ai-agent, architecture, language-modelling, tool-use, reasoning]
sources: [raw/articles/ai-agent-architecture-breakdown.md, raw/papers/synergising-reasoning-acting-llms.md]
confidence: 0.9
provenance:
  extracted: 75%
  inferred: 20%
  ambiguous: 5%
lifecycle: reviewed
created: 2026-07-08
updated: 2026-07-08
---

# AI Agent

An AI agent is an autonomous system built around a large language model that can plan, reason, use tools, and take actions iteratively toward a goal — in contrast to a chatbot, which is a stateless request-response system.

## Core Architecture

An agent is fundamentally a **stateful autonomous loop**:

$$\text{Goal} \rightarrow \text{Plan} \rightarrow \text{Act} \rightarrow \text{Observe} \rightarrow \text{Re-plan} \rightarrow \ldots \rightarrow \text{Goal Achieved}$$

The [[ai-agent-architecture-breakdown|architecture breakdown]] decomposes this into seven components: LLM brain (reasoning), [[agent-memory|memory system]] (state), [[tool-use|tool interface]] (actions), planning engine, execution loop, monitoring, and security.

## Planning Paradigms

### ReAct
[[react|ReAct (Yao et al., 2023)]] interleaves reasoning traces ("thoughts") with tool actions in a Thought–Action–Observation loop. Thoughts are actions in language space that don't affect the environment but update the reasoning context. ReAct grounds reasoning in external observations, eliminating hallucination (0% vs 56% for [[chain-of-thought|CoT]] on HotpotQA failures).

### Chain-of-Thought Planning
[[chain-of-thought]] decomposition breaks a goal into numbered steps, each specifying a tool and rationale. Effective for structured, multi-step tasks with predictable workflows.

## What Makes Agents Different from Chatbots

| Dimension | Chatbot | Agent |
|---|---|---|
| State | Stateless | Stateful (working + persistent memory) |
| Autonomy | Waits for input | Operates autonomously until goal completion |
| Actions | Generates text only | Uses tools, executes code, modifies databases |
| Planning | None | Decomposes goals into executable steps |
| Loop | Single turn | Multi-turn with observation and re-planning |
| Iteration limit | N/A | Bounded (typically 25 max iterations) |

## Relation to Other Vault Concepts

- **[[neural-turing-machine]]:** The agent architecture mirrors NTM's controller + external memory design — a central reasoning module (LLM brain) accessing addressable storage (memory system) via attention-like retrieval. The key difference is that agents operate at the systems level (Redis, vector DBs) rather than as differentiable end-to-end architectures.
- **[[retrieval-augmented-generation]]:** RAG is a specific tool/memory pattern within the agent architecture — the agent retrieves relevant context from a knowledge base before generating.
- **[[rlhf]]:** Alignment via RLHF constrains the LLM brain to follow human intent, which is critical for agents that take real-world actions.

## See Also

- [[ai-agent-architecture-breakdown|Source Summary]]
- [[tool-use]]
- [[agent-memory]]
- [[react]]
- [[chain-of-thought]]
- [[retrieval-augmented-generation]]
