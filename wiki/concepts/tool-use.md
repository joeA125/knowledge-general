---
title: "Tool Use (LLM)"
type: concept
tags: [AI, ai-agent, tool-use, language-modelling, architecture]
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

# Tool Use (LLM)

Tool use (also called function calling) is the capability of large language models to interact with external systems by selecting, parameterising, and invoking defined tools — transforming LLMs from text generators into [[ai-agent|agents]] that can act in the world.

## How It Works

Tools are defined as structured schemas (name, description, input parameters with types). The LLM receives these definitions alongside the user's request and decides whether and which tool to call, generating structured parameters. The execution flow:

$$\text{LLM decision} \rightarrow \text{Tool selection} \rightarrow \text{Parameter validation} \rightarrow \text{Security check} \rightarrow \text{Execution} \rightarrow \text{Result validation} \rightarrow \text{Return to LLM}$$

## Tool Categories

- **Information retrieval:** Web search, database queries, API calls, file system access. Overlaps with [[retrieval-augmented-generation]] when the tool retrieves knowledge for the LLM.
- **Action execution:** Send emails, create calendar events, execute code, modify databases. These tools have side effects and require permission management.

## Security Architecture

The [[ai-agent-architecture-breakdown|architecture breakdown]] details critical security layers:
- **Least privilege:** Each tool requires explicit permission grants with scope limitations.
- **Rate limiting:** Prevents tool abuse and controls API costs.
- **Parameter validation:** Sanitises inputs, prevents injection attacks, validates types.
- **Approval workflows:** Sensitive actions require human confirmation before execution.
- **Cost controls:** Per-task budget limits with alert thresholds.

## Relation to ReAct

In [[react|ReAct (Yao et al., 2023)]], tools are the "actions" in the Thought–Action–Observation loop. The action space is augmented with language: $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$, where $\mathcal{A}$ contains tool calls (with real-world effects) and $\mathcal{L}$ contains thoughts (no effects). This framing makes tool use a first-class component of the reasoning process, not just an API call.

## Practical Considerations

- **Parallel execution:** Independent tool calls can run concurrently, improving latency.
- **Idempotent caching:** Results of read-only tools can be cached to avoid redundant calls.
- **Fallback chains:** If a primary tool fails, the system retries with exponential backoff, then falls back to a degraded response.
- **Connection pooling:** Persistent connections to frequently-used APIs reduce latency.

## See Also

- [[ai-agent]]
- [[react]]
- [[retrieval-augmented-generation]]
- [[ai-agent-architecture-breakdown|Source Summary]]
