---
title: "Capability Profiling"
type: concept
tags: [evaluation, AI, cognitive-science, reasoning, model-decomposition, knowledge-intensive, single-source]
sources: [raw/papers/agi_definition.md]
confidence: 0.8
provenance:
  extracted: 60%
  inferred: 30%
  generated: 8%
  imported: 2%
  ambiguous: 0%
lifecycle: draft
created: 2026-07-27
updated: 2026-07-27
---

# Capability Profiling

Evaluating a system by **decomposing capability into domains and reporting the vector**, rather than aggregating to a single score.

[[agi-definition|Hendrycks et al. (2025)]] is the vault's instance: they ground a definition of AGI in Cattell–Horn–Carroll theory, split general intelligence into ten domains weighted equally at 10% each, and report a per-domain profile alongside the total.

## The Argument for the Vector

A composite score answers "how good", which is rarely the question that determines what to do next. The profile answers **"good at what, and bad at what"** — and the second is what identifies a bottleneck.

Their result makes the case. GPT-4 and GPT-5 score 27% and 57% overall, which reads as steady progress. The decomposition reads differently:

| Domain | GPT-4 | GPT-5 |
|---|---|---|
| General knowledge | 8% | 9% |
| Mathematical ability | 4% | 10% |
| On-the-spot reasoning | 0% | 7% |
| **Long-term memory storage** | **0%** | **0%** |
| Speed | 3% | 3% |

**Long-term memory storage sits at zero in both.** No aggregate would show that, and it is the finding the paper is built around — a capability that did not improve at all across a generation, hidden inside a number that nearly doubled.

## Jaggedness

The authors describe current systems as having a **"jagged" cognitive profile**: strong in knowledge-intensive domains, weak in foundational machinery.

Jaggedness matters beyond description. A smooth profile means an aggregate is a fair summary; a jagged one means the aggregate is a weighted average over things that are not substitutes for one another. **The more jagged the profile, the more misleading the composite** — and nothing in a single score tells you which case you are in.

## Capability Contortions

The sharpest idea here.^[imported: the term is the source's; the generalisation below is not] The authors argue that workarounds masking a missing capability are not the capability:

- **Massive context windows** substitute for long-term memory *storage*
- **[[retrieval-augmented-generation|RAG]]** substitutes for long-term memory *retrieval*

Both produce competent behaviour on tasks that would otherwise require the missing faculty, so both inflate an aggregate score while leaving the underlying deficit in place.

The general form is worth stating: **a workaround that succeeds on the benchmark and fails on the capability is invisible to any evaluation that only measures the benchmark.** That is a close cousin of the [[rare-event-proxy-targets|proxy-target problem]] — the proxy becomes the definition — arriving from the evaluation side rather than the training side.

## The Same Lesson, Elsewhere in This Vault

Decomposition beating aggregation recurs here, independently derived each time:

| Case | Composite | What the decomposition showed |
|---|---|---|
| [[receiving-efficiency]] | Defenders 1.06, midfielders 1.03 — near-identical | Split by receiving/interception, they do entirely different things |
| [[intent-vs-outcome-valuation]] | VAEP conflates decision and execution | I-VAEP and O-VAEP separate them |
| [[expected-possession-value]] | "EPV" as one term | Four distinct quantities under one name |
| [[class-imbalance-evaluation]] | Brier alone | F1 exposes a classifier that finds nothing |

Four football cases and one AI case reaching the same structural point: **an aggregate is only a fair summary when its components are substitutes.** Where they are not, the composite is the least informative number available and the one most likely to be quoted.

That is an argument for reporting profiles in the vault's own domain too. Almost every football framework here ends by summing to a per-90 rating — see the aggregation-step discussion on [[action-valuation]] — and none reports a per-domain profile of what the player is good and bad at, despite the machinery existing.

## Limitations

- **Equal weighting is a choice, not a finding.** Ten domains at 10% each is defensible as a default and unjustified as a measurement. A different weighting produces a different total, and nothing here shows rankings are stable across weightings — the same objection the vault raises about [[free-parameters-load-bearing|asserted free parameters]].
- **CHC theory is a model of *human* cognition.** Whether its factor structure is the right decomposition for a non-human system is assumed rather than argued.
- **Scores are point estimates** with no reported uncertainty. See [[uncertainty-quantification]].
- **Single source**, and a 30-author position paper rather than an experimental result.

## See Also

- [[agi-definition|Source Summary]] · [[retrieval-augmented-generation]] · [[scaling-laws]] · [[uncertainty-quantification]]
- [[receiving-efficiency]] · [[intent-vs-outcome-valuation]] · [[expected-possession-value]] · [[class-imbalance-evaluation]]
- [[rare-event-proxy-targets]] · [[structured-model-decomposition]] · [[free-parameters-load-bearing]] · [[action-valuation]]
- [[predictive-validity]] · [[split-half-reliability]] · [[model-selection]]
