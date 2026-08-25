---
title: "Constrained Decoding"
type: concept
tags: [constrained-decoding, language-modelling, autoregressive-model, generative-model, sequence-modelling, tool-use, evaluation]
sources: []
confidence: 0.6
provenance:
  extracted: 0%
  inferred: 26%
  generated: 10%
  imported: 62%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Constrained Decoding

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Restricting an [[autoregressive-model|autoregressive]] model's output to sequences satisfying validity rules, by **masking invalid tokens** at each generation step rather than filtering afterwards.

## Why Mask Rather Than Filter

Rejection sampling — generate, check, retry — is the obvious alternative and is far worse:

- **Cost scales with invalidity.** If most samples fail, most compute is discarded.
- **It gives no partial credit.** A sequence invalid at the last token is thrown away entirely, along with everything correct before it.
- **It cannot guarantee termination.** Nothing bounds how many attempts are needed.

Masking makes invalid output **unreachable**, so the first sample is valid by construction. The probability mass is redistributed over legal continuations rather than wasted on illegal ones.

## What Kind of Rules

| Constraint | Enforced by |
|---|---|
| **Grammar / schema** | A parser tracking which tokens can legally follow |
| **Type** | The expected type at the current position |
| **Vocabulary** | A restricted token set for the current field |
| **Structural** | Bracket matching, closing tags, field ordering |

The general requirement is that validity be **checkable incrementally** — decidable from the prefix, without seeing the rest. Constraints that can only be evaluated on a complete sequence cannot be masked and must be handled another way.

## The Interaction With Exposure Bias

The connection worth drawing, and it is not the obvious one.

Constrained decoding does **not** fix [[teacher-forcing|exposure bias]]. The model still conditions on its own outputs at inference and still drifts from the training distribution.

What it does is **bound the damage**. If invalid tokens carry zero probability, compounding errors cannot produce *impossible* output — only implausible output.

> ### `constraints-bound-the-damage-from-compounding-error-without-fixing-it`
> **Masking makes the failure mode of a drifting autoregressive model degrade from invalid to merely wrong. That distinction matters most for long rollouts, where unconstrained generation reliably leaves the space of well-formed outputs and constrained generation cannot.**
> ^[generated. rests-on: imported:constrained-decoding-practice]

The practical corollary: **the longer the intended generation, the more constraints are worth.** For single-token classification they buy almost nothing.

## What It Cannot Do

- **Validity is not correctness.** A well-formed answer can be entirely wrong, and masking says nothing about which.
- **It can distort the distribution.** Renormalising over a legal subset changes relative probabilities in ways that are not always benign — a model steered away from its preferred continuation may produce something it assigns low probability to overall.
- **It hides model failure.** A model that would have produced garbage now produces well-formed garbage, which is harder to notice in evaluation. **That is a real cost, and it argues for measuring quality separately from validity.**

## Where It Is Used

Structured output — JSON, XML, SQL — where downstream parsing requires well-formedness; function calling, where tool names and argument types must be legal; code generation, where syntax is checkable incrementally; and any domain whose tokens encode a grammar rather than free text.

## See Also

- [[autoregressive-model]] · [[teacher-forcing]] · [[generative-model]] · [[gpt]] · [[transformer]] · [[tokenization]]
- [[probability-calibration]] · [[model-selection]] · [[predictive-validity]] · [[interpretability]]
