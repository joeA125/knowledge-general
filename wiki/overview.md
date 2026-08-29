---
title: "Knowledge Base Overview"
type: synthesis
tags: [machine-learning, deep-learning, evaluation, reinforcement-learning, representation-learning, language-modelling]
sources: []
confidence: 0.75
provenance:
  extracted: 5%
  inferred: 50%
  generated: 35%
  imported: 8%
  ambiguous: 2%
lifecycle: reviewed
created: 2026-08-12
updated: 2026-08-23
---

# Knowledge Base Overview

A narrative map of what this vault holds, how it hangs together, and — unusually prominently — **where it is thin**. The index is the catalog; this page is the argument for the catalog's shape.

**Scope: machine learning, LLMs, and mathematical foundations.** Football and sports analytics belong to the sibling vault.

## Composition

| | Count |
|---|---|
| Total pages | **179** |
| Concepts | 108 |
| Entities | 42 |
| Source summaries | 25 |
| Dashboards | 3 |

**All 25 raw sources are ingested.** Mean page confidence 0.818; 60 pages `reviewed`, 115 `draft`, 1 `archived`.

## The Central Fact About This Vault

> ⚠️ **Roughly half the concept pages have no held source.**

This vault was created in August 2026 by splitting a sports-analytics knowledge base. The **sources** that went to football were the ones grounding much of the general ML material, because that material had been written to serve football papers. What arrived here was a set of concepts whose evidence stayed behind.

Rather than delete those concepts or pretend to sourcing, they were **written from background knowledge and marked `imported:`** — visibly, per page, with a warning box on anything above roughly 60%.

| Cluster | Held sources | Typical `imported:` |
|---|---|---|
| Transformer and attention | **Strong** — Vaswani, Bahdanau, Vinyals | 10–25% |
| Language models | **Strong** — GPT, BERT, InstructGPT, scaling laws | 10–30% |
| Generative models | **Partial** — VLAE only | 20–50% |
| Rating systems | **Partial** — TrueSkill only | 40–47% |
| Evaluation and validity | **None** | 62–70% |
| Reinforcement learning | **One** — InstructGPT, an application paper | **60–78%** |
| Vision geometry | **None** | 64–66% |
| Statistics and inference | **None** | 62–68% |

**The vault defines PPO and RLHF but not, from any source, reinforcement learning itself.** That is the sharpest instance of the pattern.

## Source Acquisition Priorities

The most useful thing that can be done with this vault is to reduce the table above. Five areas, highest need first.

### 1. Reinforcement learning foundations
**The vault's largest gap by a wide margin.** Nine pages at 60–78% imported, grounded only by an application paper.

- **Sutton & Barto, *Reinforcement Learning: An Introduction*** — would ground `reinforcement-learning`, `markov-game`, `value-iteration`, `temporal-difference-learning`, `policy-modelling` in one acquisition
- **Mnih et al. (2015), DQN** — `deep-q-network` and the stabiliser material four other pages reference
- **Schulman et al. (2017), PPO and TRPO** — `proximal-policy-optimization` is currently sourced only through InstructGPT

### 2. Measurement and validity
Seven pages with no source, and they underpin how every other page reasons about evidence.

- **A psychometrics or classical-test-theory text** — `predictive-validity`, `split-half-reliability`, and the reliability/real-change distinction
- **Guo et al. (2017), *On Calibration of Modern Neural Networks*** — `probability-calibration`, `probabilistic-classification`, `uncertainty-quantification`
- **A model-selection reference** — `model-selection` is the vault's second-largest hub at 45 inbound and rests on nothing

### 3. Statistics and inference
- **Rasmussen & Williams, *Gaussian Processes for Machine Learning*** — `gaussian-process`
- **Daley & Vere-Jones on point processes** — `point-process`, `neural-temporal-point-process`
- **A reference on identifiability** — currently the only page arguing that more data does not always help

### 4. Geometric vision
- **Hartley & Zisserman, *Multiple View Geometry*** — grounds `homography` and `camera-calibration` together; the whole cluster rests on background knowledge

### 5. Generative and ensemble methods
- **Goodfellow et al. (2014) and Isola et al. (2017)** — `conditional-gan`, and the adversarial half of `generative-model`
- **Friedman (2001); Lee & Seung (1999)** — `gradient-boosting`, `non-negative-matrix-factorization`

---

**Priorities 1 and 2 together would move fifteen pages** from the vault's weakest tier to its strongest, and are the clear first acquisition.

Two gaps are structural rather than evidential: **`random-forest` and `feature-attribution` are referenced but have no page**, left uncreated rather than expanding the `imported:` tier further.

## The Hubs

| Page | Inbound |
|---|---|
| [[transformer]] | 57 |
| [[model-selection]] | 45 |
| [[uncertainty-quantification]] | 28 |
| [[attention-mechanism]] | 23 |
| [[bayesian-inference]] | 23 |
| [[predictive-validity]] | 23 |
| [[representation-learning]] | 23 |
| [[scaling-laws]] | 22 |
| [[lstm]] | 21 |

**[[model-selection]] at 45 is the surprise.** Not a headline topic, but the asserted-parameter problem it describes is relevant on nearly every page discussing a method with knobs. Its four-kind taxonomy — horizon, shape, gate, prior strength — is the vault's most reused idea.

**[[transformer]] at 57** reflects that the substructure is meaningless without it. It was the last major page created and unblocked 60 others.

## Recurring Themes

Claims that recur across independently written clusters, which usually signals something real:

- **A metric that cannot fail is not evidence.** `self-prediction-is-not-validation`, `a-proxy-must-be-validated-against-the-thing-it-replaces`, `zero-shot-claims-depend-on-what-the-pretraining-corpus-contained`.
- **Free parameters hide in plain sight.** `asserted-parameters-are-untested-model-selection`, `factorisation-order-is-an-unswept-parameter`, `imitation-weight-tunes-the-conclusion`.
- **Measurement noise and real variation are conflated.** `instability-and-variation-are-the-same-number` and `measurement-noise-and-real-variation-need-separate-parameters`, reached from measurement theory and rating systems independently.
- **Assumptions move rather than disappear.** `transfer-imports-the-pretraining-corpus-assumptions`, `a-constructed-graph-is-a-modelling-assumption`, `tokenization-decisions-are-invisible-downstream`.

## Migration Record

Created 2026-08 from `knowledge-sports`. 96 pages and 16 raw sources moved; 9 sources are held in both vaults because a page in each depends on them.

**What the migration taught:**

- **Copied pages carry their frontmatter.** 19 phantom `sources:` entries — paths to files that never arrived — across 15 pages. Every page that arrived by file copy had at least one; **no page written here did.** Neither `find_mentioned_but_missing` nor `list_unprocessed_sources` detects this.
- **The fork list was built from one side.** It asked what football needed and never what this vault needed, so ~35 general concepts stayed behind because their *sources* were football papers, though the *concepts* were not.
- **A page is not portable; an argument is.** Every fork needed its sources pruned, links redirected and examples replaced. The reasoning survived; everything vault-specific did not.

## Dashboards

- [[health|Wiki Health]] — stale, low-confidence, draft and orphan-risk pages
- [[reinforcement|Reinforcement]] — ageing, single-source and confidence-decay watch
- [[sources|Source Tracking]] — raw sources and reference counts

## Maintenance Notes

- 155 tags declared, 134 in use. The 21 unused are mostly meta and entity vocabulary held ready
- **115 of 179 pages are `draft`**, appropriate given how many were written in a single migration
- The full page catalog lives in `index.md` at the vault root, outside the wiki folder
- Rewrite this page against `vault_stats` rather than from memory, and update the acquisition table whenever a source lands
