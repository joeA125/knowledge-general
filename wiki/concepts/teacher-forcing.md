---
title: "Teacher Forcing"
type: concept
tags: [teacher-forcing, training-technique, autoregressive-model, sequence-modelling, deep-learning, generative-model, alignment]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 26%
  generated: 8%
  imported: 64%
  ambiguous: 2%
lifecycle: draft
created: 2026-07-24
updated: 2026-08-14
---

# Teacher Forcing

> ⚠️ **No held source.** Background knowledge, marked `imported:`.

Training an [[autoregressive-model|autoregressive model]] by conditioning each prediction on the **ground-truth** history rather than on the model's own previous outputs. At step $t$ the model sees $y_{1:t-1}$ from the data and predicts $y_t$.

The default regime for [[transformer]] language models and effectively every generative sequence model in use, [[gpt]] included.

## Why It Is Used

Two reasons, both practical:

**Parallelism.** Because the whole ground-truth sequence is known in advance, all positions train simultaneously under causal masking. Feeding the model its own outputs would force sequential generation *during training*, forfeiting the Transformer's main computational advantage.

**Stability.** Early in training the model's own predictions are near-random. Conditioning on them would compound errors and supply almost no learning signal.

## Exposure Bias

The cost is a **train/inference mismatch**.

During training the model only ever sees *correct* history. At generation time it conditions on its own outputs, which contain errors — so it operates on a distribution of prefixes it never encountered. Small errors compound: an unusual token makes the next prediction slightly worse, which makes the next worse still.

**The effect scales with generation length**, which makes it a property of how a model is *used* rather than of the model. The same weights are near-unaffected when predicting one step ahead and badly affected over a long rollout.

> ### `next-token-accuracy-does-not-predict-rollout-quality`
> **A model reported on single-step accuracy has been evaluated in the regime teacher forcing optimises for. Nothing in that number indicates how a long generation degrades, and the two can diverge arbitrarily — which is why generative systems evaluated by rollout should report rollout metrics.**
> ^[generated. rests-on: imported:exposure-bias]

## Mitigations

^[imported]

- **Scheduled sampling** gradually replaces ground-truth tokens with model samples during training, annealing the mismatch away. It complicates parallelism and adds a schedule to tune.
- **Constrained decoding** does not fix exposure bias but bounds its damage. If invalid tokens carry zero probability, compounding errors cannot produce *impossible* outputs — only implausible ones. This matters far more for long rollouts than for single-step prediction.
- **Sampling and averaging.** Generating many rollouts and averaging reduces variance from any single degenerate trajectory, though it does not correct systematic drift.
- **Sequence-level objectives.** Training against a whole-sequence reward — as in [[rlhf]] — optimises what is actually evaluated. That is one framing of why RLHF helps beyond supervised fine-tuning: **it trains on the model's own generations.**

## A Family of Train/Inference Mismatches

Teacher forcing is one instance of a general pattern:

| Mismatch | Direction |
|---|---|
| **Teacher forcing** | Trained on true history, run on generated history |
| [[masked-language-model\|Masked LM]] | `[MASK]` appears in pre-training, never in fine-tuning |
| [[dropout]] | Active in training, disabled at inference |

The BERT case is instructive because the correction is explicit: 10% of masked positions are left unchanged and 10% randomised, precisely so the model does not learn to depend on seeing the mask token. **The mismatch was anticipated and priced in**, which is rarer than it should be.

The general lesson: **train on the distribution you will be evaluated on**, and where that is impractical, know which direction the mismatch runs and what it costs.

## See Also

- [[autoregressive-model]] · [[transformer]] · [[gpt]] · [[generative-model]] · [[imitation-learning]]
- [[rlhf]] · [[proximal-policy-optimization]] · [[masked-language-model]] · [[dropout]] · [[bert]]
- [[counterfactual-simulation]] · [[trajectory-prediction]] · [[event-prediction]] · [[model-selection]]