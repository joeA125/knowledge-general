---
title: "Teacher Forcing"
type: concept
tags: [teacher-forcing, training-technique, autoregressive-model, sequence-modelling, deep-learning, generative-model]
sources: [raw/papers/eventgpt-player-impact-from-team-action-sequences.md]
confidence: 0.85
provenance:
  extracted: 40%
  inferred: 50%
  ambiguous: 10%
lifecycle: reviewed
created: 2026-07-24
updated: 2026-07-24
---

# Teacher Forcing

Teacher forcing trains an [[autoregressive-model|autoregressive model]] by conditioning each prediction on the **ground-truth** history rather than on the model's own previous outputs. At step $t$ the model sees $y_{1:t-1}$ from the data and predicts $y_t$.

It is the default training regime for [[transformer]] language models and every generative sequence model in this vault, including [[gpt]], [[eventgpt]] and [[scoutgpt]].

## Why It Is Used

Two reasons, both practical:

- **Parallelism.** Because the entire ground-truth sequence is known in advance, all positions can be trained simultaneously with causal masking. Feeding the model its own outputs would force sequential generation during training, forfeiting the transformer's main computational advantage.
- **Stability.** Early in training the model's own predictions are near-random. Conditioning on them would compound errors and provide almost no learning signal.

## Exposure Bias

The cost is a **train/inference mismatch**, usually called exposure bias.

During training the model only ever sees *correct* history. At generation time it conditions on its own outputs, which contain errors — so it operates on a distribution of prefixes it never encountered during training. Small errors compound: an unusual token makes the next prediction slightly worse, which makes the next worse still.

The effect scales with generation length, which makes it directly relevant to this vault's sports models, all of which are trained on next-token prediction but evaluated by **rolling out whole episodes**:

| Model | Rollout length |
|---|---|
| [[nmstpp]] | Next event only |
| [[eventgpt]] | Fixed sequence re-scored (no rollout) |
| [[scoutgpt]] | Full episode generated |

The ordering is informative. [[eventgpt]] sidesteps exposure bias entirely for its counterfactual — it holds the observed event sequence fixed and only re-evaluates value under a substituted player, so nothing is generated. [[scoutgpt]] regenerates the whole sequence, which is a stronger counterfactual but exposes it to compounding error over ~28 events.

Neither paper measures exposure bias directly, which is a gap: reported next-event accuracy does not tell you how a 30-event rollout degrades.

## Mitigations

- **Scheduled sampling** (Bengio et al., 2015) gradually replaces ground-truth tokens with model samples during training, annealing the mismatch away. It complicates parallelism and introduces a schedule to tune.
- **[[constrained-decoding]]** does not fix exposure bias but bounds its damage. If invalid tokens carry zero probability, compounding errors cannot produce *impossible* sequences — only implausible ones. This is part of why ScoutGPT's validity masking matters more for long rollouts than for single-step prediction.
- **Sampling and averaging.** Generating many rollouts and averaging, as both models do via Monte Carlo, reduces the variance contributed by any single degenerate trajectory — though it does not correct systematic drift.
- **Sequence-level objectives** — RL-style training against a sequence reward, as in [[rlhf]] — optimise what is actually evaluated. This is one framing of why RLHF helps beyond supervised fine-tuning: it trains on the model's own generations.

## Relation to Other Training/Inference Mismatches

Teacher forcing belongs to a family of gaps between how a model is trained and how it is used:

- [[masked-language-model|BERT]]'s `[MASK]` token appears in pre-training but never in fine-tuning, which is why 10% of masked positions are left unchanged and 10% randomised.
- [[dropout]] is active during training and disabled at inference, requiring rescaling.
- [[scoutgpt]] excludes player-ID from its generative loss precisely because player identity is *injected* at inference rather than generated — an explicit correction of exactly this kind of mismatch.

The general lesson: **train on the distribution you will be evaluated on**, and where that is impractical, know which direction the mismatch runs.

## See Also

- [[autoregressive-model]]
- [[constrained-decoding]]
- [[scoutgpt]]
- [[eventgpt]]
- [[gpt]]
- [[rlhf]]
