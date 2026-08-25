---
title: "Split-Half Reliability"
type: concept
tags: [reliability, evaluation, statistics, predictive-validity, machine-learning, cognitive-science]
sources: []
confidence: 0.65
provenance:
  extracted: 0%
  inferred: 20%
  generated: 8%
  imported: 70%
  ambiguous: 2%
lifecycle: draft
created: 2026-08-14
updated: 2026-08-14
---

# Split-Half Reliability

> ⚠️ **No held source.** Background knowledge from psychometrics, marked `imported:`.

Whether a measurement is **consistent with itself**. Split the observations into two halves, compute the metric on each, and correlate the results. A metric that disagrees with itself cannot be measuring a stable property.

Because each half uses only part of the data, the raw correlation understates the full-sample reliability; the Spearman–Brown correction adjusts for this.

## Reliability Is a Ceiling on Validity

The central point, and the reason reliability is checked first.

**A measure cannot correlate with anything else more strongly than it correlates with itself.** Unreliability puts a hard ceiling on any relationship the metric can exhibit — so a weak observed correlation between two constructs may reflect noisy instruments rather than a weak underlying relationship.

That makes reliability a *precondition* for interpreting [[predictive-validity|predictive]] results, not an alternative to them.

## What Reliability Cannot Tell You

| | Reliability | [[predictive-validity\|Predictive validity]] |
|---|---|---|
| Asks | Is it stable? | Does it forecast? |
| Needs | Repeated samples | An external outcome |
| Gamed by | **A degenerate constant** | Genuinely forecasting |

A metric returning the same number regardless of input is perfectly reliable and useless. **Reliability is necessary and nowhere near sufficient**, and optimising a metric *for* reliability risks selecting exactly this failure — favouring a measure that is stable because it captures little.

The two should be reported together for that reason. Neither alone distinguishes a good metric from a degenerate one.

## Reliability and Real Change

The awkward case: a measurement can be unstable because the instrument is noisy, or because **the thing being measured genuinely varies.**

Split-half correlation cannot separate these. A low value means "the two halves disagree" and says nothing about which explanation applies.

> ### `instability-and-variation-are-the-same-number`
> **Split-half reliability and genuine within-subject variation are the same variance component read with opposite signs. One framing treats it as measurement error to be minimised; the other treats it as signal about the subject. No statistic distinguishes them without an external criterion.**
> ^[generated. rests-on: imported:classical-test-theory]

Resolving it requires something outside the measurement — a known-stable subgroup, a manipulation with an expected effect, or an external criterion the variation should predict if it is real.

## Related Forms

^[imported]

| Form | Splits by |
|---|---|
| **Split-half** | Arbitrary partition of observations |
| **Test–retest** | Time — the same subject measured twice |
| **Cronbach's $\alpha$** | Effectively the mean of all possible split-halves |
| **Inter-rater** | Rater, rather than observation |

Test–retest is the one most confounded by real change, since anything genuinely varying over the interval appears as unreliability.

## Limitations

- **The split must be arbitrary.** Splitting on anything correlated with the measured property inflates the estimate.
- **Sample size dependence** — reliability estimated on few observations is itself unreliable.
- **No absolute threshold.** Conventional cut-offs (0.7, 0.8) are conventions, not findings.

## See Also

- [[predictive-validity]] · [[uncertainty-quantification]] · [[capability-profiling]] · [[model-selection]] · [[probability-calibration]]
- [[selection-bias]] · [[class-imbalance-evaluation]] · [[interpretability]]
