## [2026-08-25 13:36] writeback | Vault established — 49 to 179 pages; 34 concepts created, 19 phantom sources cleared, half the vault knowingly unsourced
Inbound half of the vault split. This vault received the general ML/LLM material from knowledge-sports and was built out from 49 pages to 179.

=== WHAT ARRIVED, AND WHAT WAS MISSING ===
Received: 96 pages (34 concepts, 37 entities, 25 summaries) and 25 raw sources (16 general-only, 9 held in both vaults).

The fork list had been built from ONE SIDE. It asked "what does football need?" and never "what does general need?" — so ~35 general-purpose concepts stayed in the football vault because their SOURCES were football papers, though the CONCEPTS were not. reinforcement-learning, temporal-difference-learning, imitation-learning, interpretability, model-selection, predictive-validity were all in this position.

Largest single gap: TRANSFORMER. 90+ references from 60 pages, no page. The vault held the entire substructure (multi-head-attention, positional-encoding, scaled-dot-product-attention, layer-normalization, feed-forward-network, encoder-decoder) with nothing tying it together.

=== WORK DONE ===
CREATED 34 pages, in this order:
- transformer (first, unblocked 60 pages)
- RL cluster (9): reinforcement-learning, markov-game, value-iteration, temporal-difference-learning, deep-q-network, policy-modelling, imitation-learning, multi-agent-reinforcement-learning, agent-based-simulation
- Evaluation cluster (7): predictive-validity, split-half-reliability, probabilistic-classification, probability-calibration, class-imbalance-evaluation, model-selection, interpretability
- Representation (5): feature-engineering, theory-based-modelling, tokenization, graph-neural-network, structured-model-decomposition
- Sequence (3): event-prediction, point-process, neural-temporal-point-process
- Rating (3): elo-rating-system, bradley-terry-model, glicko-rating-system
- Vision (4): semantic-segmentation, feature-pyramid-network, siamese-network, camera-calibration, homography
- Other (7): transfer-learning, domain-adaptation, selection-bias, kl-divergence, counterfactual-simulation, counterfactual-baseline, zero-shot-learning, constrained-decoding, gradient-boosting, game-theory, identifiability, gaussian-process

REORIENTED 7 heavily football-framed forks: proximal-policy-optimization, trueskill, gated-recurrent-unit, uncertainty-quantification, generative-model, representation-learning, capability-profiling.

FIXED football references across ~20 further pages: teacher-forcing, fully-convolutional-network, non-negative-matrix-factorization, eigenvector, conditional-gan, attention-mechanism, lstm, autoregressive-model, encoder-decoder-bottleneck, gpt, message-passing, variational-autoencoder, trajectory-prediction, rare-event-proxy-targets, and the bert/gpt summaries.

Built index (108 concepts, 42 entities, 25 summaries) and overview from scratch — neither existed.

=== THE PHANTOM SOURCE PROBLEM ===
19 sources: entries pointing at raw files that never arrived, across 15 pages.

EVERY page that arrived by file copy had at least one. NO page written here did. The frontmatter came across unchanged and nothing validates that a cited path exists in the vault citing it.

Neither existing tool detects this: list_unprocessed_sources finds raw files with no summary (opposite direction); find_mentioned_but_missing only sees wikilinks, not frontmatter paths. Found by targeted search_notes on football source filenames, and I declared completion prematurely after four such searches — a fifth (transformer-point-process-football-event-modelling) found four more.

Worst affected were the two FORK-LATE pages copied wholesale rather than written — trajectory-prediction and rare-event-proxy-targets — which arrived with ZERO valid sources and football-only tags. Both rewritten from general knowledge.

=== PROVENANCE STATE — THE DEFINING FACT ===
Roughly half the concept pages have NO held source. The sources grounding the general ML material went to football, because that material had been written to serve football papers.

Rather than delete or pretend, those pages were written from background knowledge and marked imported: visibly, with a warning box above ~60%. Cluster breakdown recorded in the overview:
- Transformer/attention, language models: 10-30% imported, strong sourcing
- Generative, rating: 20-50%, partial
- Evaluation, RL, vision geometry, statistics: 62-78%, none or one source

The vault defines PPO and RLHF but not, from any source, reinforcement learning itself.

SOURCE ACQUISITION PRIORITIES added to the overview as a ranked table. Top two — a foundational RL text, and Mnih 2015 + Hester 2018 — would move nine pages from the weakest tier to the strongest.

=== CLAIMS DECLARED (~30) ===
Notable ones: policy-gradient-forecloses-action-valuation, on-policy-describes-off-policy-prescribes, stabilisers-track-the-feedback-loop, agent-count-is-not-interaction, self-prediction-is-not-validation, instability-and-variation-are-the-same-number, calibration-is-invisible-to-the-metric-that-selects-the-model, f1-zero-can-mean-unthresholded-not-uninformative, asserted-parameters-are-untested-model-selection, a-model-trained-on-selected-outcomes-learns-the-selector, regeneration-fidelity-scales-with-representation-coarseness, transfer-imports-the-pretraining-corpus-assumptions, reprojection-error-understates-error-where-it-matters-most.

All rest on imported: premises where no source is held — weaker footing than the football vault's claims, recorded as such.

=== PROCESS FINDINGS ===
1. A PAGE IS NOT PORTABLE; AN ARGUMENT IS. Every fork needed sources pruned, links redirected, and worked examples replaced. The reasoning survived intact; everything vault-specific did not. "Copy and prune" understated this.
2. FORK-LATE WAS THE WRONG CALL, TWICE. Both pages judged "general enough in substance to copy" arrived with zero valid sources. Copying is only safe where provenance is already valid in the destination — which was never true here.
3. TAG-AS-PAGE LINKING RECURRED ~18 TIMES. Always in a See Also list or inline alias, always while reaching for a related concept not already linked on that page. Write-time lint caught every one. Not a lapse that effort fixes; needs a mechanical guard.
4. THE LINT'S REVERSE HINT WORKS. "this is a PAGE, not a tag" caught page names used as tags — the mirror error — on event-prediction and trajectory-prediction.
5. TAXONOMY-BEFORE-PAGES HELD. Nine tags added ahead of the pages using them: theory-based-modelling, model-decomposition, event-prediction, trajectory-prediction, spatiotemporal, volatility, camera-calibration, projective-geometry, image-alignment, radial-distortion.
6. SCOPE HAS NO NATURAL STOPPING POINT. Each new page referenced further missing pages. random-forest and feature-attribution were deliberately left uncreated rather than expanding the imported: tier indefinitely.

=== OUTSTANDING ===
- 115 of 179 pages are draft
- random-forest, feature-attribution referenced but absent
- Acquisition table in the overview is the actionable next step


