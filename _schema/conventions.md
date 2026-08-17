# Conventions

## Naming

- Filenames: `lowercase-kebab-case.md`
- One concept per page
- Split pages that grow beyond ~1000 words covering
  multiple distinct ideas

## Linking

- Use `[[wikilinks]]` for all cross-references
- Every wiki page should link:
  - Downward to its sources
  - Sideways to related wiki pages
- Prefer `[[page-name|display text]]` when the link
  text should differ from the filename

**Check the target exists before linking.** Dead links are usually plausible concept names produced mid-sentence and bracketed without lookup — including names invented on the spot, not only tag names that happen to exist. Run `find_mentioned_but_missing` after any batch of page creation.

## Provenance Markers

Inline, at the claim. These are the mechanism that flags
risk — the frontmatter percentages say *how much*, the
markers say *which*.

- Plain text = extracted directly from a source
- `^[inferred: reason]` = synthesised from held sources
- `^[generated: reason]` = a claim constructed here that
  **no source states**
- `^[imported: reason]` = brought from outside the held
  corpus — model knowledge, web search, general background
- `^[ambiguous: source A says X, source B says Y]` =
  sources disagree

Any marker may carry a `rests-on:` clause recording what
the claim stands on — see [Claim Dependencies](#claim-dependencies).

### Choosing between inferred and generated

The test is whether an author of a held source would
recognise the claim as theirs.

- **Inferred** — a fair gloss, restatement, or comparison
  that follows from what sources say.
- **Generated** — a novel claim, reconciliation, or
  mechanism that exists nowhere but here.

The boundary case is a **comparison that produces a new
conclusion**. Noting that two sources disagree is inferred.
Diagnosing *why* they disagree, or resolving it, is
generated.

### Why imported is separate

Imported claims look like knowledge and have no source in
`raw/` to check against. They are the hardest error to
catch, because nothing in the vault contradicts them.

Worked example: a page on BERT states that it adapts the
Transformer *decoder*. It adapts the **encoder** — the
whole point of the bidirectional objective is that it
drops the decoder's causal mask. That error is extremely
common in secondary coverage, reads fluently, and nothing
in a vault holding only the BERT paper would contradict a
page that asserts it.

Model families are the highest-risk area for this. Parameter
counts, context lengths, training corpora, and release
chronology are all widely repeated, frequently wrong, and
almost never checkable against a held source unless that
exact source is in `raw/`.

**Rule: never state an imported claim as fact.** Mark it,
or omit it.

## Claim Dependencies

A `^[generated]` marker records *that* a claim was invented
here. It does not record what it was invented **on top of**
— so when a premise is revised, nothing finds what else
moves.

> **On the examples below.** This vault is new and has no
> correction history of its own yet. The worked examples are
> drawn from the published literature rather than from pages
> held here, and are illustrative — they show the shape of
> the failure, not something that happened in this vault.
> Replace them with real local cases as they accumulate; a
> correction that actually occurred here will teach the
> mechanism better than a borrowed one.

### Claim IDs

A generated claim referenced from more than one page gets a
short kebab-case ID, declared once at its home page:

> **`scale-substitutes-for-architecture`** — where a task
> is bottlenecked by data rather than inductive bias,
> architectural sophistication buys less than parameter
> count.

The ID is the claim's handle. Backtracking is then a text
search for it. Claims used on one page only do not need one.

### rests-on

Dependent claims record what they stand on, appended to the
marker:

```
^[generated: compute is better spent on parameters than
 on data at fixed budget.
 rests-on: source:kaplan-scaling-exponents]
```

Four dependency kinds, and they fail differently:

| Kind | Means | Fails when |
|---|---|---|
| `source:` | A held source states it | The source is misread, or superseded by a better one |
| `claim:` | Another generated claim | That claim is revised — **cascades** |
| `imported:` | Outside the corpus | Any time. Nothing here can check it |
| `absence:` | **No source does X** | **A source is acquired** |

### The two searches, and why they differ

`rests-on:` records dependencies **one way**, so finding
what a change affects needs the *opposite* search from
finding where a claim appears:

| To find | Search for | Answers |
|---|---|---|
| Where a claim is **used** | the claim ID | Which pages state it |
| What **depends on** a claim | `rests-on: claim:<id>` | What breaks if it is revised |
| All claims that **cascade** | `rests-on: claim:` | The blast radius, vault-wide |
| All claims with an **expiry date** | `absence:` | What a new source could overturn |

Searching the ID alone is the common mistake. It finds the
references and misses the dependents, which are exactly the
pages a revision needs to reach.

### Absence is the dangerous kind

Absence claims are the kind most often overturned in practice, because the overturning event is the ordinary business of ingesting a source rather than any act of review.

This domain expires absence claims faster than most. Two
patterns from the literature:

- *"Emergent abilities appear discontinuously at scale"* —
  rested on benchmark results, and on no source having
  examined the metrics producing them. Schaeffer et al.
  (2023) argued the discontinuity is an artifact of
  discontinuous scoring, and that smooth metrics show
  smooth improvement. The claim did not survive a source
  arriving in the gap it depended on.
- *"No open model matches closed models on X"* — a whole
  class of claim with a shelf life measured in months.

**Rule: a claim marked `absence:` must be re-checked
whenever a source is ingested in its area.** This belongs
in the ingest checklist, not only here.

Note that narrowing beats deleting. *"No paper reports
variance across random seeds"* falls to a single
counter-example. Narrowed to *"no paper reports seed
variance for its headline configuration"* it survives
most of them — and the narrowed version is the more
useful finding, because it locates where the reporting
gap actually is. **A narrowed absence claim locates the
boundary; a deleted one loses the finding.**

### Worked example

Take a vault holding Kaplan et al. (2020) on scaling laws.
A reasonable generated claim: *"at a fixed compute budget,
parameters are the better marginal investment than data."*
That is not stated in those words by the paper, but it
follows from its exponents — so it is generated, and it
rests on `source:kaplan-scaling-exponents`.

Downstream pages then build on it. A page on model sizing
recommends favouring width. A synthesis explains an
architecture's design choices by that logic. Neither
restates the original claim; both depend on it.

Hoffmann et al. (2022) then revise the exponents. Models
at the time were substantially undertrained: parameters and
data should scale in far closer proportion, and a 70B model
trained on more data outperformed a 280B model trained on
less. **The scaling law did not fall — its exponents moved**,
and everything resting on the old ratio moved with them.

Without `rests-on:`, the two downstream pages are found only
by remembering they exist. With it, `rests-on:
source:kaplan-scaling-exponents` returns them directly.

Note the shape of the failure. The premise was not wrong in
kind, only in quantity — which is the harder case, because a
reversal is noticed and a re-estimate is not.

### What this does not solve

Finding dependents relies on the ID being used
consistently. There is no automatic cascade and no
integrity check.

Note also that the lint helper `_link_target` currently
strips heading anchors, so if claim IDs are ever expressed
as `[[page#claim]]` they will be invisible to
`find_backlinks` and `find_mentioned_but_missing`. Plain
text IDs avoid this.

## Frontmatter Provenance

Page-level proportions, summing to 100:

```yaml
provenance:
  extracted: 60%
  inferred: 25%
  generated: 10%
  imported: 0%
  ambiguous: 5%
```

`generated` and `imported` default to 0 if absent, so pages
written before this convention remain valid.

**A percentage is not sufficient on its own.** Risk is not
proportional to volume — one wrong generated claim that
propagates across pages does more damage than thirty
percent harmless glossing.

## Supersession

When new info contradicts old, don't silently overwrite.
Record it:

> **Superseded**: This section previously stated X based
> on [[source-a]]. As of [[source-b]] (YYYY-MM), the
> current understanding is Y.

Generated and imported claims are **more likely** to need
supersession than extracted ones, and are the first place
to look when a contradiction surfaces.

### Before closing a supersession, follow the dependencies

A superseded claim may be load-bearing elsewhere. Two
searches, not one:

1. **Search the claim ID** — finds pages that *state* it.
   Update each.
2. **Search `rests-on: claim:<id>`** — finds claims that
   *depend* on it. These are the ones a revision silently
   breaks, and they will not appear in the first search.

Then decide whether each dependent survives. A dependent
may weaken rather than fall: a claim resting on the original
scaling exponents does not become false when those exponents
are re-estimated — it becomes **conditional on a superseded
fit**, and should be marked so rather than deleted.

**Record the weakening rather than deleting the claim.**
The same reasoning as narrowing an absence claim — a
qualified claim keeps the finding and its caveat together;
a deleted one loses both.

## Lifecycle States

draft → reviewed → verified → stale → archived

- **draft**: Created from a single source. Low confidence. 
- **reviewed**: Human has read and confirmed it's reasonable.
- **verified**: Multiple sources confirm. High confidence. 
- **stale**: Not updated in 90+ days or superseded. 
- **archived**: Explicitly outdated. Kept for history.

Note that **verified is unavailable to a generated claim**
by definition — no source can confirm what no source
states. A page whose central claim is generated should not
reach `verified` on the strength of its extracted material.
Where a generated claim can be tested, a `question` page is
the appropriate home for it.

**Archived pages are intentionally orphaned.** They are
superseded duplicates kept for history, so they are
expected to have no inbound links, and `find_orphan_pages`
skips them. They remain link *sources* — their outbound
links still count — so a page whose only referrer is
archived is a genuine orphan.
