---
title: "Wiki Health Dashboard"
type: dashboard
created: 2026-04-22
updated: 2026-04-22
---

# Wiki Health Dashboard

## Stale Pages (not updated in 120+ days)

```dataview
TABLE confidence, lifecycle, updated, file.size AS "Size"
FROM "wiki"
WHERE updated
  AND date(today) - date(updated) > dur(120 days)
  AND type != "dashboard"
SORT updated ASC
```

## Low Confidence Pages (below 0.5)

```dataview
TABLE confidence, sources, lifecycle
FROM "wiki"
WHERE confidence AND confidence < 0.5
  AND type != "dashboard"
SORT confidence ASC
```

## Draft Pages (need review)

```dataview
TABLE created, sources, confidence
FROM "wiki"
WHERE lifecycle = "draft"
  AND type != "dashboard"
SORT created ASC
```

## Recently Ingested (last 14 days)

```dataview
TABLE type, confidence, tags
FROM "wiki"
WHERE created
  AND date(today) - date(created) < dur(14 days)
  AND type != "dashboard"
SORT created DESC
```

## Orphan Risk (no outgoing links)

```dataview
TABLE confidence, lifecycle
FROM "wiki"
WHERE length(file.outlinks) = 0
  AND type != "dashboard"
SORT file.name ASC
```

## Knowledge by Type

```dataview
TABLE length(rows) AS "Count"
FROM "wiki"
WHERE type != "dashboard"
GROUP BY type
SORT length(rows) DESC
```

## Tag Distribution

```dataview
TABLE length(rows) AS "Page Count"
FROM "wiki"
WHERE type != "dashboard"
FLATTEN tags AS tag
GROUP BY tag
SORT length(rows) DESC
```