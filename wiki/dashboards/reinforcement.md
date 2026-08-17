---
title: "Reinforcement Dashboard"
type: dashboard
created: 2026-04-22
updated: 2026-04-22
---

# Reinforcement Dashboard

## Verified but Aging (60+ days without update)

```dataview
TABLE confidence, updated,
  date(today) - date(updated) AS "Days Since Update"
FROM "wiki"
WHERE lifecycle = "verified"
  AND date(today) - date(updated) > dur(60 days)
SORT updated ASC
```

## Single-Source Pages (fragile knowledge)

```dataview
TABLE sources, confidence
FROM "wiki"
WHERE sources AND length(sources) = 1
  AND type != "dashboard"
SORT file.name ASC
```

## Confidence Decay Watch (mid-range, at risk)

```dataview
TABLE confidence, lifecycle, updated
FROM "wiki"
WHERE confidence AND confidence > 0.3 AND confidence < 0.6
  AND lifecycle = "verified"
SORT confidence ASC
```