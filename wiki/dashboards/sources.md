---
title: "Source Tracking Dashboard"
type: dashboard
created: 2026-04-22
updated: 2026-04-22
---

# Source Tracking Dashboard

## All Sources

```dataview
TABLE file.ctime AS "Added", file.size AS "Size"
FROM "raw"
SORT file.ctime DESC
```

## Sources by Folder

```dataview
TABLE length(rows) AS "Count"
FROM "raw"
GROUP BY file.folder
SORT length(rows) DESC
```

## Most-Referenced Sources

```dataview
TABLE length(file.inlinks) AS "Referenced By"
FROM "raw"
WHERE length(file.inlinks) > 0
SORT length(file.inlinks) DESC
LIMIT 20
```