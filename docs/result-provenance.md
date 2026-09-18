# Result Provenance

Every search hit in Treeloom carries enough information to trace it back to its
exact repo, commit, and file range without a second API call. This page is for
anyone consuming search results programmatically (an agent, a CI bot, a
dashboard). It explains the provenance fields, the citation grammar, how to
build permalinks, and how to opt in to live staleness checking.

---

## 1. Why provenance?

A code search result is only as useful as its traceability. Without attribution
you know *what* was returned but not *when it was indexed* or *whether the
commit it came from is still the tip of the branch*. Treeloom captures the
indexing commit SHA at write time and attaches citation tokens and a
`sources` provenance map to every search response — so a CI bot, a review
agent, or a dashboard can render a permalink or raise a staleness alert from
a single response payload.

---

## 2. What's in a search response

`/search` (and `/graph-explore`) responses include two provenance surfaces:

- **Per-chunk**: an optional `citation` string on each `ChunkHit`.
- **Top-level `sources` map**: one entry per distinct source ID seen in the
  response, carrying the commit SHA and the metadata needed to construct
  permalinks and check freshness. The commit SHA lives only here, keyed by
  `source_id`, rather than being repeated on every chunk.

### Annotated example

```json
{
  "source_id": "a3f9c1b2e4d50f7a",
  "chunks": [
    {
      "file_path": "packages/Button/src/Button.tsx",
      "start_line": 10,
      "end_line": 42,
      "language": "tsx",
      "score": 0.94,
      "snippet": "export function Button({ variant, ...props }) { ...",
      "source_id": "a3f9c1b2e4d50f7a",
      "graph_enhanced": false,
      "citation": "material-ui@a1b2c3d4e5f6:packages/Button/src/Button.tsx:10-42"
    }
  ],
  "neighbors": [],
  "community_summaries": {},
  "sources": {
    "a3f9c1b2e4d50f7a": {
      "commit_sha": "a1b2c3d4e5f6abcdef012345",
      "indexed_at": "2026-06-10T14:23:07.412000",
      "path": "",
      "url": "https://github.com/mui/material-ui.git",
      "branch": "main",
      "permalink_base": "https://github.com/mui/material-ui/blob/a1b2c3d4e5f6abcdef012345"
    }
  }
}
```

The `ChunkHit` contract is defined in `src/treeloom/domain/shared.py`.
`citation` defaults to `""` and is absent from the wire response when
provenance was not attached (e.g. the source has no git metadata). The model
also declares a `commit_sha` field for backward compatibility, but the search
path no longer populates it per chunk — read the SHA from `sources`. Use
`exclude_defaults=True` on `model_dump()` if you re-serialize hits downstream.

### `sources` entry fields

| field | always present | notes |
|---|---|---|
| `commit_sha` | yes | full SHA captured at index time; empty string when unknown |
| `indexed_at` | yes | ISO 8601 UTC timestamp of the last completed index job |
| `path` | yes | local filesystem path, or `""` for URL-only sources |
| `url` | yes | remote git URL, or `""` for local-path sources |
| `branch` | yes | branch at index time, or `""` |
| `permalink_base` | no | only present when **both** `url` and `commit_sha` are non-empty |
| `is_stale` | no | only when `check_staleness=true` in the request |
| `current_sha` | no | only when `check_staleness=true` in the request |

---

## 3. Citation token grammar

```
<repo>@<sha12>:<file_path>:<start>-<end>
```

Where `<sha12>` is the first 12 hex characters of `commit_sha` and `<repo>`
is the `repo_label` — derived by `domain/provenance.py:repo_label()`:

1. URL basename with trailing `.git` stripped (e.g. `material-ui`)
2. Fallback: filesystem path basename
3. Fallback: `source_id`

**Degraded forms** when information is missing:

| situation | example citation |
|---|---|
| Full metadata | `material-ui@a1b2c3d4e5f6:src/Button.tsx:10-42` |
| No commit SHA | `bitcoin_bitcoin:src/validation.cpp:100-140` |
| No line range (`start_line == end_line == 0`) | `myrepo@a1b2c3d4e5f6:src/Foo.py` |
| No label, no sha, no lines | `src/Foo.py` |

The citation is always non-empty: at minimum it contains `file_path`.

---

## 4. Building a permalink

`permalink_base` holds everything up to (but not including) the file path:

```
<url-without-.git>/blob/<commit_sha>
```

Append the file path and optional anchor to get a clickable link:

```
{permalink_base}/{file_path}#L{start_line}-L{end_line}
```

This format is compatible with GitHub, Gitea, and Forgejo's blob viewer.

Example — given a sources entry with:
```json
"permalink_base": "https://github.com/mui/material-ui/blob/a1b2c3d4e5f6abcdef012345"
```
and a chunk with `file_path = "packages/Button/src/Button.tsx"`, `start_line = 10`,
`end_line = 42`:

```
https://github.com/mui/material-ui/blob/a1b2c3d4e5f6abcdef012345/packages/Button/src/Button.tsx#L10-L42
```

`permalink_base` is **absent** when the source was indexed from a bare local
path (no `url`) or when no `commit_sha` was captured. In those cases the
citation still identifies file and line range; there is just no stable URL to
link to.

---

## 5. Staleness at query time

By default provenance uses the commit SHA captured at index time — no live
network call. To also know whether the indexed commit is still current, add
`check_staleness: true` to the `/search` request body:

```bash
curl -X POST http://localhost:8001/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "feature flag evaluation",
    "source_id": "a3f9c1b2e4d50f7a",
    "check_staleness": true
  }'
```

When `check_staleness` is true, each `sources` entry gains two additional
fields:

```json
"is_stale": true,
"current_sha": "d9e8f7c6b5a4..."
```

`is_stale` is `true` when `current_sha != commit_sha`, `false` when they
match, and `null` when staleness cannot be determined (non-git source or
network error). The check calls `git rev-parse` for local paths or
`git ls-remote` for remote URLs — one round-trip per distinct source — so
there is a latency cost on multi-source responses. Leave it off for
latency-sensitive production queries.

For cheap age signals without a live check, `indexed_at` is always present
and shows when the source was last indexed.

For a standalone staleness check outside of a search request:

```bash
curl http://localhost:8001/sources/{source_id}/staleness
```

---

## 6. MCP

Provenance flows through the MCP tools `search_code` and
`search_code_enhanced` in both output formats.

### `response_format="markdown"` (default)

Each chunk heading gets the citation appended after a dash:

```
### src/Button.tsx:10-42 (score 0.94) — material-ui@a1b2c3d4e5f6:src/Button.tsx:10-42
```

And a `provenance:` block is appended at the end of the response listing
each source's truncated SHA, its `permalink_base`, and a `STALE` marker when
`is_stale` is true.

### `response_format="json"`

The returned dict includes the top-level `sources` map and per-chunk
`citation` fields — identical to the `/search` HTTP response documented above.

### `check_staleness` parameter

Both `search_code` and `search_code_enhanced` accept a `check_staleness: bool
= False` parameter that maps directly to the same flag on the underlying
`/search` request.

### `find_definition` does not carry provenance

`/find-definition` returns entity dicts (name, signature, file path) as stored
in the graph. Source membership is a `(:Source)-[:CONTAINS]->(e:Entity)` graph
relationship, not a property on the entity node, so there is no `source_id` to
resolve provenance from. If you need a citable result, use `search_code`
instead.

---

## 7. Caveat: requires indexed metadata

Provenance is only as rich as what was captured at index time.

| source type | citation | permalink |
|---|---|---|
| Git URL, commit captured | full `repo@sha12:path:lines` | yes (`permalink_base` present) |
| Local git path, commit captured | `dirname@sha12:path:lines` | no (no URL) |
| Local path, no git metadata | `dirname:path:lines` (no sha) | no |

If `commit_sha` is empty — a source indexed from a plain directory that is not
a git repo, or indexed before the provenance feature landed — the citation
degrades to `repo:path:lines` and there is no permalink. Re-index with a git
URL (or from a git-tracked local path) to get full attribution.

The `indexed_at` timestamp is always populated regardless of git metadata.
