# Fleet Operations

This guide is for operators running Treeloom across many repositories (tens to
hundreds) who don't want to index and monitor each one by hand. It covers:

1. **Bulk onboarding.** Index a whole list of repos from one manifest file.
2. **Scheduled refresh.** Automatically re-index repos whose git HEAD has moved.
3. **Fleet health rollup.** One call that answers "is everything indexed and healthy?"
4. **Scaling envelope.** How throughput scales, for capacity planning.

Terminology: a **source** is one indexed input (a local path or git URL, plus
branch), identified by a `source_id` derived from those values. A **fleet** is
simply the set of all sources one indexer knows about.

Everything here is built on existing per-source features, with no extra storage:

- `POST /index-repo` skips sources that are already indexed, unless you pass `force`.
- `GET /sources/{id}/staleness` compares the indexed commit to the current HEAD.
- The `python -m treeloom.rebuild_graphs` CLI rebuilds the code graph for many sources in bulk.
- The **freshness sampler**, a background loop in the indexer, publishes
  per-source staleness to Prometheus. See
  [incremental-indexing.md §3](incremental-indexing.md#3-freshness-metrics).

The main [README](../README.md) covers the rest of the indexer API (`CLAUDE.md` at
the repo root holds the maintainers' more detailed notes).

---

## 1. Bulk onboarding from a manifest

A **manifest** declares a set of repositories to index in one command. Re-running
an unchanged manifest is a **no-op** — `/index-repo` short-circuits sources that
are already `done` (pass `--force` to re-index regardless).

### Manifest format (YAML or JSON)

`yaml.safe_load` parses both. Two shapes are accepted.

Mapping form with an optional shared `defaults` block merged into every entry:

```yaml
defaults:
  branch: main
  skip_graph: false
repos:
  - path: ~/source/featbit
  - url: https://github.com/acme/widgets.git
    branch: release
    skip_patterns: ["vendor/**", "node_modules/**"]
  - url: https://github.com/acme/legacy.git
    force: true            # always re-index this one
```

Bare-list form (no defaults):

```yaml
- path: ~/source/a
- url: https://github.com/acme/b.git
```

Per-entry keys (all optional except exactly one of `path`/`url`):

| key            | meaning                                                        |
| -------------- | -------------------------------------------------------------- |
| `path`         | local filesystem path (mutually exclusive with `url`)          |
| `url`          | remote git URL (validated server-side by `_is_safe_git_url`)   |
| `branch`       | git branch                                                     |
| `skip_patterns`| glob patterns to skip (honored only when `TREELOOM_FEATURE_SKIP_PATTERNS=1`) |
| `skip_graph`   | index chunks/embeddings only; build the code graph later with `POST /index-graph` |
| `force`        | re-index even if already `done` (OR'd with the CLI `--force`)  |

An entry must carry **exactly one** of `path`/`url`; unknown keys are rejected
so typos surface loudly.

### Onboard CLI

```bash
# Preview without calling the indexer
python -m treeloom.fleet onboard repos.yaml --dry-run

# Index everything, then wait for all jobs to reach a terminal state
python -m treeloom.fleet onboard repos.yaml --wait

# Re-index the whole manifest
python -m treeloom.fleet onboard repos.yaml --force --wait
```

Output rolls up `enqueued / already up-to-date / failed`. Because onboarding is
idempotent, a second run of an unchanged manifest reports every repo
"up-to-date" and enqueues nothing.

The repos in one run are grouped under a job group named after the manifest
file. Use `--label` to pick a different name. `--poll-interval` sets how often
`--wait` checks job status (default 2 s).

Auth: when `AUTH_ENABLED=true`, `/index-repo` needs a bearer token. Put it in
`--key-file` (preferred), pass `--key -` to read it from stdin, or set
`TREELOOM_KEY` (falls back to `TREELOOM_MCP_API_KEY`). A token passed literally
as `--key <token>` shows up in `ps` output and shell history. Point at a
non-default indexer with `--indexer-url` (or `$INDEXER_URL`; default
`http://localhost:8001`).

---

## 2. Scheduled staleness-driven refresh

An **in-process background loop** in the indexer that re-indexes sources whose
git HEAD has moved, on a cadence, so the fleet stays fresh without manual
`force`. **Default OFF** — the operator opts in.

| env var                          | default | meaning                                            |
| -------------------------------- | ------- | -------------------------------------------------- |
| `FLEET_AUTO_REFRESH_ENABLED`     | `false` | enable the loop                                    |
| `FLEET_REFRESH_INTERVAL_SECONDS` | `900`   | seconds between ticks                               |
| `FLEET_REFRESH_MAX_PER_TICK`     | `25`    | cap re-indexes **enqueued** per tick (bounds bursts) |
| `FLEET_REFRESH_MAX_SOURCES`      | `200`   | cap staleness **checks** per tick (see rotation below) |

Each tick lists sources, computes staleness (`compute_source_staleness` — the
same git HEAD comparison `/sources/{id}/staleness` uses), and enqueues a
re-index for the stale ones, **oldest-indexed first**, capped at
`FLEET_REFRESH_MAX_PER_TICK`. Only sources with a determinable, *moved* HEAD are
selected; non-git / directory sources (staleness `null`) are skipped. The
single-active-job-per-source invariant prevents pile-ups, and a source with an
active job is skipped. Every enqueue increments the
`treeloom_fleet_refresh_enqueued_total` counter.

**Re-index is kind-faithful.** A source is re-indexed as the **same kind** it
was first indexed (`repo` / `directory` / `file`), recovered from the persisted
`source_records.kind` column. This matters: re-indexing a file-indexed source as
a `repo` job would fail (a file isn't a directory), and a directory-indexed
source would silently switch to repo-walk semantics. Legacy rows that predate
the `kind` column fall back to filesystem shape (a file path → a `file` job).

**Staleness checks are capped per tick.** Each URL source costs a `git
ls-remote` round-trip, so above `FLEET_REFRESH_MAX_SOURCES` the loop checks only
a **rotating window** of that size each tick (advancing the offset across ticks
so the whole fleet is covered over several ticks) instead of firing hundreds of
serial git calls every interval. This mirrors the `/fleet` endpoint's
`FLEET_STALENESS_MAX_SOURCES` guard.

Because each tick caps at `FLEET_REFRESH_MAX_PER_TICK`, a fleet-wide HEAD move
(a mass rebase) drains over several ticks rather than flooding the queue at
once. Size the interval and cap to your indexing throughput.

---

## 3. Fleet health rollup

One view answers "is the fleet healthy?" across every source **without
per-source API calls**.

```bash
GET /fleet                              # cheap: registry + latest-job join
GET /fleet?summary_only=true            # just the aggregate block
GET /fleet?staleness=true               # + live per-source git HEAD comparison
GET /fleet?entities=true                # + per-source graph entity counts
```

Response shape:

```jsonc
{
  "summary": {
    "sources_total": 312,
    "sources_with_errors": 4,
    "sources_graph_missing": 1,
    "total_chunks": 481203,
    "total_files": 39112,
    "oldest_index_age_seconds": 864000,
    "sources_stale": 7,          // only when ?staleness=true
    "total_entities": 91022,     // only when ?entities=true
    "staleness_truncated": true  // see cap below
  },
  "sources": [
    { "id": "...", "label": "https://github.com/acme/b.git", "branch": "main",
      "indexed_at": 1718000000, "age_seconds": 3600, "chunk_count": 1200,
      "file_count": 84, "graph_indexed": true, "entity_count": 410,
      "is_stale": false,
      "last_job": { "job_id": "...", "status": "done", "error": "", "errors": 0,
                    "finished_at": 1718000123.0 } }
  ]
}
```

Design notes:

- **Cheap by default.** The base call joins the source registry (the
  `source_records` Postgres table) with the latest job per `source_id` (for `last_job` / last-error). No git, no graph.
- **`?staleness=true` is capped** at `FLEET_STALENESS_MAX_SOURCES` (default
  `200`). Above the cap, live resolution would mean hundreds of `git ls-remote`
  calls in one request — a self-inflicted DoS — so it is skipped and
  `summary.staleness_truncated` is set. Use the freshness sampler's Prometheus
  gauges (`treeloom_source_index_stale`, `treeloom_sources_stale_total`) for
  staleness at scale instead.
- **`?entities=true` is resilient.** It runs one aggregate graph-store query
  (not N). If the graph store is down, entity counts are simply absent — the
  rollup still answers fleet health from the rest.
- **Auth + visibility mirror `/sources`.** When `AUTH_ENABLED=true`, non-admins
  see their own sources plus legacy (pre-auth) records; admins see all.

### Health CLI

```bash
python -m treeloom.fleet health                       # table
python -m treeloom.fleet health --staleness --entities
python -m treeloom.fleet health --json                # raw rollup
```

---

## 4. Horizontal-scale envelope

How a full-stack deployment scales, for capacity planning.

**Indexing throughput** is `INDEX_CONCURRENCY` workers (default 8) per indexer process ×
however many processes share the **Postgres-backed `job_queue`**. Workers pop
with `DELETE ... FOR UPDATE SKIP LOCKED`, so N processes share one queue without
double-running a job — scale out by adding indexer processes against the same
`DATABASE_URL`. Backpressure: the webhook sheds (HTTP 429) at
`MAX_QUEUE_DEPTH` (default 500).

**Per-source ceiling.** At most one active job per `source_id`
(`jobs_active_source_uniq` partial unique index). A single repo cannot be
indexed by more than one worker concurrently; fleet-wide parallelism comes from
having **many distinct sources**, not many workers on one source.

**Observability cardinality.** The freshness sampler and
`/fleet?staleness=true` both guard against label/`git ls-remote` explosion at
**~200 sources** (`FRESHNESS_MAX_SOURCES`, `FLEET_STALENESS_MAX_SOURCES`). Above
that, per-source staleness collapses to aggregates. Plan dashboards around the
aggregate gauges for large fleets.

**Auto-refresh load.** `FLEET_REFRESH_MAX_PER_TICK` × (sources stale per
interval) sets the steady-state re-index rate the loop adds on top of webhook
traffic. Keep it under your worker throughput so refresh doesn't starve
interactive/webhook jobs.

**Simple mode is not a scale tier.** `TREELOOM_PROFILE=simple` (embedded SQLite
graph + asyncio lock, embedded LanceDB) is **single-writer** by design — it has
no Postgres queue and no cross-process worker pool. Use the full stack (Postgres
queue + Milvus + Neo4j) for fleet-scale deployments. See
[simple-mode.md](simple-mode.md).
