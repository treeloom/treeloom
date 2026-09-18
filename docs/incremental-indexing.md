# Incremental Indexing

This guide is for operators who want a Treeloom index to stay current as code
is pushed, without re-indexing whole repositories. It explains how a git
provider's push webhook turns into an **incremental job** (re-index only the
changed files), how failed jobs are retried and eventually parked in a
**dead-letter** state, which Prometheus metrics report freshness, how the
indexer sheds load, and what to do when something is stuck. Sections 8–10
cover the two load-test harnesses that ship in `scripts/`.

Only GitHub and Gitea webhooks are supported. A **source** is one indexed
repository (URL + branch), identified by its `source_id`.

For the general observability setup (tracing, log correlation, Grafana
dashboards) see [docs/observability-runbook.md](observability-runbook.md).

---

## 1. Webhook → incremental lifecycle

### 1.1 Webhook receipt

A git provider (GitHub or Gitea) POSTs a **merged pull-request** event
(`pull_request` with `action: closed` and `merged: true`; plain pushes and
unmerged PRs are acknowledged and ignored) to the Treeloom indexer's
`POST /webhook` endpoint. The handler (`handle_webhook` in
`src/treeloom/application/routes_webhook.py`):

1. **Authenticates** the delivery by its HMAC signature (`X-Hub-Signature-256`
   for GitHub, `X-Gitea-Signature` for Gitea) against `WEBHOOK_GITHUB_SECRET` /
   `WEBHOOK_GITEA_SECRET`. The endpoint returns 503 until the secret is set
   and 401 on a bad signature.
2. **Extracts** the base branch and the PR's changed-file list — fetched from
   the provider API (`/pulls/{n}/files`, authenticated with
   `WEBHOOK_GITHUB_TOKEN` / `WEBHOOK_GITEA_TOKEN`; Gitea also needs
   `WEBHOOK_GITEA_BASE_URL`) unless the load-test-only
   `WEBHOOK_TRUST_INLINE_FILES=1` is set (§9).
3. **Checks backpressure** — if the queue depth ≥ `MAX_QUEUE_DEPTH` (env,
   default 500), returns **HTTP 429** with a `Retry-After` header and
   increments `treeloom_webhook_shed_total`. The provider will retry.
4. **Caps the payload** — if `len(changed_files) > MAX_CHANGED_FILES` (env,
   default 5000), falls back to a **full re-index** job for the source instead
   of a giant incremental payload. This keeps the `jobs.payload` JSONB column
   bounded and avoids memory spikes in the worker that reads it.
5. **Persists the payload** — sets `job["payload"] = {"changed_files": [...],
   "branch": "<branch>", "webhook_accepted_at": <timestamp>}` and writes a
   `kind="incremental"` row to the `jobs` table (migration
   `017_jobs_payload.sql`: `payload JSONB`, `attempts INTEGER DEFAULT 0`).
6. **Enqueues** the `job_id` into `job_queue` via the same Postgres-backed
   queue used by `repo`/`directory`/`graph` jobs — `DELETE … FOR UPDATE SKIP
   LOCKED` so any worker process can pick it up.

The `webhook_accepted_at` timestamp carried in the payload is used later to
compute `treeloom_merge_to_searchable_seconds`.

### 1.2 Queue routing and worker pickup

Workers poll `job_queue` every `JOB_QUEUE_POLL_INTERVAL` seconds (default
1.0 s). When a worker pops an incremental job, `dispatch_job()` (in
`src/treeloom/application/indexer_runners.py`) reads `jobs.payload`, reconstructs the `changed_files` list and source info, and
calls `_run_incremental_index(source_id, repo_url, branch, changed_files, jd)`.

Up to `INDEX_CONCURRENCY` (default 8) jobs can run concurrently per process.
The Postgres `jobs_active_source_uniq` partial index prevents two workers from
running simultaneous jobs for the same source.

### 1.3 Incremental runner

`_run_incremental_index` (same file) processes each changed file as a
delete-then-insert:

- **Deleted/renamed files**: removes the vector-store chunks
  (`delete_chunks_by_file`) and graph entities (`delete_entities_by_file`) for
  the old path.
- **Added/modified files**: re-indexes via `_process_file()` (parse → embed →
  Milvus insert → graph extract). If a file was previously indexed its old
  chunks are deleted first so results are never stale.

The runner observes:
- `treeloom_incremental_job_duration_seconds` — wall-clock time of the full run.
- `treeloom_incremental_jobs_total{status="done|failed|dead_letter"}` — bumped
  on completion.
- `treeloom_merge_to_searchable_seconds` — computed as `now -
  payload["webhook_accepted_at"]` on successful completion. This is the
  end-to-end latency from git push to searchable.

### 1.4 Resume on restart

Because the payload is persisted in `jobs.payload`, a worker crash or
deliberate restart does **not** lose the job. On startup
(`src/treeloom/application/lifecycle.py`) the indexer finds `RUNNING`/`QUEUED`
jobs in `JobStore` and re-enqueues them. The
incremental runner is idempotent (delete-then-insert per file), so re-running
from the start of a partially-completed job is safe — no chunks are duplicated.

> Earlier versions kept the changed-file list only in memory, so a restart
> marked the job `failed` and relied on the provider to re-send the webhook.
> Persisting the payload removed that dependency.

---

## 2. Retry and dead-letter flow

### 2.1 Automatic retry

When an incremental job fails (unhandled exception in the runner), the queue
worker:

1. Increments `jobs.attempts`.
2. If `attempts < MAX_JOB_ATTEMPTS` (env, default 3): re-enqueues the job.
   The job_id is re-inserted into `job_queue`; the next available worker picks
   it up. Retries are **not** delayed in v1 (no `next_attempt_at` column) — a
   simple attempt cap is used. Each retry is fully idempotent.
3. If `attempts >= MAX_JOB_ATTEMPTS`: sets `jobs.status = "dead_letter"`,
   persists, and increments `treeloom_dead_letter_jobs_total`. The job is NOT
   re-enqueued.

> **Don't retry on structural errors** (e.g. the source repo URL is gone).
> A simple attempt cap means the job will reach dead_letter in
> `MAX_JOB_ATTEMPTS` tries. Inspect the error in `GET /jobs/{id}/errors`
> before re-queuing.

### 2.2 Dead-letter endpoints

Both endpoints require admin authentication when `AUTH_ENABLED=true`.

**`GET /jobs/dead-letter`**

Lists all jobs in `dead_letter` state. Response fields per job:
- `job_id`, `source_id`, `source_url`, `source_branch`
- `attempts` — number of times the job was tried (always `MAX_JOB_ATTEMPTS`)
- `error` — last error message
- `finished_at` — when it entered dead_letter

**`POST /jobs/{job_id}/retry`**

Re-queues a dead-letter (or any terminal) job. Behaviour:
- Resets `jobs.status` to `queued`.
- Keeps `jobs.attempts` as-is (does NOT reset to 0 — the job will enter
  dead_letter again on the next failure).
- Re-inserts `job_id` into `job_queue`.
- Returns **404** if the job is not found; **409** if the job is not in a
  terminal or dead-letter state.

### 2.3 Startup recovery and dead-letter exclusion

`DEAD_LETTER` jobs are terminal — they are **never** automatically re-enqueued
on indexer restart. Only `POST /jobs/{id}/retry` can re-queue them. This is
intentional: a dead-letter job has already failed multiple times and may
require operator intervention before re-running.

---

## 3. Freshness metrics

All metrics are defined in `src/treeloom/infrastructure/metrics.py`. The exact
names below are the contract consumed by `assets/grafana/treeloom-freshness.json`
and `assets/prometheus/freshness-alerts.yml` — **do not rename without updating
both files**.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `treeloom_source_index_stale` | Gauge 0/1 | `source_id` | 1 if indexed SHA ≠ current HEAD |
| `treeloom_source_index_age_seconds` | Gauge | `source_id` | Seconds since `source.indexed_at` |
| `treeloom_incremental_jobs_total` | Counter | `status` (done\|failed\|dead_letter) | Incremental job completions by outcome |
| `treeloom_incremental_job_duration_seconds` | Histogram | — | Runner wall-clock time (webhook pop → done/failed) |
| `treeloom_merge_to_searchable_seconds` | Histogram | — | Webhook-accept → incremental job done |
| `treeloom_queue_saturation` | Gauge 0..1 | — | `min(depth / MAX_QUEUE_DEPTH, 1.0)` |
| `treeloom_webhook_shed_total` | Counter | — | Webhooks rejected with 429 |
| `treeloom_dead_letter_jobs_total` | Counter | — | Jobs entering dead_letter state |

### 3.1 Lag sampler

A background `asyncio` task started in the indexer `startup` hook samples
per-source staleness on a configurable interval:

- **`FRESHNESS_SAMPLER_ENABLED`** (default `true`) — set to `false` to disable.
- **`FRESHNESS_SAMPLE_INTERVAL_SECONDS`** (default `60`) — how often the task
  loops. Each tick: list all sources, call `compute_source_staleness()` (the
  shared helper extracted from `GET /sources/{id}/staleness`), set
  `treeloom_source_index_stale` and `treeloom_source_index_age_seconds` per
  source.
- Per-source errors are caught and logged; one bad remote (e.g. a deleted git
  repo) does not stall the loop.
- **Cardinality guard**: only label by `source_id` up to `FRESHNESS_MAX_SOURCES`
  (default 200). Above that threshold, per-source labels are dropped and an
  aggregate `treeloom_sources_stale_total` gauge is exported instead.

### 3.2 Why age, not commit-lag

`git ls-remote` returns a SHA, not a timestamp. The number of commits behind
HEAD is not reliably computable across all git providers without a local clone.
Instead, Treeloom uses:

- `treeloom_source_index_stale` (stale flag: indexed SHA ≠ current HEAD)
- `treeloom_source_index_age_seconds` (age of the last index, i.e. how long
  ago the source was last successfully indexed)

Together these answer "is this source stale and for how long?" which is the
actionable signal for freshness alerts.

---

## 4. Backpressure

### 4.1 Queue depth and 429

The Postgres `job_queue` depth is exposed as `treeloom_queue_saturation` in
[0, 1]:

```
saturation = min(current_depth / MAX_QUEUE_DEPTH, 1.0)
```

When a new webhook arrives and `depth >= MAX_QUEUE_DEPTH`, the handler:
1. Returns **HTTP 429** with `Retry-After: 60` (providers treat this as
   transient and redeliver).
2. Increments `treeloom_webhook_shed_total`.
3. Does **not** enqueue the job.

`/health`, `/search`, and `/status` are unaffected by queue saturation.

Environment variables:
- **`MAX_QUEUE_DEPTH`** — maximum `job_queue` rows before 429s begin (default
  500).
- **`MAX_CHANGED_FILES`** — per-webhook changed-file cap; above this the job
  becomes a full re-index (default 5000).

### 4.2 Oversized push → full re-index

When a webhook payload contains more than `MAX_CHANGED_FILES` changed files
(e.g. a mass rename or branch switch), enqueuing a giant `changed_files` array
in `jobs.payload` would waste memory and JSONB storage. Instead, Treeloom
falls back to a standard full `repo` re-index job for the source. The job is
bounded, idempotent, and resumable. The webhook returns `202 Accepted` (not
429) so the provider does not retry.

---

## 5. Operator runbook

### 5.1 Source stuck stale

**Symptom**: `treeloom_source_index_stale{source_id="…"} == 1` for > 1 hour
after a code push.

**Checklist**:
1. Is the webhook configured? Check the git provider's delivery log for the
   relevant push event.
2. Did the webhook reach the indexer? Look for `POST /webhook` in the indexer
   logs. If absent, check network/firewall rules.
3. Is there an active job? `curl http://localhost:8001/status` or
   `GET /jobs?source_id=<id>`.
4. Is the source in dead-letter? `GET /jobs/dead-letter` (admin auth).
5. Is the queue saturated? Check `treeloom_queue_saturation` or `GET /status`.
6. Force a fresh full index:
   ```bash
   curl -X POST http://localhost:8001/index-repo \
     -H "Authorization: Bearer $TREELOOM_KEY" \
     -H "Content-Type: application/json" \
     -d '{"path": "/path/to/repo", "force": true}'
   ```
7. Verify staleness resolves with `GET /sources/<id>/staleness`.

### 5.2 Queue saturated

**Symptom**: `treeloom_queue_saturation > 0.8` sustained; webhooks returning
429; `treeloom_webhook_shed_total` rising.

**Checklist**:
1. How many jobs are queued? `GET /status` shows queue depth.
2. Are jobs completing? Watch `rate(treeloom_incremental_jobs_total{status="done"}[5m])`.
3. Are jobs failing fast (error loop)? Watch `treeloom_incremental_jobs_total{status="failed"}`.
4. Is the embedding service (TEI, HuggingFace Text Embeddings Inference)
   healthy? A dead TEI service causes every index job to stall.
5. **Short-term relief**: increase `INDEX_CONCURRENCY` (restart indexer with new
   env var) or add another indexer process (they share the Postgres queue).
6. **If the backlog is temporary** (e.g. a large batch push): raise
   `MAX_QUEUE_DEPTH` temporarily and restart. Git providers will redeliver shed
   webhooks.
7. **If jobs are in a tight error loop**: set `MAX_JOB_ATTEMPTS=1` temporarily
   so they reach dead_letter fast rather than flooding the queue with retries,
   then investigate the root cause.

### 5.3 Dead-letters piling up

**Symptom**: `increase(treeloom_dead_letter_jobs_total[15m]) > 0`; alert
`WebhookDeadLetters` firing.

**Checklist**:
1. List dead-letter jobs:
   ```bash
   curl -H "Authorization: Bearer $TREELOOM_KEY" \
     http://localhost:8001/jobs/dead-letter
   ```
2. Read the error for a specific job:
   ```bash
   curl -H "Authorization: Bearer $TREELOOM_KEY" \
     http://localhost:8001/jobs/<job_id>/errors
   ```
3. Common root causes:
   - Source repo deleted or moved (404 on git ls-remote)
   - TEI embedding service unreachable
   - Milvus unavailable
   - Malformed webhook payload (persisted before validation was added)
4. After fixing the root cause, re-queue:
   ```bash
   curl -X POST http://localhost:8001/jobs/<job_id>/retry \
     -H "Authorization: Bearer $TREELOOM_KEY"
   ```
5. **Do NOT restart the indexer** expecting dead-letter jobs to retry — they
   are excluded from startup recovery by design. Only `POST /jobs/{id}/retry`
   re-queues them.

### 5.4 Merge-to-searchable latency spike

**Symptom**: `MergeToSearchableSlow` alert; p95 of
`treeloom_merge_to_searchable_seconds` above 300 s.

**Checklist**:
1. Is the queue saturated? A full queue delays the incremental job start.
2. Is the job processing slowly? Check p95 of
   `treeloom_incremental_job_duration_seconds`.
3. Is TEI embedding latent? Check `tei-embedding` service logs and
   `http://localhost:8082/metrics`.
4. Are changed-file counts high? Large PRs approach `MAX_CHANGED_FILES` (5000)
   and embed many chunks per job.
5. Is HyDE/summary generation slow? `LLM_URL` latency adds to indexing time.
6. Mitigation: increase `INDEX_CONCURRENCY`; or, for batch pushes, accept
   temporarily higher latency and let the queue drain.

---

## 6. Configuration reference

| Variable | Default | Description |
|---|---|---|
| `MAX_QUEUE_DEPTH` | 500 | Webhook 429 threshold; also denominator of `treeloom_queue_saturation` |
| `MAX_CHANGED_FILES` | 5000 | Per-webhook changed-file cap; above this → full re-index |
| `MAX_JOB_ATTEMPTS` | 3 | Retry cap before entering dead_letter |
| `FRESHNESS_SAMPLER_ENABLED` | true | Enable the background lag-sampler task |
| `FRESHNESS_SAMPLE_INTERVAL_SECONDS` | 60 | Lag-sampler tick interval |
| `FRESHNESS_MAX_SOURCES` | 200 | Per-source label cardinality cap |
| `JOB_QUEUE_POLL_INTERVAL` | 1.0 | Seconds between queue poll cycles |
| `WEBHOOK_GITHUB_SECRET` | (unset → 503) | HMAC secret for GitHub deliveries |
| `WEBHOOK_GITHUB_TOKEN` | (unset) | Token for the GitHub `/pulls/{n}/files` call |
| `WEBHOOK_GITEA_SECRET` | (unset → 503) | HMAC secret for Gitea deliveries |
| `WEBHOOK_GITEA_BASE_URL` | (unset → 503) | Gitea instance URL for the files API call |
| `WEBHOOK_GITEA_TOKEN` | (unset) | Token for the Gitea files API call |
| `WEBHOOK_TRUST_INLINE_FILES` | off | Load testing only — read the file list from the payload (§9) |
| `INDEX_CONCURRENCY` | 8 | Max concurrent index jobs per indexer process |

---

## 7. Related assets

- **Grafana dashboard**: `assets/grafana/treeloom-freshness.json` — per-source
  staleness, incremental throughput, latency, queue saturation, dead-letters.
- **Prometheus alerts**: `assets/prometheus/freshness-alerts.yml` — four alert
  rules: `SourceIndexStale`, `MergeToSearchableSlow`, `IndexQueueSaturated`,
  `WebhookDeadLetters`.
- **Observability runbook**: [docs/observability-runbook.md](observability-runbook.md) —
  tracing setup, log correlation, Grafana trace dashboard.

## 8. Load testing

`scripts/loadtest_incremental.py` is a dependency-light CLI harness that
drives synthetic webhook traffic and measures **merge→searchable latency**
— the wall-clock time from when a webhook is accepted to when the changed
content is retrievable via `POST /search`.

### Prerequisites

| Requirement | Notes |
|---|---|
| Running Treeloom indexer | Any backend (Milvus/LanceDB, GPU or CPU) |
| At least N indexed sources | Harness picks the first `--repos` from `GET /sources` |
| `httpx` (stdlib-only otherwise) | Already a core treeloom dep — `pip install -e .` |
| Bearer token | Only when `AUTH_ENABLED=true`; pass `--token` or set `$TREELOOM_KEY` |

No GPU, no Milvus required — the simple profile (`TREELOOM_PROFILE=simple`)
works fine.

### Quick start

```bash
# Smoke test: 5 merges/min for 2 minutes, p95 target 120 s
python scripts/loadtest_incremental.py \
    --url http://localhost:8001 \
    --token "$TREELOOM_KEY" \
    --repos 2 \
    --rate 5 \
    --duration 120 \
    --files-per-merge 3 \
    --p95-target-seconds 120

# Sustained load: 30 merges/min for 10 minutes across 5 repos
python scripts/loadtest_incremental.py \
    --url http://localhost:8001 \
    --repos 5 \
    --rate 30 \
    --duration 600

# Dry run — print config and exit
python scripts/loadtest_incremental.py --dry-run
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8001` | Indexer base URL (env: `TREELOOM_INDEXER_URL`) |
| `--token` | `$TREELOOM_KEY` | Bearer token for auth |
| `--secret` | `$WEBHOOK_GITHUB_SECRET` | HMAC secret used to sign payloads; must match the indexer's `WEBHOOK_GITHUB_SECRET` (see §9) |
| `--repos N` | `2` | Number of pre-indexed sources to spread load across |
| `--rate MERGES/MIN` | `10` | Sustained webhook rate |
| `--duration SECONDS` | `120` | Test wall-clock duration |
| `--files-per-merge N` | `5` | Synthetic changed-file count per webhook |
| `--p95-target-seconds` | `300` | PASS/FAIL threshold for p95 latency |
| `--memory-growth-limit-pct` | `50` | PASS/FAIL threshold for RSS growth % |
| `--search-timeout-seconds` | `360` | Per-merge poll timeout |
| `--poll-interval` | `2` | Polling cadence when waiting for job done |
| `--probe-dir PATH` | (none) | Write probe files here for real searchability; see below |
| `--output-dir` | `benchmarks/results/loadtest` | JSON result directory (gitignored — results are local run output, not shipped) |
| `--dry-run` | off | Print config and exit |

### Searchability modes

**Lifecycle-only mode** (default, no `--probe-dir`): the harness fires
webhooks with synthetic file paths, then polls `GET /jobs/{job_id}` for
`status=done`.  This validates the full webhook→queue→worker pipeline but
cannot verify that content is actually retrievable via search.

**Content-search mode** (`--probe-dir PATH`): the harness writes a
temporary file containing a unique probe token into `--probe-dir` (which
must be a directory already indexed by the indexer) *before* firing the
webhook.  After the incremental job completes, `POST /search` is polled
for the probe token in result snippets.  This verifies end-to-end
retrieval.  Requires the indexer to have filesystem access to the directory.

### Metrics used

The harness scrapes `GET /metrics` at start and end of the run:

| Metric | Source | Used for |
|---|---|---|
| `treeloom_merge_to_searchable_seconds` | freshness histogram (§3) | Canonical latency (logged as mean) |
| `treeloom_queue_saturation` | backpressure gauge (§4) | Peak saturation reported in summary |
| `treeloom_webhook_shed_total` | backpressure counter (§4) | 429 shed count delta |
| `process_resident_memory_bytes` | prometheus_client default | RSS growth tracking |

If the indexer does not expose the histogram (an older build), the harness
falls back to in-process timing from webhook accept to job done.

### Output

- Live progress is printed to stdout (one line per merge).
- A summary table is printed at the end with PASS/FAIL per criterion.
- A JSON result file is written to `benchmarks/results/loadtest/<UTC>.json`
  for trend tracking.
- Exit code `0` = all criteria met; `1` = one or more breached; `2` = setup error.

### Sample result table

```
────────────────────────────────────────────────────────────
  TREELOOM INCREMENTAL LOAD TEST — SUMMARY
────────────────────────────────────────────────────────────
  Run ID  : c3d4e5f6-...
  URL     : http://localhost:8001
  Rate    : 10.0 merges/min for 300s

  Offered   :    50 merges
  Accepted  :    50  (100.0%)
  Shed(429) :     0
  Searchable:    50  (of accepted)

  Latency p50 :  12.34s
  Latency p95 :  28.91s  ✓ PASS (target 300s)
  Latency p99 :  45.23s

  Throughput offered  : 10.0 merges/min
  Throughput achieved : 10.0 merges/min
  Queue saturation pk :   0.02
  Webhook shed total  :    0.0

  RSS start  :  412.3 MB
  RSS end    :  438.7 MB
  RSS growth :   26.4 MB (6.4%)  ✓ PASS

────────────────────────────────────────────────────────────
  VERDICT: PASS
────────────────────────────────────────────────────────────
```

### CI integration

```yaml
# .github/workflows/loadtest.yml (illustrative)
- name: Load test incremental indexing
  run: |
    python scripts/loadtest_incremental.py \
      --url $INDEXER_URL \
      --token $TREELOOM_KEY \
      --repos 2 --rate 10 --duration 120 \
      --p95-target-seconds 120
  # Exits 1 if p95 or memory target breached → CI fails
```

## 9. Load testing the webhook path (test mode)

Driving sustained *synthetic* webhook traffic is hard against a real provider:
every webhook adapter resolves the changed-file list from the provider API
(`/pulls/{n}/files`), which is rate-limited and requires the PR to exist. Two
affordances make a self-contained load test possible:

- **`WEBHOOK_TRUST_INLINE_FILES=1`** (env, **default off — never enable in
  production**). When set, the GitHub adapter reads the changed-file list from
  the webhook payload (`pull_request.files`) instead of calling the GitHub API.
  The file list is only authenticated by the HMAC signature, so a signer could
  claim arbitrary paths — fine for a trusted load generator, unsafe in prod.
- **`scripts/loadtest_incremental.py --secret <WEBHOOK_GITHUB_SECRET>`** — the
  harness signs each payload with `X-Hub-Signature-256` (the indexer 503s
  without a secret configured and 401s on a bad signature).

Run against an **isolated** indexer (separate `DATABASE_URL` + `MILVUS_COLLECTION`),
never production:

```bash
WEBHOOK_GITHUB_SECRET=loadtest-secret WEBHOOK_TRUST_INLINE_FILES=1 \
DATABASE_URL=postgresql://.../treeloom_loadtest MILVUS_COLLECTION=treeloom_loadtest \
  uvicorn treeloom.indexer_service:app --port 8002
# index one URL-backed source, then:
python scripts/loadtest_incremental.py --url http://localhost:8002 \
  --secret loadtest-secret --repos 1 --rate 120 --duration 120 --p95-target-seconds 60
```

`/metrics` now exposes `process_resident_memory_bytes` (the metrics registry
registers `ProcessCollector`) so RSS growth is observable.

`loadtest_incremental.py` uses **synthetic** file paths, which is fine for
stressing the queue/backpressure/memory path but cannot measure real
merge→searchable (the paths don't exist in the clone, so `modified`/`added`
jobs fail → the latency histogram records *time-to-fail*). For a real
end-to-end number, use the harness below.

## 10. Real merge→searchable measurement

`scripts/loadtest_merge_searchable.py` drives the **whole real pipeline** —
real commit → signed webhook → queue → worker → clone → chunk → embed → Milvus
insert → `/search` finds a unique probe token — and measures the latency from
webhook accept to that token becoming searchable. It is self-contained:

- It creates N throwaway git repos, seeds them, and serves them over the
  **`git://` protocol** with `git daemon` (no auth, no network, no provider
  API or rate limits). `git://` is in the indexer's `GIT_ALLOW_PROTOCOL`
  allow-list, so the clone in the incremental job succeeds; a bare **local
  path is rejected** (`transport 'file' not allowed`), which is why the daemon
  is required.
- It round-robins merges across the N repos. This matters because the indexer
  enforces **one active job per `source_id`** — two overlapping merges on the
  *same* repo collapse into one job (the second is acked "in-flight job already
  covers this source"). N independent repos is what lets a sustained
  *concurrent* rate run without that collapse.
- The searchable poll runs **asynchronously** (a thread pool), so the generator
  sustains the offered rate instead of serializing on each merge's wait.

Together these remove the two limitations of the synthetic harness.

Two indexer settings are required for the run to measure anything:

- **`TREELOOM_GIT_ALLOW_LOOPBACK=true`** — the indexer refuses git remotes on
  the loopback interface by default (a clone of `127.0.0.1` looks like an SSRF
  probe), and every URL this harness serves is `git://localhost/<repo>`.
  Without the opt-in every clone is rejected and the run still reports
  timings. Set it on the **indexer** process.
- A **reachable** embedding backend. A fresh, isolated database seeds its
  `embedding_backends` registry from `.env`'s `EMBEDDING_URLS`; if those point
  at offline hosts, every `embed()` raises and every job finishes `done` with
  0 chunks. Override it inline:

```bash
DATABASE_URL=postgresql://.../treeloom_loadtest \
MILVUS_COLLECTION=treeloom_loadtest VECTOR_DIM=768 \
EMBEDDING_URLS=http://localhost:8082 \  # reachable backend, NOT the .env overflow URLs
GRAPH_STORE=sqlite GRAPH_DB_PATH=/tmp/treeloom_loadtest/graph.db \
WEBHOOK_GITHUB_SECRET=loadtest-secret WEBHOOK_TRUST_INLINE_FILES=1 AUTH_ENABLED=false \
TREELOOM_GIT_ALLOW_LOOPBACK=true \
  uvicorn treeloom.indexer_service:app --port 8002

python scripts/loadtest_merge_searchable.py --url http://localhost:8002 \
  --secret loadtest-secret --repos 8 --merges 200 --rate 120 --p95-target-seconds 60
```

**Documented result (2026-06-14, isolated indexer, embeddings from a local
TEI `jina-embeddings-v2-base-code` on `:8082`, Milvus, `BAAI/bge-reranker-v2-m3`
on `:8081`):** 200/200 merges searchable (0 collapsed, 0
shed), **p50 1.47s / p95 6.67s / p99 13.48s** (target 60s), sustained 112
merges/min, **RSS growth 3.4%** (395.6→408.9 MB). The indexer's own
`treeloom_merge_to_searchable_seconds` histogram recorded all 200 as terminal.
The p95 target was met with a wide margin. The result JSON was written to
`benchmarks/results/loadtest/<UTC>_merge_searchable.json` (local run output;
that directory is gitignored, so the file is not in the repository).
