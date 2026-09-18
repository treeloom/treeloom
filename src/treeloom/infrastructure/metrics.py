"""Prometheus metrics for Treeloom indexer.

Metric naming is stable — the Grafana dashboards (assets/grafana/) and the load-test
harnesses (scripts/loadtest_*.py) code against the exact names defined here. Do not rename without
updating those.
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from prometheus_client import ProcessCollector

registry = CollectorRegistry()

# Expose process-level metrics (process_resident_memory_bytes, process_cpu_*,
# open fds, …) on our custom registry. The default registry isn't scraped by
# GET /metrics, so without this the load-test harness and any RSS/CPU
# dashboard can't observe indexer memory growth.
try:
    ProcessCollector(registry=registry)
except Exception:  # pragma: no cover - platform without /proc
    pass

# Counters
files_processed = Counter(
    "treeloom_files_processed_total",
    "Total files indexed",
    registry=registry,
)
chunks_embedded = Counter(
    "treeloom_chunks_embedded_total",
    "Total chunks embedded and inserted into Milvus",
    registry=registry,
)
embed_errors = Counter(
    "treeloom_embed_errors_total",
    "Total embedding failures",
    registry=registry,
)
job_creates = Counter(
    "treeloom_jobs_created_total",
    "Total indexing jobs submitted",
    ["status"],
    registry=registry,
)
rerank_fallbacks = Counter(
    "treeloom_rerank_fallbacks_total",
    "Searches that degraded to raw vector order because the reranker call "
    "failed (dead RERANKER_URL, timeout). Any nonzero rate means ranking "
    "quality is silently degraded — alert on it.",
    registry=registry,
)
encoding_fallbacks = Counter(
    "treeloom_encoding_fallbacks_total",
    "Files indexed via UTF-8 replacement fallback (invalid UTF-8 input)",
    registry=registry,
)

# ── Incremental-job counters ────────────────────────────────────────

incremental_jobs_total = Counter(
    "treeloom_incremental_jobs_total",
    "Incremental (webhook) indexing jobs by terminal status",
    ["status"],  # done | failed | dead_letter
    registry=registry,
)

dead_letter_jobs_total = Counter(
    "treeloom_dead_letter_jobs_total",
    "Jobs that exhausted all retry attempts and entered dead-letter state",
    registry=registry,
)

webhook_shed_total = Counter(
    "treeloom_webhook_shed_total",
    "Webhook requests rejected with 429 due to queue saturation",
    registry=registry,
)

fleet_refresh_enqueued_total = Counter(
    "treeloom_fleet_refresh_enqueued_total",
    "Re-index jobs enqueued by the staleness-driven fleet auto-refresh loop",
    registry=registry,
)

# Histograms
embed_latency = Histogram(
    "treeloom_embed_latency_seconds",
    "Embed server request latency",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=registry,
)
chunk_size = Histogram(
    "treeloom_chunk_size_bytes",
    "Chunk text size in bytes before embedding",
    buckets=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536),
    registry=registry,
)

# ── Incremental-job histograms ─────────────────────────────────────

incremental_job_duration_seconds = Histogram(
    "treeloom_incremental_job_duration_seconds",
    "Wall-clock time for a completed incremental (webhook) index job",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
    registry=registry,
)

merge_to_searchable_seconds = Histogram(
    "treeloom_merge_to_searchable_seconds",
    "Time from webhook acceptance to incremental job completion (merge→searchable). "
    "git ls-remote cannot resolve 'commits behind HEAD', so we use index-age "
    "gauges and this histogram for SLO alerting instead.",
    buckets=(5, 15, 30, 60, 120, 300, 600, 1800, 3600),
    registry=registry,
)

# Gauges
active_jobs = Gauge(
    "treeloom_active_jobs",
    "Currently running indexing jobs",
    registry=registry,
)
queue_depth = Gauge(
    "treeloom_queue_depth",
    "Jobs queued awaiting execution",
    registry=registry,
)

# ── Freshness / staleness gauges ───────────────────────────────────

source_index_stale = Gauge(
    "treeloom_source_index_stale",
    "1 if the source's indexed commit SHA differs from current HEAD, else 0. "
    "Only exported for up to FRESHNESS_MAX_SOURCES sources; above that, "
    "see treeloom_sources_stale_total instead. "
    "git ls-remote resolves HEAD SHA, not commit timestamp.",
    ["source_id"],
    registry=registry,
)

source_index_age_seconds = Gauge(
    "treeloom_source_index_age_seconds",
    "Seconds since the source was last indexed (now - indexed_at). "
    "Exported per-source up to FRESHNESS_MAX_SOURCES; use as SLO signal "
    "when combined with treeloom_source_index_stale.",
    ["source_id"],
    registry=registry,
)

sources_stale_total = Gauge(
    "treeloom_sources_stale_total",
    "Aggregate count of stale sources when cardinality exceeds "
    "FRESHNESS_MAX_SOURCES (replaces per-label gauges above that threshold).",
    registry=registry,
)

# ── Queue saturation gauge ─────────────────────────────────────────

queue_saturation = Gauge(
    "treeloom_queue_saturation",
    "Queue fill ratio: depth / MAX_QUEUE_DEPTH, capped at 1.0. "
    "Alert when this approaches 1 — new webhooks will be shed with 429.",
    registry=registry,
)


def get_metrics() -> bytes:
    """Generate Prometheus text format metrics."""
    return generate_latest(registry)
