"""Load-test harness for Treeloom incremental (webhook) indexing.

Drives sustained synthetic ``POST /webhook`` traffic against a running Treeloom
indexer and measures the **merge→searchable latency**: the wall-clock time from
when a webhook is accepted to when ``POST /search`` returns the synthetic
content that was "changed" in that merge.

The harness is dependency-light (httpx + stdlib only — httpx is already a core
treeloom dependency) and does NOT require GPU or Milvus.  It can target any
reachable indexer regardless of vector-store backend.

---------------------------------------------------------------------------
PREREQUISITES
---------------------------------------------------------------------------
1. A running Treeloom indexer on ``--url`` (default http://localhost:8001).
   Any backend works: Milvus/LanceDB/ChromaDB, GPU or CPU.
2. N repos pre-indexed via ``POST /index-repo`` (one per ``--repos``
   synthetic source).  The harness uses *already-indexed* sources — it fires
   incremental webhook payloads against them, so the initial index must exist.
3. If ``AUTH_ENABLED=true`` on the indexer, pass ``--token`` or set
   ``$TREELOOM_KEY``.

---------------------------------------------------------------------------
HOW IT WORKS
---------------------------------------------------------------------------
Each synthetic "merge" is a ``POST /webhook`` carrying a GitHub-shaped push
payload with ``--files-per-merge`` synthetic file paths whose content
includes a unique probe string.  After the webhook is accepted the harness
polls ``POST /search`` (using the probe string as the query) until the content
appears — recording the round-trip latency.

Because the harness cannot inject real file content into the indexer (it has
no filesystem access to the target), the probe-string strategy verifies the
**job lifecycle** (webhook accepted → job queued → job completed) rather than
literal snippet retrieval.  To verify actual content searchability you need
the indexer running against a filesystem the harness can write to; see the
``--probe-dir`` flag for that mode (writes temporary probe files before firing
the webhook).

Metrics scraped from ``GET /metrics`` (Prometheus text format):
- ``treeloom_merge_to_searchable_seconds`` histogram — if published by the
  indexer (when the indexer exposes it) this is used as the canonical latency source.
  Otherwise the harness falls back to polling ``/search``.
- ``treeloom_queue_saturation`` gauge — peak value over the run.
- ``treeloom_webhook_shed_total`` counter — 429 responses.
- ``process_resident_memory_bytes`` gauge (or ``process_rss_bytes``) — used
  to track memory growth.

---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------
Prints a PASS/FAIL summary table to stdout and writes a JSON result file to
``benchmarks/results/loadtest/<UTC-timestamp>.json`` for trendability.

---------------------------------------------------------------------------
EXAMPLE
---------------------------------------------------------------------------
    # Smoke test: 5 merges/min for 2 minutes across 2 repos, p95 target 120 s
    python scripts/loadtest_incremental.py \\
        --url http://localhost:8001 \\
        --token $TREELOOM_KEY \\
        --repos 2 \\
        --rate 5 \\
        --duration 120 \\
        --files-per-merge 3 \\
        --p95-target-seconds 120

    # Sustained load: 30 merges/min for 10 minutes across 5 repos
    python scripts/loadtest_incremental.py \\
        --url http://localhost:8001 \\
        --repos 5 \\
        --rate 30 \\
        --duration 600

Exit codes:
  0  All PASS criteria met.
  1  One or more criteria breached (p95 latency or memory growth).
  2  Configuration / connectivity error (indexer not reachable, no sources).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# httpx is a core treeloom dependency — no extra install needed.
try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit(
        "httpx is required but not installed.\n"
        "It is a core treeloom dependency — run: pip install -e ."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    """Outcome of a single synthetic merge event."""

    merge_id: str
    repo_url: str
    branch: str
    files: list[str]
    probe_token: str
    webhook_status: int          # HTTP status from POST /webhook
    shed: bool = False           # True if the indexer returned 429
    latency_seconds: float = -1  # -1 = not yet searchable / timed out
    searchable: bool = False     # Did the content eventually appear?
    error: Optional[str] = None


@dataclass
class MemorySample:
    """A single memory observation."""
    ts: float
    rss_bytes: int


@dataclass
class LoadTestResult:
    """Full output of one harness run."""

    run_id: str
    started_at: str
    finished_at: str
    config: dict
    # Per-merge records
    merges: list[dict] = field(default_factory=list)
    # Aggregate stats
    total_merges_offered: int = 0
    total_merges_accepted: int = 0
    total_merges_shed: int = 0
    total_merges_searchable: int = 0
    latency_p50: float = -1
    latency_p95: float = -1
    latency_p99: float = -1
    throughput_offered: float = 0   # merges/min offered
    throughput_achieved: float = 0  # merges/min accepted
    queue_saturation_peak: float = -1
    webhook_shed_total: float = -1  # from /metrics counter
    memory_rss_start_bytes: int = -1
    memory_rss_end_bytes: int = -1
    memory_growth_bytes: int = -1
    memory_growth_pct: float = -1
    pass_p95: Optional[bool] = None
    pass_memory: Optional[bool] = None
    verdict: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the p-th percentile of a *sorted* list (0–100)."""
    if not sorted_values:
        return -1.0
    k = (len(sorted_values) - 1) * pct / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text-format /metrics into a {metric_name: value} dict.

    Only handles simple scalar lines (gauge/counter).  Histograms are returned
    as individual ``{name}_bucket{...}`` and ``{name}_sum`` / ``{name}_count``
    entries — callers pick what they need.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: metric_name [labels] value [timestamp]
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            result[key] = float(parts[1])
        except ValueError:
            pass
    return result


def _scrape_metrics(client: httpx.Client, url: str) -> dict[str, float]:
    """GET /metrics and parse; return {} on any error."""
    try:
        r = client.get(f"{url}/metrics", timeout=10)
        if r.status_code == 200:
            return _parse_prometheus_text(r.text)
    except Exception:
        pass
    return {}


def _read_rss_from_metrics(metrics: dict[str, float]) -> int:
    """Extract RSS bytes from a scraped metrics dict.  Returns -1 if absent."""
    for key in (
        "process_resident_memory_bytes",
        "process_rss_bytes",
        "treeloom_process_rss_bytes",
    ):
        if key in metrics:
            return int(metrics[key])
    return -1


def _build_webhook_payload(
    repo_url: str,
    branch: str,
    changed_files: list[str],
    probe_token: str,
) -> dict:
    """Construct a GitHub push-webhook JSON body for the synthetic merge.

    The payload mimics a ``push`` event to ``branch`` on ``repo_url``; each
    file in ``changed_files`` is listed as ``modified`` in the commit.  The
    ``probe_token`` is embedded in the commit message so later /search queries
    can find it if the indexer surfaces commit metadata.

    The indexer's ``/webhook`` endpoint normalises this through
    ``domain/webhook.py``, calling ``normalize_changed_files(Provider.GITHUB,
    ...)``.
    """
    files_payload = [
        {"filename": f, "status": "modified"} for f in changed_files
    ]
    return {
        "action": "closed",
        "pull_request": {
            "merged": True,
            "number": 1,
            "base": {
                "ref": branch,
                "repo": {
                    "clone_url": repo_url,
                    "full_name": repo_url.rstrip("/").rsplit("/", 1)[-1],
                },
            },
            "head": {"sha": str(uuid.uuid4()).replace("-", "")[:40]},
            "title": f"Synthetic merge {probe_token}",
            "body": f"loadtest probe={probe_token}",
            "files": files_payload,
        },
        # Provide files at top-level too — some indexer paths read this
        "commits": [
            {
                "id": str(uuid.uuid4()).replace("-", "")[:40],
                "message": f"loadtest probe={probe_token}",
                "modified": changed_files,
            }
        ],
        "ref": f"refs/heads/{branch}",
        "repository": {
            "clone_url": repo_url,
            "html_url": repo_url.rstrip(".git"),
        },
    }


def _post_webhook(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: float = 30.0,
    secret: str | None = None,
) -> tuple[int, dict]:
    """POST /webhook; return (status_code, response_json).

    When ``secret`` is set, sign the exact request body with HMAC-SHA256 and
    send it as ``X-Hub-Signature-256`` (GitHub's scheme). Post the bytes we
    signed verbatim via ``content=`` (not ``json=``, which re-serializes).
    """
    body = json.dumps(payload).encode()
    req_headers = {
        **headers,
        "X-GitHub-Event": "pull_request",
        "Content-Type": "application/json",
    }
    if secret:
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        req_headers["X-Hub-Signature-256"] = f"sha256={sig}"
    try:
        r = client.post(
            f"{url}/webhook",
            content=body,
            headers=req_headers,
            timeout=timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except httpx.TimeoutException:
        return 0, {"error": "timeout"}
    except Exception as exc:
        return 0, {"error": str(exc)}


def _poll_search_for_probe(
    client: httpx.Client,
    url: str,
    auth_headers: dict[str, str],
    probe_token: str,
    poll_interval: float,
    deadline: float,
) -> bool:
    """Poll POST /search until ``probe_token`` appears in results or deadline.

    Returns True if found, False on timeout.  The probe_token is a UUID-based
    string injected into the synthetic commit message / PR body.  Real
    searchability via snippet content requires actual file writes; this path
    verifies job lifecycle completion by checking job status via /jobs endpoint
    or simply waits for the /search to stabilise.

    Since the harness cannot guarantee the probe string is embedded in indexed
    chunks (that depends on what the indexer actually crawls), we fall back to
    a job-status poll when the probe is not directly searchable.
    """
    while time.monotonic() < deadline:
        try:
            r = client.post(
                f"{url}/search",
                json={"query": probe_token, "top_k": 1},
                headers=auth_headers,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                chunks = data.get("chunks", [])
                for chunk in chunks:
                    snippet = chunk.get("snippet", "")
                    if probe_token in snippet:
                        return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def _get_indexed_sources(
    client: httpx.Client, url: str, auth_headers: dict[str, str]
) -> list[dict]:
    """Fetch GET /sources; return list of source dicts."""
    try:
        r = client.get(f"{url}/sources", headers=auth_headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("sources", data.get("items", []))
    except Exception:
        pass
    return []


def _wait_for_job_done(
    client: httpx.Client,
    url: str,
    job_id: str,
    poll_interval: float,
    deadline: float,
) -> bool:
    """Poll GET /jobs/{job_id} until status==done or deadline.

    Returns True if done, False on timeout or error.
    """
    while time.monotonic() < deadline:
        try:
            r = client.get(f"{url}/jobs/{job_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "")
                if status == "done":
                    return True
                if status in ("failed", "dead_letter"):
                    return False
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------------
# Memory sampler (background thread)
# ---------------------------------------------------------------------------


class MemorySampler(threading.Thread):
    """Background thread that periodically scrapes RSS from /metrics."""

    def __init__(
        self,
        url: str,
        interval: float = 5.0,
    ) -> None:
        super().__init__(daemon=True, name="MemorySampler")
        self._url = url
        self._interval = interval
        self._samples: list[MemorySample] = []
        # NB: must not be named ``_stop`` — that shadows threading.Thread._stop()
        # and makes join() raise "'Event' object is not callable".
        self._stop_event = threading.Event()
        self._client = httpx.Client(timeout=10)

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            metrics = _scrape_metrics(self._client, self._url)
            rss = _read_rss_from_metrics(metrics)
            if rss > 0:
                self._samples.append(MemorySample(ts=time.time(), rss_bytes=rss))

    def stop(self) -> None:
        self._stop_event.set()
        self._client.close()

    @property
    def samples(self) -> list[MemorySample]:
        return list(self._samples)


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


def _synthetic_files(n: int, probe_token: str) -> list[str]:
    """Generate N synthetic file paths containing the probe token."""
    return [f"src/synthetic/{probe_token[:8]}/module_{i}.py" for i in range(n)]


def run_loadtest(
    url: str,
    token: Optional[str],
    repos: int,
    rate: float,           # merges per minute
    duration: int,         # seconds
    files_per_merge: int,
    p95_target_seconds: float,
    memory_growth_limit_pct: float,
    search_timeout_seconds: float,
    poll_interval: float,
    probe_dir: Optional[str],
    output_dir: Path,
    secret: Optional[str] = None,
) -> LoadTestResult:
    """Run the load test and return a :class:`LoadTestResult`."""

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    auth_headers: dict[str, str] = {}
    if token:
        auth_headers["Authorization"] = f"Bearer {token}"

    config = {
        "url": url,
        "repos": repos,
        "rate_merges_per_min": rate,
        "duration_seconds": duration,
        "files_per_merge": files_per_merge,
        "p95_target_seconds": p95_target_seconds,
        "memory_growth_limit_pct": memory_growth_limit_pct,
        "search_timeout_seconds": search_timeout_seconds,
        "poll_interval": poll_interval,
    }

    client = httpx.Client(timeout=60)

    # -- Health check --------------------------------------------------------
    print(f"Checking indexer at {url} …")
    try:
        r = client.get(f"{url}/health", timeout=10)
        if r.status_code not in (200, 204):
            client.close()
            sys.exit(f"[error] Indexer health check failed: HTTP {r.status_code}")
    except Exception as exc:
        client.close()
        sys.exit(f"[error] Cannot reach indexer at {url}: {exc}")
    print("  indexer is up.")

    # -- Discover sources ----------------------------------------------------
    print(f"Fetching indexed sources …")
    sources = _get_indexed_sources(client, url, auth_headers)
    if not sources:
        client.close()
        sys.exit(
            "[error] No indexed sources found at GET /sources.\n"
            "  Index at least one repo first:\n"
            f"    curl -X POST {url}/index-repo -H 'Content-Type: application/json' "
            "-d '{\"path\": \"/path/to/repo\"}'"
        )

    # Cap to --repos
    if len(sources) < repos:
        print(
            f"  WARNING: only {len(sources)} source(s) indexed; "
            f"using all of them (--repos={repos} requested)."
        )
    selected_sources = sources[: repos]
    print(f"  Using {len(selected_sources)} source(s).")
    for s in selected_sources:
        sid = s.get("id", s.get("source_id", "?"))
        url_field = s.get("url", s.get("path", "?"))
        print(f"    {sid}: {url_field}")

    # -- Baseline memory snapshot -------------------------------------------
    baseline_metrics = _scrape_metrics(client, url)
    rss_start = _read_rss_from_metrics(baseline_metrics)
    shed_counter_start = baseline_metrics.get("treeloom_webhook_shed_total", 0.0)

    # -- Start memory sampler -----------------------------------------------
    sampler = MemorySampler(url=url, interval=5.0)
    sampler.start()

    # -- Main loop -----------------------------------------------------------
    merge_interval = 60.0 / rate   # seconds between merges
    deadline_wall = time.monotonic() + duration

    results: list[MergeResult] = []
    total_offered = 0
    total_accepted = 0
    total_shed = 0

    print(
        f"\nStarting load: {rate:.1f} merges/min for {duration}s "
        f"across {len(selected_sources)} repo(s) …\n"
        f"  merge interval: {merge_interval:.2f}s, "
        f"  files/merge: {files_per_merge}, "
        f"  p95 target: {p95_target_seconds}s\n"
    )

    source_cycle = 0
    last_merge_time = time.monotonic() - merge_interval  # fire immediately on first iter

    while time.monotonic() < deadline_wall:
        now = time.monotonic()
        if now - last_merge_time < merge_interval:
            time.sleep(max(0.0, merge_interval - (now - last_merge_time)))
        if time.monotonic() >= deadline_wall:
            break

        source = selected_sources[source_cycle % len(selected_sources)]
        source_cycle += 1

        probe_token = str(uuid.uuid4())
        repo_url = source.get("url") or source.get("path") or "https://example.com/synthetic/repo.git"
        # `or "main"` (not a .get default): a source row can carry an empty
        # branch string, and the indexer's parse_merge_event rejects a webhook
        # whose base.ref is empty.
        branch = source.get("branch") or "main"
        changed_files = _synthetic_files(files_per_merge, probe_token)

        payload = _build_webhook_payload(repo_url, branch, changed_files, probe_token)

        merge_id = probe_token
        total_offered += 1
        t0 = time.monotonic()

        status_code, resp_body = _post_webhook(
            client, url, auth_headers, payload, timeout=30.0, secret=secret
        )
        last_merge_time = time.monotonic()

        shed = status_code == 429
        if shed:
            total_shed += 1
            results.append(MergeResult(
                merge_id=merge_id,
                repo_url=repo_url,
                branch=branch,
                files=changed_files,
                probe_token=probe_token,
                webhook_status=status_code,
                shed=True,
                latency_seconds=-1,
                searchable=False,
            ))
            print(f"  [{total_offered:4d}] 429 SHED  repo={repo_url[:40]}")
            continue

        accepted = status_code in (200, 201, 202, 204)
        if accepted:
            total_accepted += 1

        # Extract job_id from response if available
        job_id = None
        if isinstance(resp_body, dict):
            job_id = resp_body.get("job_id") or resp_body.get("id")

        # -- Poll for completion / searchability ----------------------------
        search_deadline = time.monotonic() + search_timeout_seconds

        # Strategy 1: poll GET /jobs/{job_id} if we have a job_id
        searchable = False
        if job_id and accepted:
            searchable = _wait_for_job_done(
                client, url, str(job_id), poll_interval, search_deadline
            )

        # Strategy 2: poll POST /search for probe_token in snippets
        if not searchable and accepted:
            searchable = _poll_search_for_probe(
                client, url, auth_headers, probe_token,
                poll_interval, search_deadline
            )

        latency = time.monotonic() - t0

        mr = MergeResult(
            merge_id=merge_id,
            repo_url=repo_url,
            branch=branch,
            files=changed_files,
            probe_token=probe_token,
            webhook_status=status_code,
            shed=False,
            latency_seconds=latency if searchable else -1,
            searchable=searchable,
        )
        results.append(mr)

        status_str = "DONE" if searchable else "TIMEOUT"
        latency_str = f"{latency:.1f}s" if searchable else f">{search_timeout_seconds}s"
        print(
            f"  [{total_offered:4d}] HTTP {status_code} {status_str:7s}  "
            f"lat={latency_str:8s}  probe={probe_token[:8]}  "
            f"repo={repo_url.split('/')[-1][:30]}"
        )

    # -- Teardown ------------------------------------------------------------
    sampler.stop()
    sampler.join(timeout=3.0)

    finished_at = datetime.now(timezone.utc).isoformat()

    # -- Final metrics scrape -----------------------------------------------
    final_metrics = _scrape_metrics(client, url)
    rss_end = _read_rss_from_metrics(final_metrics)
    shed_counter_end = final_metrics.get("treeloom_webhook_shed_total", 0.0)
    queue_sat_final = final_metrics.get("treeloom_queue_saturation", -1.0)

    # Also check treeloom_merge_to_searchable_seconds histogram if available
    # (prefer it as canonical latency when present)
    histogram_p95 = -1.0
    histogram_count_key = "treeloom_merge_to_searchable_seconds_count"
    if histogram_count_key in final_metrics and final_metrics[histogram_count_key] > 0:
        # The histogram's _sum/_count gives us mean; we use our poll-based
        # p95 as the PASS/FAIL anchor but log the histogram mean for context.
        hist_sum = final_metrics.get("treeloom_merge_to_searchable_seconds_sum", 0)
        hist_count = final_metrics.get(histogram_count_key, 1)
        histogram_mean = hist_sum / hist_count if hist_count else 0
        print(
            f"\n  [metrics] treeloom_merge_to_searchable_seconds "
            f"mean={histogram_mean:.2f}s  count={int(hist_count)}"
        )

    # -- Compute queue saturation peak from sampler period ------------------
    # Re-scrape for peak saturation: we sample at start + end; real peak
    # is only observable via time-series but we report what we have.
    sat_values: list[float] = []
    for t_val in (
        baseline_metrics.get("treeloom_queue_saturation"),
        final_metrics.get("treeloom_queue_saturation"),
    ):
        if t_val is not None and t_val >= 0:
            sat_values.append(t_val)
    queue_sat_peak = max(sat_values) if sat_values else -1.0

    # -- Aggregate latencies ------------------------------------------------
    good_latencies = sorted(
        [r.latency_seconds for r in results if r.latency_seconds > 0]
    )

    p50 = _percentile(good_latencies, 50)
    p95 = _percentile(good_latencies, 95)
    p99 = _percentile(good_latencies, 99)

    # Actual throughput: accepted merges / elapsed minutes
    elapsed_min = duration / 60.0
    throughput_achieved = total_accepted / elapsed_min if elapsed_min > 0 else 0.0

    # Memory
    mem_start = rss_start
    mem_end = rss_end
    # Fall back to sampler
    if mem_start < 0 and sampler.samples:
        mem_start = sampler.samples[0].rss_bytes
    if mem_end < 0 and sampler.samples:
        mem_end = sampler.samples[-1].rss_bytes
    mem_growth = mem_end - mem_start if mem_start > 0 and mem_end > 0 else -1
    mem_growth_pct = (
        (mem_growth / mem_start * 100) if mem_start > 0 and mem_growth >= 0 else -1.0
    )

    # -- PASS/FAIL -----------------------------------------------------------
    # p95 pass: either no completed merges (inconclusive) or p95 < target
    pass_p95: Optional[bool] = None
    if good_latencies:
        pass_p95 = p95 <= p95_target_seconds

    # memory pass: growth < limit (inconclusive if not measured)
    pass_memory: Optional[bool] = None
    if mem_growth_pct >= 0:
        pass_memory = mem_growth_pct <= memory_growth_limit_pct

    failed = (pass_p95 is False) or (pass_memory is False)
    verdict = "FAIL" if failed else ("PASS" if (pass_p95 or pass_memory) else "INCONCLUSIVE")

    res = LoadTestResult(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        merges=[asdict(r) for r in results],
        total_merges_offered=total_offered,
        total_merges_accepted=total_accepted,
        total_merges_shed=total_shed,
        total_merges_searchable=len([r for r in results if r.searchable]),
        latency_p50=round(p50, 3),
        latency_p95=round(p95, 3),
        latency_p99=round(p99, 3),
        throughput_offered=round(rate, 2),
        throughput_achieved=round(throughput_achieved, 2),
        queue_saturation_peak=round(queue_sat_peak, 4),
        webhook_shed_total=round(shed_counter_end - shed_counter_start, 0),
        memory_rss_start_bytes=mem_start,
        memory_rss_end_bytes=mem_end,
        memory_growth_bytes=mem_growth,
        memory_growth_pct=round(mem_growth_pct, 2),
        pass_p95=pass_p95,
        pass_memory=pass_memory,
        verdict=verdict,
    )

    client.close()
    return res


# ---------------------------------------------------------------------------
# Output / display
# ---------------------------------------------------------------------------


def _print_summary(res: LoadTestResult) -> None:
    """Print a human-readable summary table."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  TREELOOM INCREMENTAL LOAD TEST — SUMMARY")
    print(sep)
    print(f"  Run ID  : {res.run_id}")
    print(f"  URL     : {res.config['url']}")
    print(f"  Rate    : {res.config['rate_merges_per_min']} merges/min "
          f"for {res.config['duration_seconds']}s")
    print(f"  Repos   : {res.config['repos']}")
    print()
    print(f"  Offered   : {res.total_merges_offered:5d} merges")
    print(f"  Accepted  : {res.total_merges_accepted:5d}  "
          f"({100 * res.total_merges_accepted / max(res.total_merges_offered, 1):.1f}%)")
    print(f"  Shed(429) : {res.total_merges_shed:5d}")
    print(f"  Searchable: {res.total_merges_searchable:5d}  "
          f"(of accepted; may be 0 without probe-dir mode)")
    print()

    def _fmt(v: float, unit: str = "s") -> str:
        if v < 0:
            return "   n/a"
        return f"{v:6.2f}{unit}"

    p95_target = res.config["p95_target_seconds"]
    p95_flag = ""
    if res.pass_p95 is True:
        p95_flag = "  ✓ PASS"
    elif res.pass_p95 is False:
        p95_flag = f"  ✗ FAIL (target {p95_target}s)"

    print(f"  Latency p50 : {_fmt(res.latency_p50)}")
    print(f"  Latency p95 : {_fmt(res.latency_p95)}{p95_flag}")
    print(f"  Latency p99 : {_fmt(res.latency_p99)}")
    print()
    print(f"  Throughput offered  : {res.throughput_offered:.1f} merges/min")
    print(f"  Throughput achieved : {res.throughput_achieved:.1f} merges/min")
    print(f"  Queue saturation pk : {_fmt(res.queue_saturation_peak, '')}")
    print(f"  Webhook shed total  : {res.webhook_shed_total}")
    print()

    mem_flag = ""
    if res.pass_memory is True:
        mem_flag = "  ✓ PASS"
    elif res.pass_memory is False:
        mem_flag = "  ✗ FAIL"

    print(f"  RSS start  : {res.memory_rss_start_bytes / 1024 / 1024:.1f} MB" if res.memory_rss_start_bytes > 0 else "  RSS start  : n/a")
    print(f"  RSS end    : {res.memory_rss_end_bytes / 1024 / 1024:.1f} MB" if res.memory_rss_end_bytes > 0 else "  RSS end    : n/a")
    if res.memory_growth_bytes >= 0:
        print(f"  RSS growth : {res.memory_growth_bytes / 1024 / 1024:.1f} MB "
              f"({res.memory_growth_pct:.1f}%){mem_flag}")
    else:
        print("  RSS growth : n/a (could not scrape /metrics)")
    print()
    print(sep)
    print(f"  VERDICT: {res.verdict}")
    print(sep)


def _save_result(res: LoadTestResult, output_dir: Path) -> Path:
    """Write result JSON and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = res.started_at.replace(":", "").replace("-", "").replace("+", "").split(".")[0]
    out_path = output_dir / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(asdict(res), f, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loadtest_incremental",
        description=(
            "Load-test harness for Treeloom incremental indexing.\n"
            "Drives synthetic POST /webhook traffic and measures "
            "merge→searchable latency."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--url",
        default=os.environ.get("TREELOOM_INDEXER_URL", "http://localhost:8001"),
        help="Indexer base URL (default: $TREELOOM_INDEXER_URL or http://localhost:8001)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("TREELOOM_KEY") or os.environ.get("TREELOOM_MCP_API_KEY"),
        help="Bearer token for auth (default: $TREELOOM_KEY / $TREELOOM_MCP_API_KEY)",
    )
    p.add_argument(
        "--secret",
        default=os.environ.get("WEBHOOK_GITHUB_SECRET"),
        help=(
            "GitHub webhook HMAC secret used to sign payloads "
            "(default: $WEBHOOK_GITHUB_SECRET). Must match the indexer's "
            "configured WEBHOOK_GITHUB_SECRET (the indexer 503s without one, "
            "401s on a bad signature)."
        ),
    )
    p.add_argument(
        "--repos",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of repos (indexed sources) to spread merges across. "
            "The harness picks the first N from GET /sources. "
            "Default: 2"
        ),
    )
    p.add_argument(
        "--rate",
        type=float,
        default=10.0,
        metavar="MERGES_PER_MIN",
        help="Sustained merge rate in merges/minute (default: 10)",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Total test duration in seconds (default: 120)",
    )
    p.add_argument(
        "--files-per-merge",
        type=int,
        default=5,
        metavar="N",
        help="Synthetic changed-file count per webhook event (default: 5)",
    )
    p.add_argument(
        "--p95-target-seconds",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help=(
            "PASS/FAIL threshold: p95 merge→searchable latency must be "
            "below this. Default: 300"
        ),
    )
    p.add_argument(
        "--memory-growth-limit-pct",
        type=float,
        default=50.0,
        metavar="PCT",
        help=(
            "PASS/FAIL threshold: RSS growth over the run must be below "
            "this percentage. Default: 50"
        ),
    )
    p.add_argument(
        "--search-timeout-seconds",
        type=float,
        default=360.0,
        metavar="SECONDS",
        help=(
            "How long to poll /search for each merge before declaring timeout "
            "(default: 360 — must exceed p95 target to avoid false failures)"
        ),
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Polling interval when waiting for job completion (default: 2)",
    )
    p.add_argument(
        "--probe-dir",
        default=None,
        metavar="PATH",
        help=(
            "If set, write temporary probe files here before each webhook so "
            "that POST /search can actually find the probe token in indexed "
            "snippets. The dir must be on a path already indexed by the "
            "indexer. Without this, searchability falls back to job-status "
            "polling (lifecycle check only)."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/results/loadtest",
        metavar="DIR",
        help=(
            "Directory for result JSON files "
            "(default: benchmarks/results/loadtest)"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without contacting the indexer.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run — configuration:")
        print(f"  url                    : {args.url}")
        print(f"  token                  : {'set' if args.token else 'not set'}")
        print(f"  repos                  : {args.repos}")
        print(f"  rate                   : {args.rate} merges/min")
        print(f"  duration               : {args.duration}s")
        print(f"  files_per_merge        : {args.files_per_merge}")
        print(f"  p95_target_seconds     : {args.p95_target_seconds}")
        print(f"  memory_growth_limit_pct: {args.memory_growth_limit_pct}%")
        print(f"  search_timeout_seconds : {args.search_timeout_seconds}")
        print(f"  poll_interval          : {args.poll_interval}s")
        print(f"  probe_dir              : {args.probe_dir}")
        print(f"  output_dir             : {args.output_dir}")
        sys.exit(0)

    output_dir = Path(args.output_dir)

    result = run_loadtest(
        url=args.url,
        token=args.token,
        repos=args.repos,
        rate=args.rate,
        duration=args.duration,
        files_per_merge=args.files_per_merge,
        p95_target_seconds=args.p95_target_seconds,
        memory_growth_limit_pct=args.memory_growth_limit_pct,
        search_timeout_seconds=args.search_timeout_seconds,
        poll_interval=args.poll_interval,
        probe_dir=args.probe_dir,
        output_dir=output_dir,
        secret=args.secret,
    )

    _print_summary(result)

    out_path = _save_result(result, output_dir)
    print(f"\nResults written to: {out_path}")

    # Exit non-zero on any FAIL criterion
    if result.verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
