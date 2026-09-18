#!/usr/bin/env python3
"""End-to-end merge→searchable load test.

Unlike ``scripts/loadtest_incremental.py`` (synthetic file paths against a
public repo — the changed files never exist in the clone, so every
``modified``/``added`` job fails → dead-letters, and "merge→searchable" is
really "time-to-fail"), this harness drives **real commits** so the whole
pipeline actually runs:

    real commit on a served repo
      → signed webhook (GitHub merged-PR shape, inline files)
      → /webhook 202 → queue → worker
      → clone (git:// — allowed by the indexer's GIT_ALLOW_PROTOCOL)
      → chunk → embed → Milvus insert
      → /search finds the probe token  ← merge→searchable measured here

It is fully self-contained: it creates N throwaway git repos, serves them over
the ``git://`` protocol with ``git daemon`` (no auth, no network, no provider
API / rate limits), indexes each into the target indexer, then drives merges.

Why N repos: the indexer enforces **one active job per source_id**, so two
overlapping merges on the *same* repo collapse into one job (the second is
acked as "in-flight job already covers this source" and its file never gets
its own job). Round-robining merges across N independent repos is what lets a
sustained *concurrent* rate run without that collapse — addressing the second
limitation called out in docs/incremental-indexing.md §"Known limitations".

The searchable poll fires **asynchronously** (a thread pool) so the generator
sustains the offered rate instead of serializing on each merge's wait —
addressing the first limitation.

Prerequisites: an **isolated** indexer (separate DATABASE_URL +
MILVUS_COLLECTION, reachable EMBEDDING_URLS) with
``WEBHOOK_TRUST_INLINE_FILES=1``, ``WEBHOOK_GITHUB_SECRET`` and
``TREELOOM_GIT_ALLOW_LOOPBACK=true`` set. See docs/incremental-indexing.md §9.

That last one is not boilerplate: the indexer refuses git remotes on the
loopback interface by default, because a request to clone 127.0.0.1 is the
shape of an SSRF probe. Every URL this harness serves is
``git://localhost/<repo>``, so without the opt-in every clone is rejected
and the run measures nothing while still reporting timings. The harness
checks for it up front rather than letting that happen — set it on the
**indexer** process, not on this script.

Example::

    python scripts/loadtest_merge_searchable.py \\
        --url http://localhost:8002 --secret loadtest-secret \\
        --base-path /tmp/treeloom_loadtest --repos 6 \\
        --merges 60 --rate 30 --p95-target-seconds 60

Exit code 0 = criteria met, 1 = a target breached, 2 = setup error.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

GIT_DAEMON_PORT = 9418


# ── repo fixtures ───────────────────────────────────────────────────────────

_SEED_FILES = {
    "src/auth.py": '''\
"""Authentication helpers for the loadtest fixture repo."""


def hash_password(raw: str, salt: str) -> str:
    """Return a salted hash for the given password."""
    import hashlib
    return hashlib.sha256((salt + raw).encode()).hexdigest()


class SessionManager:
    """Tracks active user sessions in memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}

    def create(self, user_id: str, token: str) -> None:
        self._sessions[token] = user_id
''',
    "src/cache.py": '''\
"""A tiny LRU cache used by the loadtest fixture."""
from collections import OrderedDict


class LRUCache:
    """Least-recently-used cache with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]
''',
}


def _probe_source(token: str) -> str:
    """Source for one merge's probe file.

    The token appears as an identifier, a docstring, and a return literal so
    it lands in ``chunk_text`` and is matched by hybrid search's sparse BM25
    leg (exact-token match) regardless of dense ranking.
    """
    return (
        '"""Loadtest probe module — merge->searchable marker."""\n\n\n'
        f"def probe_{token}() -> str:\n"
        f'    """Searchable loadtest probe token {token}."""\n'
        f'    return "{token}"\n'
    )


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def create_repo(base: Path, name: str) -> Path:
    repo = base / name
    if repo.exists():
        shutil.rmtree(repo)
    (repo / "src").mkdir(parents=True)
    for rel, content in _SEED_FILES.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (repo / "loadtest").mkdir(exist_ok=True)
    (repo / "README.md").write_text(
        f"# {name} — throwaway treeloom merge->searchable fixture\n"
    )
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "loadtest@treeloom.local")
    _git(repo, "config", "user.name", "Loadtest Bot")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed loadtest fixture repo")
    return repo


def start_git_daemon(base: Path) -> subprocess.Popen:
    """Serve every repo under *base* read-only over git:// (port 9418)."""
    proc = subprocess.Popen(
        [
            "git", "daemon", "--reuseaddr", "--export-all",
            f"--base-path={base}", "--enable=upload-pack", str(base),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for the port to accept a clone.
    deadline = time.time() + 10
    test_url = None
    while time.time() < deadline:
        for child in base.iterdir():
            if (child / ".git").exists() or (child / "HEAD").exists():
                test_url = child.name
                break
        if test_url:
            dst = base / "_daemon_probe"
            if dst.exists():
                shutil.rmtree(dst)
            r = subprocess.run(
                ["git", "clone", "-q", f"git://localhost/{test_url}", str(dst)],
                capture_output=True, text=True,
            )
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            if r.returncode == 0:
                return proc
        time.sleep(0.3)
    proc.terminate()
    raise RuntimeError("git daemon did not become ready")


# ── webhook payload + signing ───────────────────────────────────────────────

def build_payload(repo_url: str, branch: str, rel_path: str, token: str) -> dict:
    """GitHub merged-PR webhook body with the changed file inline (trust mode)."""
    return {
        "action": "closed",
        "pull_request": {
            "merged": True,
            "number": 1,
            "base": {"ref": branch, "repo": {"clone_url": repo_url}},
            "head": {"sha": uuid.uuid4().hex[:40]},
            "title": f"probe {token}",
            "files": [{"filename": rel_path, "status": "added"}],
        },
        "ref": f"refs/heads/{branch}",
        "repository": {"clone_url": repo_url},
    }


def post_webhook(
    client: httpx.Client, url: str, secret: str, payload: dict
) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": f"sha256={sig}",
    }
    try:
        r = client.post(f"{url}/webhook", content=body, headers=headers, timeout=30.0)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return r.status_code, data
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": str(exc)}


def poll_searchable(
    client: httpx.Client, url: str, source_id: str, token: str,
    poll_interval: float, deadline: float,
) -> bool:
    while time.monotonic() < deadline:
        try:
            r = client.post(
                f"{url}/search",
                json={"query": token, "top_k": 10, "source_id": source_id},
                timeout=15.0,
            )
            if r.status_code == 200:
                for c in r.json().get("chunks", []):
                    if token in (c.get("snippet") or ""):
                        return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(poll_interval)
    return False


# ── metrics scrape ──────────────────────────────────────────────────────────

def scrape_metrics(client: httpx.Client, url: str) -> dict:
    out: dict = {}
    try:
        r = client.get(f"{url}/metrics", timeout=10.0)
        if r.status_code != 200:
            return out
        for line in r.text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name, _, val = line.partition(" ")
            base = name.split("{", 1)[0]
            try:
                fval = float(val)
            except ValueError:
                continue
            if base == "process_resident_memory_bytes":
                out["rss_bytes"] = fval
            elif base == "treeloom_queue_saturation":
                out["queue_saturation"] = max(out.get("queue_saturation", 0.0), fval)
            elif base == "treeloom_webhook_shed_total":
                out["webhook_shed_total"] = out.get("webhook_shed_total", 0.0) + fval
            elif base == "treeloom_merge_to_searchable_seconds_sum":
                out["m2s_sum"] = fval
            elif base == "treeloom_merge_to_searchable_seconds_count":
                out["m2s_count"] = fval
    except Exception:  # noqa: BLE001
        pass
    return out


# ── index bootstrap ─────────────────────────────────────────────────────────

def index_repo(client: httpx.Client, url: str, repo_url: str, branch: str) -> str:
    r = client.post(
        f"{url}/index-repo",
        json={"url": repo_url, "branch": branch, "force": True, "skip_graph": False},
        timeout=30.0,
    )
    if r.status_code == 400 and "Unsafe git URL" in r.text:
        # The indexer refuses loopback git remotes by default. Every URL this
        # harness serves is git://localhost/<repo>, so this is the first thing
        # that fails — say what to do instead of surfacing a bare 400.
        raise RuntimeError(
            f"the indexer at {url} rejected {repo_url} as an unsafe git URL.\n"
            "It refuses git remotes on the loopback interface unless told "
            "otherwise, and this harness serves every fixture over "
            "git://localhost/. Restart the target indexer with "
            "TREELOOM_GIT_ALLOW_LOOPBACK=true."
        )
    r.raise_for_status()
    data = r.json()
    job_id = data["job_id"]
    source_id = data.get("source_id", "")
    deadline = time.time() + 180
    while time.time() < deadline:
        j = client.get(f"{url}/jobs/{job_id}", timeout=10.0).json()
        st = j.get("status")
        if st == "done":
            return source_id
        if st in ("failed", "error"):
            raise RuntimeError(f"initial index failed for {repo_url}: {j.get('error')}")
        time.sleep(1.0)
    raise RuntimeError(f"initial index timed out for {repo_url}")


# ── result model ────────────────────────────────────────────────────────────

@dataclass
class MergeResult:
    merge_id: str
    repo_url: str
    rel_path: str
    webhook_status: int = 0
    collapsed: bool = False          # acked as in-flight (same-source dedup)
    shed: bool = False               # 429 backpressure
    searchable: bool = False
    latency_seconds: float = -1.0
    error: str | None = None


@dataclass
class RunResult:
    run_id: str
    started_at: str
    finished_at: str = ""
    config: dict = field(default_factory=dict)
    merges: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return -1.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[idx]


# ── driver ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    base = Path(args.base_path).resolve()
    base.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    result = RunResult(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        config=vars(args).copy(),
    )

    client = httpx.Client()
    daemon: subprocess.Popen | None = None
    commit_lock = threading.Lock()

    try:
        # 1. Fixtures + daemon
        print(f"Creating {args.repos} fixture repos under {base} …")
        repos: list[tuple[str, Path]] = []  # (name, path)
        for i in range(args.repos):
            name = f"repo{i}"
            create_repo(base, name)
            repos.append((name, base / name))
        daemon = start_git_daemon(base)
        print(f"git daemon serving git://localhost/<repo> (pid {daemon.pid})")

        # 2. Initial index of every repo
        sources: list[dict] = []  # {repo_url, branch, source_id, path, name}
        for name, path in repos:
            repo_url = f"git://localhost/{name}"
            print(f"Indexing {repo_url} …", end=" ", flush=True)
            sid = index_repo(client, args.url, repo_url, args.branch)
            sources.append({
                "repo_url": repo_url, "branch": args.branch,
                "source_id": sid, "path": path, "name": name,
            })
            print(f"source_id={sid}")

        # 3. Baseline metrics
        m_start = scrape_metrics(client, args.url)

        # 4. Drive merges (round-robin), poll searchable asynchronously
        interval = 60.0 / args.rate if args.rate > 0 else 0.0
        deadline_per = args.search_timeout
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.repos * 2 + 4)
        futures: list[concurrent.futures.Future] = []
        results: list[MergeResult] = []
        results_lock = threading.Lock()

        print(f"\nDriving {args.merges} merges at {args.rate}/min across "
              f"{args.repos} repos (per-merge searchable timeout {deadline_per}s)…")
        loop_start = time.monotonic()

        def do_searchable(mr: MergeResult, source_id: str, token: str, t0: float):
            sc = poll_searchable(
                client, args.url, source_id, token,
                args.poll_interval, t0 + deadline_per,
            )
            mr.searchable = sc
            mr.latency_seconds = (time.monotonic() - t0) if sc else -1.0
            with results_lock:
                done = sum(1 for r in results if r.latency_seconds >= 0 or r.error)
            tag = f"{mr.latency_seconds:6.2f}s" if sc else "  TIMEOUT"
            print(f"  [{mr.rel_path:40s}] {tag}"
                  + ("  (collapsed)" if mr.collapsed else ""))

        for n in range(args.merges):
            src = sources[n % len(sources)]
            token = uuid.uuid4().hex
            rel_path = f"loadtest/probe_{token}.py"
            mr = MergeResult(merge_id=token, repo_url=src["repo_url"], rel_path=rel_path)

            # commit the real probe file (serialized; commits are cheap)
            with commit_lock:
                fp = src["path"] / rel_path
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(_probe_source(token))
                try:
                    _git(src["path"], "add", rel_path)
                    _git(src["path"], "commit", "-q", "-m", f"probe {token}")
                except subprocess.CalledProcessError as exc:
                    mr.error = f"git commit failed: {exc.stderr}"
                    results.append(mr)
                    continue

            t0 = time.monotonic()
            status, data = post_webhook(client, args.url, args.secret,
                                        build_payload(src["repo_url"], src["branch"],
                                                      rel_path, token))
            mr.webhook_status = status
            msg = (data.get("message") or "").lower()
            if status == 429:
                mr.shed = True
            elif "in-flight" in msg or "already covers" in msg:
                mr.collapsed = True
            results.append(mr)

            if status in (200, 202) and not mr.shed and not mr.collapsed:
                futures.append(pool.submit(do_searchable, mr, src["source_id"], token, t0))
            elif mr.collapsed:
                # measure it anyway — its file may become searchable via a later job
                futures.append(pool.submit(do_searchable, mr, src["source_id"], token, t0))

            if interval and n < args.merges - 1:
                time.sleep(interval)

        offered_window = time.monotonic() - loop_start
        print("\nAll merges offered; waiting for outstanding searchable polls…")
        concurrent.futures.wait(futures)
        pool.shutdown(wait=True)

        # 5. Final metrics
        time.sleep(2.0)
        m_end = scrape_metrics(client, args.url)

        # 6. Summarize
        result.merges = [asdict(_clean(r)) for r in results]
        accepted = [r for r in results if r.webhook_status in (200, 202) and not r.shed]
        searchable = [r for r in results if r.searchable]
        latencies = [r.latency_seconds for r in results if r.searchable and r.latency_seconds >= 0]
        collapsed = [r for r in results if r.collapsed]
        shed = [r for r in results if r.shed]

        p50, p95, p99 = _pct(latencies, 50), _pct(latencies, 95), _pct(latencies, 99)
        rss_start = m_start.get("rss_bytes")
        rss_end = m_end.get("rss_bytes")
        rss_growth_pct = (
            (rss_end - rss_start) / rss_start * 100.0
            if rss_start and rss_end else None
        )
        achieved_rate = len(results) / offered_window * 60.0 if offered_window else 0.0

        p95_pass = p95 >= 0 and p95 <= args.p95_target_seconds
        mem_pass = rss_growth_pct is None or rss_growth_pct <= args.memory_growth_limit_pct
        searchable_pass = len(searchable) > 0
        verdict = p95_pass and mem_pass and searchable_pass

        result.summary = {
            "offered": len(results),
            "accepted": len(accepted),
            "collapsed": len(collapsed),
            "shed": len(shed),
            "searchable": len(searchable),
            "latency_p50": p50, "latency_p95": p95, "latency_p99": p99,
            "p95_target_seconds": args.p95_target_seconds,
            "rate_offered_per_min": args.rate,
            "rate_achieved_per_min": round(achieved_rate, 1),
            "queue_saturation_peak": m_end.get("queue_saturation", 0.0),
            "webhook_shed_total_delta": (
                m_end.get("webhook_shed_total", 0.0) - m_start.get("webhook_shed_total", 0.0)
            ),
            "rss_start_bytes": rss_start, "rss_end_bytes": rss_end,
            "rss_growth_pct": rss_growth_pct,
            "histogram_m2s_count_delta": (
                m_end.get("m2s_count", 0.0) - m_start.get("m2s_count", 0.0)
            ),
            "p95_pass": p95_pass, "memory_pass": mem_pass,
            "searchable_pass": searchable_pass, "verdict": verdict,
        }
        result.finished_at = datetime.now(timezone.utc).isoformat()

        _print_summary(result.summary, run_id, args)
        out_path = _write_result(result)
        print(f"\nResult written: {out_path}")
        return 0 if verdict else 1

    finally:
        if daemon is not None:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except Exception:  # noqa: BLE001
                daemon.kill()
        client.close()


def _clean(r: MergeResult) -> MergeResult:
    # path objects aren't in MergeResult; nothing to scrub, kept for symmetry
    return r


def _print_summary(s: dict, run_id: str, args: argparse.Namespace) -> None:
    def mb(b):
        return f"{b/1e6:7.1f} MB" if b else "    n/a"
    line = "─" * 62
    print(f"\n{line}")
    print("  TREELOOM MERGE→SEARCHABLE LOAD TEST — SUMMARY")
    print(line)
    print(f"  Run ID  : {run_id}")
    print(f"  URL     : {args.url}")
    print(f"  Repos   : {args.repos}   Rate: {args.rate}/min   Merges: {args.merges}")
    print()
    print(f"  Offered    : {s['offered']:5d} merges")
    print(f"  Accepted   : {s['accepted']:5d}")
    print(f"  Collapsed  : {s['collapsed']:5d}  (same-source in-flight dedup)")
    print(f"  Shed (429) : {s['shed']:5d}")
    print(f"  Searchable : {s['searchable']:5d}  (of offered)")
    print()
    pass_tag = "✓ PASS" if s["p95_pass"] else "✗ FAIL"
    print(f"  Latency p50 : {s['latency_p50']:7.2f}s")
    print(f"  Latency p95 : {s['latency_p95']:7.2f}s  {pass_tag} (target {s['p95_target_seconds']}s)")
    print(f"  Latency p99 : {s['latency_p99']:7.2f}s")
    print()
    print(f"  Throughput offered  : {s['rate_offered_per_min']:.1f} merges/min")
    print(f"  Throughput achieved : {s['rate_achieved_per_min']:.1f} merges/min")
    print(f"  Queue saturation pk : {s['queue_saturation_peak']:.2f}")
    print(f"  Webhook shed delta  : {s['webhook_shed_total_delta']:.0f}")
    print()
    mem_tag = "✓ PASS" if s["memory_pass"] else "✗ FAIL"
    gp = s["rss_growth_pct"]
    print(f"  RSS start  : {mb(s['rss_start_bytes'])}")
    print(f"  RSS end    : {mb(s['rss_end_bytes'])}")
    print(f"  RSS growth : {gp:6.1f}%  {mem_tag}" if gp is not None else "  RSS growth :    n/a")
    print()
    print(f"{line}")
    print(f"  VERDICT: {'PASS' if s['verdict'] else 'FAIL'}")
    print(line)


def _write_result(result: RunResult) -> Path:
    out_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "loadtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{ts}_merge_searchable.json"
    out_path.write_text(json.dumps(asdict(result), indent=1, default=str))
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8002",
                   help="Isolated indexer base URL")
    p.add_argument("--secret", required=True,
                   help="WEBHOOK_GITHUB_SECRET the indexer is configured with")
    p.add_argument("--base-path", default="/tmp/treeloom_loadtest",
                   help="Directory for throwaway repos + git daemon base-path")
    p.add_argument("--branch", default="main")
    p.add_argument("--repos", type=int, default=6,
                   help="Number of independent repos (concurrency without same-source collapse)")
    p.add_argument("--merges", type=int, default=60, help="Total merges to drive")
    p.add_argument("--rate", type=float, default=30.0, help="Offered merges/min (aggregate)")
    p.add_argument("--search-timeout", type=float, default=60.0,
                   help="Per-merge searchable deadline (seconds)")
    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--p95-target-seconds", type=float, default=60.0)
    p.add_argument("--memory-growth-limit-pct", type=float, default=50.0)
    args = p.parse_args()

    try:
        return run(args)
    except KeyboardInterrupt:
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
