"""Cost-aware embedding proxy for multiple TEI backends.

Routes each batch to the backend with the lowest *in-flight estimated token
cost* rather than the lowest connection count. Token cost is estimated as
`sum(len(t) // 3 for t in batch)` — the same heuristic `_pick_url` already
uses for GPU/CPU classification.

Preserves the predictive GPU/CPU split: any batch containing a chunk whose
estimated tokens exceed `GPU_MAX_TOKENS` is pinned to a CPU-class backend.
Falls back to the next-cheapest backend on HTTP error.

Interface is compatible with the existing `embed(texts) -> list[list[float]]`
free function — `EmbeddingProxy.embed` is the same shape.

See docs/adr-001-cost-aware-embedding-proxy.md for the rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class _CostEstimator(Protocol):
    def batch_cost(self, texts: list[str]) -> int: ...
    def max_chunk_cost(self, texts: list[str]) -> int: ...


class _CharHeuristicEstimator:
    """Char/3 fallback when the real tokenizer cannot be loaded."""

    def batch_cost(self, texts: list[str]) -> int:
        return sum(len(t) // 3 for t in texts)

    def max_chunk_cost(self, texts: list[str]) -> int:
        return max((len(t) // 3 for t in texts), default=0)


class _TokenizerEstimator:
    """Real tokenizer from HuggingFace, keyed by EMBEDDING_MODEL."""

    def __init__(self, model: str) -> None:
        from tokenizers import Tokenizer

        try:
            self._tok = Tokenizer.from_pretrained(model)
        except Exception:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=model, filename="tokenizer.json")
            self._tok = Tokenizer.from_file(path)

    def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def batch_cost(self, texts: list[str]) -> int:
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        return sum(len(e.ids) for e in encs)

    def max_chunk_cost(self, texts: list[str]) -> int:
        if not texts:
            return 0
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        return max(len(e.ids) for e in encs)


_estimator: _CostEstimator | None = None


def _get_estimator() -> _CostEstimator:
    global _estimator
    if _estimator is not None:
        return _estimator
    model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not model:
        logger.warning("[embed] EMBEDDING_MODEL not set; using char/3 cost heuristic")
        _estimator = _CharHeuristicEstimator()
        return _estimator
    try:
        _estimator = _TokenizerEstimator(model)
        logger.info("[embed] loaded tokenizer for %s", model)
    except Exception as exc:
        logger.warning("[embed] failed to load tokenizer for %s (%s); using char/3 heuristic", model, exc)
        _estimator = _CharHeuristicEstimator()
    return _estimator


class BackendClass(Enum):
    GPU = "gpu"
    CPU = "cpu"


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation — requests pass through
    OPEN = "open"            # Tripped — requests blocked until cooldown
    HALF_OPEN = "half_open"  # Probing — single request allowed through


# Circuit breaker defaults (overridable via env)
_CB_FAILURE_THRESHOLD = int(os.environ.get("EMBED_CB_THRESHOLD", "5"))
_CB_WINDOW_SECONDS = float(os.environ.get("EMBED_CB_WINDOW", "60.0"))
_CB_COOLDOWN_SECONDS = float(os.environ.get("EMBED_CB_COOLDOWN", "30.0"))


@dataclass
class Backend:
    url: str
    klass: BackendClass
    in_flight_tokens: int = 0
    failures: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # --- Circuit breaker state ---
    cb_state: CircuitState = CircuitState.CLOSED
    cb_failure_times: list[float] = field(default_factory=list)  # monotonic timestamps of recent failures
    cb_opened_at: float = 0.0  # monotonic timestamp when tripped to OPEN
    # Backend's advertised max inputs-per-/embed-request, learned from a 422
    # "batch size N > maximum allowed batch size M". None = unknown
    # (send optimistically); once learned, oversized batches are pre-split.
    max_batch_size: int | None = None


# A TEI backend rejects an oversized /embed with HTTP 422 and a body like
# {"error":"batch size 40 > maximum allowed batch size 32"}. We parse the cap so
# the proxy can split + retry on the SAME backend instead of tripping the
# circuit breaker (the batch is too big, the backend is healthy).
_BATCH_LIMIT_RE = re.compile(r"maximum allowed batch size (\d+)")


def _parse_batch_limit(text: str) -> int | None:
    m = _BATCH_LIMIT_RE.search(text or "")
    return int(m.group(1)) if m else None


def _estimate_batch_tokens(batch: list[str]) -> int:
    return _get_estimator().batch_cost(batch)


def _estimate_max_chunk_tokens(batch: list[str]) -> int:
    return _get_estimator().max_chunk_cost(batch)


class EmbeddingProxy:
    """Least-loaded (by pending token cost) router across TEI backends.

    Backends can be live-reloaded from a Postgres registry; see
    `from_registry()`, `reload()`, and `start_listener()`. The legacy
    `from_env()` path stays for development/CI use without a database.
    """

    def __init__(
        self,
        backends: list[Backend],
        *,
        gpu_max_tokens: int = 4096,
        max_batch_size: int = 128,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not backends:
            raise ValueError("EmbeddingProxy requires at least one backend")
        self._backends = backends
        self._gpu_max_tokens = gpu_max_tokens
        self._max_batch_size = max_batch_size
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._select_lock = asyncio.Lock()
        # Set by start_listener; nulls keep stop_listener idempotent.
        self._listener_task: asyncio.Task | None = None
        self._listener_conn = None  # asyncpg.Connection | None
        self._listener_stopping = False

    @classmethod
    def from_env(cls) -> EmbeddingProxy:
        gpu_urls = [u.strip() for u in os.environ.get("EMBEDDING_URLS", "").split(",") if u.strip()]
        if not gpu_urls and (single := os.environ.get("EMBEDDING_URL", "").strip()):
            gpu_urls = [single]
        cpu_urls = [u.strip() for u in os.environ.get("EMBEDDING_FALLBACK_URLS", "").split(",") if u.strip()]
        if not cpu_urls and (single := os.environ.get("EMBEDDING_FALLBACK_URL", "").strip()):
            cpu_urls = [single]

        backends = [Backend(url=u, klass=BackendClass.GPU) for u in gpu_urls]
        backends.extend(Backend(url=u, klass=BackendClass.CPU) for u in cpu_urls)
        return cls(
            backends=backends,
            gpu_max_tokens=int(os.environ.get("GPU_MAX_TOKENS", "4096")),
            max_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "128")),
        )

    @classmethod
    async def from_registry(cls, store) -> EmbeddingProxy:
        """Build a proxy from the live `embedding_backends` registry."""
        rows = await store.list_enabled()
        backends = [
            Backend(url=r.url, klass=BackendClass(r.klass))
            for r in rows
        ]
        return cls(
            backends=backends,
            gpu_max_tokens=int(os.environ.get("GPU_MAX_TOKENS", "4096")),
            max_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "128")),
        )

    async def reload(self, store) -> None:
        """Re-read the registry and swap the backend list.

        Carries over `in_flight_tokens` and `failures` for URLs that
        already exist by reusing the same `Backend` instance — that way
        any in-flight request still referencing the old object sees
        consistent state.

        Empty registry keeps the last good config and logs a warning.
        """
        rows = await store.list_enabled()
        if not rows:
            logger.warning(
                "[embed] reload: registry is empty; keeping last good backend list (%d entries)",
                len(self._backends),
            )
            return

        old_by_url = {b.url: b for b in self._backends}
        new_backends: list[Backend] = []
        for r in rows:
            if r.url in old_by_url:
                existing = old_by_url[r.url]
                existing.klass = BackendClass(r.klass)
                new_backends.append(existing)
            else:
                new_backends.append(Backend(url=r.url, klass=BackendClass(r.klass)))

        async with self._select_lock:
            self._backends = new_backends
        logger.info("[embed] reloaded backends: %d entries", len(new_backends))

    async def start_listener(self, store, dsn: str) -> None:
        """Spawn a background task that LISTENs on `embedding_backends_changed`.

        Uses a *dedicated* asyncpg connection — the pool's release/reuse
        cycle would break LISTEN bindings. Reconnects on connection drop
        with exponential backoff (cap 30s). Idempotent: a second call
        replaces the running listener.
        """
        await self.stop_listener()
        self._listener_stopping = False
        self._listener_task = asyncio.create_task(
            self._listener_supervisor(store, dsn)
        )

    async def stop_listener(self) -> None:
        self._listener_stopping = True
        task = self._listener_task
        self._listener_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._listener_conn is not None:
            try:
                await self._listener_conn.close()
            except Exception:
                pass
            self._listener_conn = None

    async def _listener_supervisor(self, store, dsn: str) -> None:
        from treeloom.adapters.postgresql.embedding_backend_store import NOTIFY_CHANNEL

        import asyncpg

        backoff = 1.0
        while not self._listener_stopping:
            try:
                self._listener_conn = await asyncpg.connect(dsn)

                def _on_notify(_conn, _pid, _channel, _payload):
                    # The callback runs in the asyncpg connection's task
                    # context; schedule the reload as its own task so we
                    # don't block notifications.
                    asyncio.create_task(self.reload(store))

                await self._listener_conn.add_listener(NOTIFY_CHANNEL, _on_notify)
                logger.info("[embed] listening on Postgres channel %r", NOTIFY_CHANNEL)
                backoff = 1.0  # reset after successful connect

                # Hold the connection open. asyncpg dispatches NOTIFY
                # callbacks on its own reader task — this loop just
                # parks until cancellation or the conn drops.
                while not self._listener_stopping:
                    if self._listener_conn.is_closed():
                        raise ConnectionError("listener connection closed")
                    await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._listener_stopping:
                    break
                logger.warning(
                    "[embed] listener disconnected (%s); reconnecting in %.1fs",
                    exc, backoff,
                )
                try:
                    if self._listener_conn is not None and not self._listener_conn.is_closed():
                        await self._listener_conn.close()
                except Exception:
                    pass
                self._listener_conn = None
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, 30.0)

    async def _select(self, *, require_cpu: bool) -> Backend | None:
        """Pick the backend with the lowest in-flight token cost.

        Holding the select lock during the read+pick is the cheap way to keep
        two concurrent callers from picking the same idle backend; we release
        before issuing the request.

        Returns None when all eligible backends are tripped (OPEN or failed probe).
        """
        now = time.monotonic()
        async with self._select_lock:
            pool = [b for b in self._backends if b.klass is BackendClass.CPU] if require_cpu else list(self._backends)
            if not pool:
                pool = list(self._backends)  # no CPU configured — fall through to whatever exists

            # Promote any OPEN backend past its cooldown to HALF_OPEN
            for b in pool:
                if b.cb_state is CircuitState.OPEN and (now - b.cb_opened_at) >= _CB_COOLDOWN_SECONDS:
                    b.cb_state = CircuitState.HALF_OPEN
                    logger.info("[embed] backend %s OPEN→HALF_OPEN (cooldown elapsed)", b.url)

            # Filter: CLOSED or HALF_OPEN are eligible; OPEN is not
            eligible = [b for b in pool if b.cb_state is not CircuitState.OPEN]
            if not eligible:
                return None
            return min(eligible, key=lambda b: b.in_flight_tokens)

    async def _post_split(self, backend: Backend, batch: list[str], limit: int) -> list[list[float]]:
        """Post `batch` to `backend` in order, in chunks of `limit`, concatenating
        the embeddings. Used when a batch exceeds the backend's max-batch-size
; backend.max_batch_size is already set so the per-chunk
        _post_one calls won't re-trigger the 422 split."""
        out: list[list[float]] = []
        for i in range(0, len(batch), limit):
            out.extend(await self._post_one(backend, batch[i : i + limit]))
        return out

    async def _post_one(self, backend: Backend, batch: list[str]) -> list[list[float]]:
        from treeloom.infrastructure.tracing import get_tracer

        # Pre-split if we've already learned this backend's batch ceiling, so we
        # never knowingly send an oversized request.
        if backend.max_batch_size and len(batch) > backend.max_batch_size:
            return await self._post_split(backend, batch, backend.max_batch_size)

        cost = _estimate_batch_tokens(batch)
        async with backend._lock:
            backend.in_flight_tokens += cost
        with get_tracer("treeloom.tei").start_as_current_span(
            "tei.embed",
            attributes={
                "treeloom.backend_url": backend.url,
                "treeloom.backend_class": backend.klass.value,
                "treeloom.token_cost": cost,
                "treeloom.batch_size": len(batch),
            },
        ):
            try:
                resp = await self._client.post(
                    f"{backend.url}/embed",
                    json={"inputs": batch, "normalize": True},
                )
                # Oversized-batch 422: the backend is HEALTHY, the batch is too
                # big. Learn the cap, split, and retry on the SAME backend —
                # don't trip the circuit breaker or fall through to a backend
                # with the same limit.
                if resp.status_code == 422:
                    limit = _parse_batch_limit(resp.text)
                    if limit and len(batch) > limit:
                        backend.max_batch_size = limit
                        logger.warning(
                            "[embed] backend %s rejected batch of %d (max %d); "
                            "learned cap, splitting", backend.url, len(batch), limit,
                        )
                        async with backend._lock:
                            backend.in_flight_tokens -= cost
                        return await self._post_split(backend, batch, limit)
                resp.raise_for_status()
                # Success — reset circuit breaker
                async with backend._lock:
                    backend.in_flight_tokens -= cost
                    if backend.cb_state is CircuitState.HALF_OPEN:
                        backend.cb_state = CircuitState.CLOSED
                        backend.cb_failure_times.clear()
                        logger.info("[embed] backend %s HALF_OPEN→CLOSED (probe succeeded)", backend.url)
                    elif backend.cb_state is CircuitState.CLOSED:
                        backend.cb_failure_times.clear()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.HTTPError) as exc:
                # Track failure with sliding window; trip if threshold exceeded
                now = time.monotonic()
                async with backend._lock:
                    backend.in_flight_tokens -= cost
                    backend.failures += 1
                    backend.cb_failure_times.append(now)
                    # Prune old failures outside the window
                    cutoff = now - _CB_WINDOW_SECONDS
                    backend.cb_failure_times = [t for t in backend.cb_failure_times if t >= cutoff]
                    recent = len(backend.cb_failure_times)
                    if backend.cb_state is CircuitState.CLOSED and recent >= _CB_FAILURE_THRESHOLD:
                        backend.cb_state = CircuitState.OPEN
                        backend.cb_opened_at = now
                        logger.warning(
                            "[embed] backend %s CLOSED→OPEN (%d failures in %.0fs, threshold=%d)",
                            backend.url, recent, _CB_WINDOW_SECONDS, _CB_FAILURE_THRESHOLD,
                        )
                    elif backend.cb_state is CircuitState.HALF_OPEN:
                        backend.cb_state = CircuitState.OPEN
                        backend.cb_opened_at = now
                        logger.warning("[embed] backend %s HALF_OPEN→OPEN (probe failed)", backend.url)
                raise

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        require_cpu = _estimate_max_chunk_tokens(batch) > self._gpu_max_tokens
        tried: set[str] = set()
        last_exc: Exception | None = None

        while len(tried) < len(self._backends):
            backend = await self._select(require_cpu=require_cpu)
            if backend is None:
                # All backends are tripped (OPEN) — wait for cooldown and retry
                logger.warning("[embed] all backends tripped; waiting for cooldown")
                raise RuntimeError("All embedding backends are unavailable (circuit breaker tripped)")
            if backend.url in tried:
                # Selected backend already tried — widen the pool by relaxing the CPU constraint
                if require_cpu:
                    require_cpu = False
                    continue
                break
            tried.add(backend.url)
            try:
                return await self._post_one(backend, batch)
            except (httpx.HTTPStatusError, httpx.HTTPError) as exc:
                last_exc = exc
                logger.warning("[embed] backend %s failed (%s); trying next", backend.url, exc)
                continue

        assert last_exc is not None
        raise last_exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch_size):
            out.extend(await self._embed_batch(texts[i : i + self._max_batch_size]))
        return out


MAX_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "128"))

_proxy: EmbeddingProxy | None = None


def set_proxy(proxy: EmbeddingProxy) -> None:
    """Install a process-global proxy. The indexer startup hook calls this
    after building the registry-backed proxy so call sites that use
    `get_proxy()` pick it up — including legacy free functions and tests."""
    global _proxy
    _proxy = proxy


def get_proxy() -> EmbeddingProxy:
    """Return the process-global proxy, falling back to env-based init.

    The fallback exists for tests, scripts, and dev runs without
    Postgres. The indexer service overrides this at startup via
    `set_proxy(await EmbeddingProxy.from_registry(...))`.
    """
    global _proxy
    if _proxy is None:
        _proxy = EmbeddingProxy.from_env()
    return _proxy


EMBEDDING_QUERY_PREFIX = os.environ.get("EMBEDDING_QUERY_PREFIX", "")


async def embed(texts: list[str]) -> list[list[float]]:
    return await get_proxy().embed(texts)


async def embed_query(texts: list[str]) -> list[list[float]]:
    """Embed search queries, applying the model's instruction prefix if configured."""
    if EMBEDDING_QUERY_PREFIX:
        texts = [EMBEDDING_QUERY_PREFIX + t for t in texts]
    return await get_proxy().embed(texts)


class TEIEmbeddingAdapter:
    """EmbeddingPort adapter that delegates to the module-level cost-aware proxy.

    The domain port declares `embed` as sync, but every call site awaits the
    underlying free function, so this adapter matches the actual usage (async).
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await get_proxy().embed(texts)
