# ADR-001: Cost-Aware Embedding Proxy (vs. nginx / LiteLLM)

- Status: Accepted
- Date: 2026-05-28
- Owner: indexer / embedding adapter

> An ADR records a decision at a point in time. The Context below describes
> the code as it was on the date above; the Decision and its consequences are
> what still bind. Notes marked *[2026-09]* say what has changed since.

## Context

The indexer fans out `embed(texts)` calls to one or more TEI (HuggingFace Text
Embeddings Inference) backends. The then-current adapter (`src/treeloom/adapters/tei/embedding_adapter.py`, since
replaced by `embedding_proxy.py`) advertised
"round-robin with failover" but in practice it:

1. Iterates `EMBEDDING_URLS` starting from index 0 on every call (so it is
   first-try-then-failover, not round-robin).
2. Defines `_pick_url()` for predictive GPU/CPU routing — but `embed()` never
   calls it.

Even if true round-robin were wired up, it would still be wrong for our
workload. Embedding request cost is highly **non-uniform** and **invisible at
the connection level**:

- A batch of 256 short snippets and a batch of 256 two-thousand-line files
  look identical to a connection-counting balancer (same count, same in-flight
  request) but differ by ~an order of magnitude in GPU/CPU time.
- TEI's `/embed` endpoint is synchronous from the client's perspective, so
  slow batches do block their backend — but a round-robin balancer keeps
  feeding it work anyway, while an idle peer sits empty.
- The chunker can emit oversized chunks that exceed the GPU's context window
  and OOM-kill the TEI process (see `docs/gpu-cpu-routing.md`). Those batches
  must be **pinned** to a CPU backend, not balanced.

We want routing that is aware of (a) the estimated token cost of each batch
and (b) which backends can physically serve it.

## Decision

Introduce `EmbeddingProxy` in `src/treeloom/adapters/tei/embedding_proxy.py`:

- Each backend carries an `in_flight_tokens` counter that is incremented by
  the **tokenizer-reported token count** of the batch before the POST and
  decremented when the response (or failure) returns.
- Token counts come from the actual model tokenizer (loaded via
  `tokenizers.Tokenizer.from_pretrained(EMBEDDING_MODEL)`, with a
  `huggingface_hub.hf_hub_download("tokenizer.json")` fallback for cases where
  the auto-load path doesn't resolve). A char/3 heuristic remains as a
  last-resort fallback if the tokenizer can't be loaded at all (air-gapped,
  missing `EMBEDDING_MODEL`).
- For each batch, pick the backend with the **lowest `in_flight_tokens`**
  among eligible backends. Ties broken by failure count.
- Eligibility: if `max(token_count_per_chunk) > GPU_MAX_TOKENS`, restrict the
  pool to `CPU`-class backends (preserves the existing predictive guard
  against GPU OOM, now computed against real tokens).
- On HTTP error, mark a failure, drop the backend from the per-batch retry
  set, and re-select.
- Keep the `embed(texts: list[str]) -> list[list[float]]` free function as a
  thin shim over a module-global `EmbeddingProxy.from_env()` so existing call
  sites do not change.
- `TEIEmbeddingAdapter` (the class `application/wiring.py` imports as an
  `EmbeddingPort`) is a thin async wrapper around the same proxy.

## Rejected alternatives

### nginx (or any L7 reverse proxy)

- Routes on connections / requests, not on payload cost. Would re-create the
  exact "256 tiny vs. 256 huge batches look identical" problem.
- The GPU-vs-CPU pin needs to inspect *payload* (max chunk length). Doing that
  in nginx means a Lua/njs body-reading filter that parses JSON before
  proxying — operationally heavier than a 150-line Python class, and it
  doesn't observe success/failure to update its routing weights.
- Adds a separate process to run, configure, and observe; our deploy is a
  single Python service plus the TEI containers.

### LiteLLM Router

- LiteLLM's `least-busy` strategy counts **requests in flight**, not token
  cost. Same blind spot as round-robin for our workload, since our request
  count per batch is always 1.
- Its `usage-based-routing` mode tracks input/output tokens, but it is
  designed around per-minute quota windows for paid model APIs, not the
  instantaneous-load question we actually need answered. We would still be
  computing our own cost estimates and feeding them in.
- LiteLLM does not model "this backend can't accept payloads above N tokens"
  — the GPU-OOM pin would need a custom router subclass anyway.
- Pulls in a dependency with a substantial surface area (provider adapters,
  callback system, proxy server, key management) we don't otherwise use. Two
  TEI backends do not justify it.

We may revisit LiteLLM if/when we start mixing in hosted embedding providers
with rate limits or auth — that is the workload it was built for.

## Consequences

### Positive

- Routing decision matches the actual scarce resource (GPU/CPU time
  per-batch), not a proxy for it.
- GPU OOM predictive-pin and reactive-failover live in one place instead of
  one dead helper + one inlined retry loop.
- Surfaces a real `EmbeddingProxy` class that can be unit-tested with a fake
  `httpx.AsyncClient`; the current code can only be tested end-to-end.
- `TEIEmbeddingAdapter` (at the time imported by `application/wiring.py` but
  not defined anywhere — broken) can be implemented as a thin wrapper around
  the proxy, closing that gap. *[2026-09: done — it lives in
  `embedding_proxy.py`.]*

### Negative

- Adds two runtime deps: `tokenizers` (HF Rust wheel, ~3MB, no torch) and
  `huggingface_hub` (for the `tokenizer.json` fallback path). First indexer
  start with a cold HF cache downloads the tokenizer (~MBs, one-time).
- Token count from the tokenizer is a precise measure of input tokens, not
  GPU wall-clock time. Tokens-to-seconds is reliably monotonic at fixed
  batch shape but not strictly linear (attention is roughly quadratic in
  sequence length). For our batches this is close enough that we accept it.
- The `in_flight_tokens` counter lives in-process. If we ever run multiple
  indexer replicas pointed at the same TEI backends, each replica will pick
  independently — which was fine for a single indexer host but would need
  revisiting before horizontal scale. *[2026-09: the indexer now supports
  several worker processes sharing a Postgres job queue, so this caveat is
  live — each process routes on its own counters. The backend list itself
  moved from `.env` into the Postgres `embedding_backends` registry, which
  every process reloads on change.]*
- We own the code. If a future need (rate-limit-aware routing, multi-provider
  cost models, retry budgets across calls) starts pulling us toward
  LiteLLM-shaped features, the build-vs-adopt calculus flips. ADR-002
  candidate.

## Out of scope

- Persisting routing stats across restarts.
- Cross-process load coordination (would need Redis or similar).
- Modelling attention-shape nonlinearity in the cost function. The
  tokenizer-token count is a good-enough proxy for relative load at the
  batch shapes we send; if profiling later shows pathological cases, the
  cost estimator is the only thing that needs to change.

## References

- `src/treeloom/adapters/tei/embedding_proxy.py` — the proxy, the tokenizer
  estimator, and `TEIEmbeddingAdapter`.
- `src/treeloom/embedder.py` — re-export shim used by call sites.
- `docs/gpu-cpu-routing.md` — predictive GPU/CPU split, OOM history, and the
  `GPU_MAX_TOKENS` threshold this proxy preserves.
