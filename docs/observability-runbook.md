# Observability runbook

How to collect every signal (metrics, logs, traces) from a Treeloom deployment into a central observability host — **<obs-host>**, running the OpenTelemetry Collector, Prometheus, Loki, Tempo and Grafana — and observe it in Grafana there.

**What this describes.** One reference deployment: a GPU host (`<gpu-host>`) running the HuggingFace Text Embeddings Inference (TEI) containers and a llama.cpp LLM server, a small Kubernetes (k3s) cluster (`<k3s-node-1..3>`) hosting Neo4j and Milvus, the compose services from this repo, and the indexer running on a host. Every hostname is a placeholder; adapt the topology to yours (for the default `local-infra` compose profile, Neo4j and Milvus are local containers, and the k3s sections do not apply). Sections marked **proposed** describe changes to this repository that have not been made; everything else is shipped.

## Topology

```
              ┌──────────────────────────── <obs-host> ────────────────────────────┐
              │  OTel Collector (4317/4318, prom exporter on 8889)                  │
              │    ─► Tempo  (traces)                                                │
              │    ─► Loki   (logs via otlphttp)                                     │
              │    ─► Prometheus scrapes :8889 for OTLP-pushed metrics               │
              │  Prometheus (9090) — also file_sd_configs for pull targets           │
              │  Loki (3100), Tempo (3200), Grafana (3000)                           │
              └──────────────────────────────────────────────────────────────────────┘
                   ▲              ▲                ▲                  ▲
   ┌───────────────┘              │                │                  └─────────────────┐
   │                              │                │                                    │
GPU host <gpu-host>         k3s (<k3s-nodes>)  Local compose (this repo)     Indexer (host)
- TEI/reranker containers    - Neo4j               - mcp-server                  - treeloom indexer
- llama-coder-server         - Milvus              - postgres + postgres-exporter - traces → OTLP
- node_exporter              - kube-state-metrics  - alloy → Loki                 - logs → file → alloy → Loki
- dcgm-exporter (GPU)        - node_exporter       - local-obs (PROPOSED)         - metrics → OTLP
- alloy → Loki               - alloy → Loki          see "Changes in this repo"
```

## What the central stack already provides

From the central host's own `docker-compose.yml` (the observability stack's compose
file — not this repo's; the `./grafana`, `./prometheus`, `./otel` paths below are
directories in that stack):

- `grafana` (3000) — anonymous admin, dashboards provisioned from `./grafana/dashboards/`
- `prometheus` (9090) — config at `./prometheus/prometheus.yml`, with `./prometheus/k8s/` already mounted for the k3s scrape jobs
- `loki` 3.1.0 (3100) — supports OTLP-native log ingest at `/otlp/v1/logs`
- `tempo` (3200)
- `otel-collector` (4317 gRPC, 4318 HTTP, 8889 prom-export) — already wired to expose OTLP-pushed metrics in Prometheus format on `:8889`

## Sources of signal

### GPU host (<gpu-host>)

| Container | Port | Signal | How |
|---|---|---|---|
| `tei-01` (reranker) | 8081 | metrics | TEI built-in `/metrics`, Prom scrape |
| `tei-02` (on this deployment the Qwen3 reranker service, not TEI) | 8086 | metrics | same |
| `tei-03` (embedding) | 8082 | metrics | same |
| `tei-04` | 8083 | metrics | same |
| `llama-coder-server` | 8080 | metrics | needs `--metrics` flag, then `/metrics`, Prom scrape |
| `node_exporter` (new) | 9100 | host metrics | Prom scrape |
| `dcgm-exporter` (new) | 9400 | GPU metrics | Prom scrape |
| all containers | — | logs | alloy docker discovery → OTLP → central collector → Loki |

### k3s cluster (<k3s-node-1..3>)

| Source | Port | Signal | How |
|---|---|---|---|
| Neo4j | 2004 (NodePort) | metrics | enable `server.metrics.prometheus.*`, Prom scrape |
| Milvus components | 9091 (NodePort) | metrics | enable metrics service, Prom scrape |
| kube-state-metrics, node_exporter | (existing) | metrics | already covered by `./prometheus/k8s/` |
| pod logs | — | logs | alloy DaemonSet → Loki |

### Local docker-compose (this repo)

| Service | Signal | How |
|---|---|---|
| `mcp-server` | traces | OTLP env passed in, ships to <obs-host>:4317 |
| `mcp-server` | logs | docker container logs picked up by host alloy → Loki |
| `postgres` | metrics | new `postgres-exporter` sidecar on 9187, Prom scrape |
| `postgres` | logs | container logs via host alloy |

### Indexer (host process from `run.sh`)

| Signal | How |
|---|---|
| traces | OTel SDK already initialized in startup; set `OTEL_EXPORTER_OTLP_ENDPOINT=http://<obs-host>:4317` in `.env` |
| logs | structlog already injects `otelTraceID`/`otelSpanID`; redirect stdout to a file, host alloy ships to Loki |
| metrics | the indexer serves Prometheus metrics at `GET /metrics` on its port (8001) — add `<indexer-host>:8001` to the scrape targets. (Optionally wire an OTel metrics exporter instead; pushed metrics land on `otel:8889`.) |

## Changes required on the central host

### 1. Extend the OTel collector config

The central host's collector config (`./otel/otel-collector-config.yml` in that stack) should have all three signal pipelines:

```yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }

processors:
  batch:
    timeout: 5s
    send_batch_size: 512
  tail_sampling:
    decision_wait: 30s
    num_traces: 50000
    expected_new_traces_per_sec: 100
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: keep-slow-root-spans
        type: latency
        latency: { threshold_ms: 30000 }
      - name: keep-baseline
        type: probabilistic
        probabilistic: { sampling_percentage: 10 }

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls: { insecure: true }
  otlphttp/loki:
    endpoint: http://loki:3100/otlp
  prometheus:
    endpoint: 0.0.0.0:8889
    resource_to_telemetry_conversion: { enabled: true }

service:
  pipelines:
    traces:  { receivers: [otlp], processors: [tail_sampling, batch], exporters: [otlp/tempo] }
    metrics: { receivers: [otlp], processors: [batch], exporters: [prometheus] }
    logs:    { receivers: [otlp], processors: [batch], exporters: [otlphttp/loki] }
```

Restart: `docker compose restart otel-collector`.

### 2. Add Prometheus scrape targets

On the central host, create `./prometheus/treeloom-targets.yml`:

```yaml
- targets: ['<gpu-host>:8081','<gpu-host>:8082','<gpu-host>:8083','<gpu-host>:8086']
  labels: { job: tei, host: gpu-host }
- targets: ['<gpu-host>:8080']
  labels: { job: llama-coder }
- targets: ['<gpu-host>:9100']
  labels: { job: node, host: gpu-host }
- targets: ['<gpu-host>:9400']
  labels: { job: dcgm, host: gpu-host }
- targets: ['<indexer-host>:8001']
  labels: { job: treeloom-indexer }
- targets: ['otel:8889']
  labels: { job: otel-collector-export }
# Filled in after k3s NodePorts are exposed:
# - targets: ['<k3s-node-1>:32004','<k3s-node-2>:32004','<k3s-node-3>:32004']
#   labels: { job: neo4j }
# - targets: ['<k3s-node-1>:32091', ...]
#   labels: { job: milvus }
# Filled in after postgres-exporter is up on the local docker host:
# - targets: ['<local-host-ip>:9187']
#   labels: { job: postgres }
```

Reference it from `./prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: treeloom
    file_sd_configs:
      - files: ['/etc/prometheus/treeloom-targets.yml']
```

Reload: `curl -X POST http://localhost:9090/-/reload` (or `docker compose restart prometheus`).

### 3. Provision dashboards

Drop into `./grafana/dashboards/`:

- `assets/grafana/treeloom-indexer-traces.json` from this repo
- DCGM exporter dashboard (grafana.com id `12239`)
- node_exporter full (id `1860`)
- Neo4j (id `2777`)
- Milvus official (id `13332`)
- llama.cpp (search "llama.cpp" on grafana.com)

Grafana auto-loads on next restart.

### 4. Tempo → Loki cross-link

In the Tempo datasource definition on the central host, ensure:

```yaml
jsonData:
  tracesToLogsV2:
    datasourceUid: loki
    tags: [{ key: 'service.name', value: 'service' }]
    filterByTraceID: true
```

Mirrors the local provisioning in `assets/observability/grafana-datasources.yaml`.

## Changes required on the GPU host (<gpu-host>)

### 1. Add `--metrics` to llama-coder-server

Verify with `curl -s http://<gpu-host>:8080/metrics | head`. If empty, restart the container with `--metrics` appended to its args.

### 2. Run node_exporter and dcgm-exporter

```bash
docker run -d --restart=unless-stopped --net=host --pid=host --name node-exporter \
  -v /:/host:ro,rslave quay.io/prometheus/node-exporter:latest --path.rootfs=/host

docker run -d --restart=unless-stopped --gpus all --name dcgm-exporter \
  -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
```

### 3. Run alloy for container logs → central collector

`/etc/alloy/config.river`:

```river
discovery.docker "containers" { host = "unix:///var/run/docker.sock" }

loki.source.docker "containers" {
  host       = "unix:///var/run/docker.sock"
  targets    = discovery.docker.containers.targets
  forward_to = [otelcol.receiver.loki.in.receiver]
}

otelcol.receiver.loki "in" {
  output { logs = [otelcol.exporter.otlp.central.input] }
}

otelcol.exporter.otlp "central" {
  client {
    endpoint = "<obs-host>:4317"
    tls { insecure = true }
  }
}
```

```bash
docker run -d --restart=unless-stopped --name alloy --net=host \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v /etc/alloy:/etc/alloy:ro \
  grafana/alloy:latest run /etc/alloy/config.river
```

## Changes required in k3s

### Neo4j

In `neo4j.conf` (or Helm values):

```
server.metrics.prometheus.enabled=true
server.metrics.prometheus.endpoint=0.0.0.0:2004
```

Expose Service of type NodePort on port 2004 (e.g. nodePort `32004`). Add the three node IPs:port to `treeloom-targets.yml`.

### Milvus

Enable component metrics. With the official Helm chart:

```yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: false   # we scrape externally from the central host
```

Expose each component's `:9091` via NodePort (or one aggregate NodePort Service). Add to `treeloom-targets.yml`.

### Pod logs

If you don't already have alloy or promtail as a DaemonSet, install the `grafana/alloy` Helm chart with the OTLP exporter pointed at `<obs-host>:4317`. Same config shape as the GPU host alloy.

## Changes required in this repo

### 1. `.env` (shipped)

`.env.example` already carries `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317` (the bundled local collector). Point it at the central one:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://<obs-host>:4317
```

### 2. `docker-compose.yml`

- **Shipped:** `mcp-server.environment` passes `OTEL_EXPORTER_OTLP_ENDPOINT` (via `MCP_OTEL_ENDPOINT`, default `http://otel-collector:4317`); set `MCP_OTEL_ENDPOINT` to point it at the central collector.
- **Proposed, not implemented:** move the `otel-collector`, `tempo` and `grafana` services under `profiles: [local-obs]`. Today they have no profile; `run.sh` starts services by name, so they only come up when you ask for them explicitly (`docker compose up -d otel-collector tempo grafana`), which has the same effect in practice.
- **Proposed, not implemented:** add a `postgres-exporter` sidecar:

  ```yaml
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    environment:
      DATA_SOURCE_NAME: "postgresql://${POSTGRES_USER:-treeloom}:${POSTGRES_PASSWORD:-treeloom_pass}@postgres:5432/${POSTGRES_DB:-treeloom}?sslmode=disable"
    ports: ["9187:9187"]
    depends_on: [postgres]
    restart: unless-stopped
  ```

### 3. Indexer logs (proposed)

`run.sh` runs the indexer as a foreground `uvicorn` process bound to `INDEXER_HOST`/`INDEXER_PORT` from `.env`. Redirect to a file so alloy can tail it:

```bash
exec uvicorn treeloom.indexer_service:app --host "$INDEXER_HOST" --port "$INDEXER_PORT" \
  2>&1 | tee -a /var/log/treeloom-indexer.log
```

Add a `local.file_match` block to the host alloy config tailing `/var/log/treeloom-indexer.log` and forwarding to the central collector (same exporter as containers). With the structlog OTel processor in `infrastructure/logging.py`, `otelTraceID` is already in every line — trace↔log linking works as soon as the lines reach Loki.

## Bring-up order

1. **Central stack**: extend collector config, add scrape file, restart `otel-collector` and `prometheus`.
2. **GPU host**: add `--metrics` to llama, run node_exporter + dcgm-exporter + alloy.
3. **k3s**: expose Neo4j and Milvus metrics via NodePort, add to scrape file.
4. **This repo**: edit `.env` (and `docker-compose.yml` if you adopt the proposed changes), then `./run.sh`.
5. **Indexer**: `./run.sh` (with the logging redirect in place).
6. **Grafana**: drop dashboard JSON into `./grafana/dashboards/` on the central host.

## Verification checklist

- `curl -s http://<obs-host>:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health}'` shows all new jobs `up`
- `up{job="tei"} == 1` for all four TEI ports
- `DCGM_FI_DEV_GPU_UTIL` series present for `host=gpu-host`
- `neo4j_*` and `milvus_*` series present
- After a test index run: in Tempo, `{service.name="treeloom-indexer"}` returns recent traces
- Click a span in Tempo → "Logs for this span" jumps to Loki with the matching `otelTraceID` log lines
- Indexer dashboard panels populate from the imported `treeloom-indexer-traces.json`

## Files in this repo

| Path | Role |
|---|---|
| `assets/observability/otel-collector-config.yaml` | Local-only collector config (no sampling — `always_on`; tail sampling belongs in the central collector) |
| `assets/observability/tempo-config.yaml` | Local-only Tempo |
| `assets/observability/grafana-datasources.yaml` | Local-only Grafana datasource provisioning |
| `assets/observability/grafana-dashboards.yaml` | Local-only Grafana dashboard provider |
| `assets/grafana/treeloom-indexer-traces.json` | Dashboard JSON — copy to the central host's `./grafana/dashboards/` for prod use |
| `src/treeloom/infrastructure/tracing.py` | OTel SDK init — controlled by `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `src/treeloom/infrastructure/logging.py` | structlog processor that injects `otelTraceID`/`otelSpanID` |

Span/attribute names use the stable `treeloom.` prefix — don't rename them, the Grafana dashboard depends on them.
