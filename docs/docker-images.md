# Docker images

Treeloom publishes its container images to Docker Hub under the `treeloom`
namespace. This page covers what is published, how to consume the images, and
how a maintainer cuts a release.

## Published images

| Image | Built from | Platforms | Role |
|---|---|---|---|
| `treeloom/indexer` | `Dockerfile.indexer` | `linux/amd64` | The indexer FastAPI service (`treeloom.indexer_service:app`, port 8001). Ships the full dependency set and a pre-downloaded tiktoken cache so it runs without egress. |
| `treeloom/mcp-server` | `Dockerfile` | `linux/amd64`, `linux/arm64` | The deprecated HTTP+SSE MCP transport (`treeloom.main:app`, port 8000). Pure HTTP proxy to an indexer; opt-in via the `http-mcp` compose profile. The default stdio transport (`treeloom-mcp`) needs no container. |
| `treeloom/ui` | `ui/Dockerfile` | `linux/amd64`, `linux/arm64` | The operator SPA served by nginx on port 80. The API base is injected at container start from `TREELOOM_API_BASE`. |
| `treeloom/qwen3-reranker` | `services/qwen3_reranker/Dockerfile` | `linux/amd64` | GPU reranker service for `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` (TEI cannot host Qwen3 rerankers). CUDA 12.1 base; needs the NVIDIA container runtime. |

Not published: `services/jina_reranker` (an experimental alternative reranker,
same CUDA base) and the embedding/reranking TEI services, Postgres, Neo4j,
Milvus and the observability stack, which are upstream images referenced
directly by `docker-compose.yml`.

The indexer image is `amd64` only because its dependency set (pymilvus,
tree-sitter grammars, chonkie) is slow to build under QEMU emulation. The
indexer is normally run on the host anyway (it must read arbitrary user paths;
see the Quick Start), so the image matters mostly for server deployments.

## Tags

| Trigger | Tags pushed |
|---|---|
| Git tag `vX.Y.Z` | `X.Y.Z`, `X.Y`, `latest` |
| Manual workflow run (any branch) | `edge`, `sha-<short commit>` |
| Pull request touching a Dockerfile | build only, nothing pushed |

`latest` always tracks the most recent release tag, never `main`.

## Using the published images

Both compose files declare an `image:` next to each `build:`, so a checkout can
either build locally (the default, what `run.sh` does) or pull the published
images:

```bash
# Pull the release images instead of building
docker compose pull ui mcp-server qwen3-reranker
docker compose up -d ui

# Pin a specific release
TREELOOM_IMAGE_TAG=0.3.0 docker compose pull ui
```

`TREELOOM_IMAGE_NAMESPACE` (default `treeloom`) and `TREELOOM_IMAGE_TAG`
(default `latest`) select the registry namespace and tag for every Treeloom
image at once. Running `docker compose build` still works and tags the local
build with the same name, which is why a subsequent `up` uses whichever you
did last; pass `--pull` to `up` to force the registry copy.

Running the indexer from its image outside compose:

```bash
docker run --rm -p 8001:8001 --env-file .env \
  -v /path/to/repos:/data/repos:ro \
  treeloom/indexer:latest
```

Remember the container cannot see host paths unless they are mounted, and
service URLs in `.env` that point at `localhost` must be rewritten to
addresses reachable from inside the container.

## Releasing (maintainers)

The publish workflow is `.github/workflows/docker-publish.yml`. It needs two
repository secrets and accepts one optional variable, all set under the
GitHub repository's Settings → Secrets and variables → Actions:

| Name | Kind | Value |
|---|---|---|
| `DOCKERHUB_USERNAME` | secret | The Docker Hub login that owns (or is a member of) the namespace |
| `DOCKERHUB_TOKEN` | secret | A Docker Hub access token with Read & Write scope (Account settings → Personal access tokens). Never a password. |
| `DOCKERHUB_NAMESPACE` | variable | Optional. Defaults to `treeloom`; set it if the account name differs. |

Equivalent CLI, run by whoever holds the token:

```bash
gh secret set DOCKERHUB_USERNAME --repo treeloom/treeloom
gh secret set DOCKERHUB_TOKEN --repo treeloom/treeloom
gh variable set DOCKERHUB_NAMESPACE --repo treeloom/treeloom --body treeloom
```

Cutting a release:

1. Bump `version` in `pyproject.toml` (the image tag is taken from the git
   tag, but the two must agree) and merge to `main`.
2. Tag and push the tag:
   ```bash
   git tag v0.3.0 && git push origin v0.3.0
   ```
3. Watch the "Docker images" workflow. Four jobs run in parallel, one per
   image; the indexer and reranker jobs take the longest (the reranker pulls a
   ~6 GB CUDA base).
4. Verify on Docker Hub, or from any machine:
   ```bash
   docker pull treeloom/ui:0.3.0 && docker pull treeloom/indexer:0.3.0
   ```

A pre-release smoke build without touching `latest`: run the workflow manually
from the Actions tab (or `gh workflow run docker-publish.yml --ref <branch>`).
That pushes `edge` and `sha-<commit>` only.

Docker Hub repositories are created automatically on first push. To make them
public they must be public on Docker Hub; a free account defaults new
repositories to public.

## Building locally

The published names are the same ones `docker compose build` produces, so a
local build can be pushed by hand when needed:

```bash
docker login
docker compose build indexer ui mcp-server
docker push treeloom/ui:latest
```

Prefer the workflow: it applies the OCI labels, builds multi-arch manifests
for the ui and MCP images, and never has a developer's `.env` in scope.
