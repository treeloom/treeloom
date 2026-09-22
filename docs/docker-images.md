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

## Versioning

Treeloom uses CalVer: a release is named for the date it was cut, in the
form `YYYY.M.D`, for example `2026.9.22`.

- No zero padding (`2026.9.2`, never `2026.09.02`). PEP 440 normalises
  `09` to `9`, so padding would make the wheel version and the git tag
  disagree.
- A second release on the same day appends a counter: `2026.9.22.1`,
  `2026.9.22.2`.
- The git tag is the version with a `v` prefix: `v2026.9.22`.
- The tag must equal `version` in `pyproject.toml`. The publish workflow's
  `check-tag` job refuses a tag that is not CalVer-shaped or that does not
  match, before any image is built.

## Tags

| Trigger | Tags pushed |
|---|---|
| Git tag `vYYYY.M.D[.N]` | `YYYY.M.D[.N]`, `YYYY.M` (rolling month), `latest` |
| Manual workflow run (any branch) | `edge`, `sha-<short commit>` |
| Pull request touching a Dockerfile | build only, nothing pushed; amd64 only, reranker skipped, superseded by a newer push to the PR |

`latest` always tracks the most recent release tag, never `main`. The
rolling month tag (`2026.9`) moves to the newest release within that month.

## Using the published images

Both compose files declare an `image:` next to each `build:`, so a checkout can
either build locally (the default, what `run.sh` does) or pull the published
images:

```bash
# Pull the release images instead of building
docker compose pull ui mcp-server qwen3-reranker
docker compose up -d ui

# Pin a specific release (or a month: TREELOOM_IMAGE_TAG=2026.9)
TREELOOM_IMAGE_TAG=2026.9.22 docker compose pull ui
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

1. Set `version` in `pyproject.toml` to today's date in CalVer form (see
   "Versioning" above) and merge to `main`. The unit suite checks the shape.
2. Tag and push the tag — the same string with a `v` prefix:
   ```bash
   git tag v2026.9.22 && git push origin v2026.9.22
   ```
3. Watch the "Docker images" workflow. `check-tag` runs first and fails the
   run if the tag and pyproject disagree; then four build jobs run in
   parallel, one per image. The indexer and reranker jobs take the longest
   (the reranker pulls a ~6 GB CUDA base).
4. Verify on Docker Hub, or from any machine:
   ```bash
   docker pull treeloom/ui:2026.9.22 && docker pull treeloom/indexer:2026.9.22
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
