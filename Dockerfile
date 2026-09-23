FROM python:3.14-slim

WORKDIR /app

# ── Minimal MCP server dependencies ────────────────────────────────
# The MCP server is a pure API consumer that proxies all search/index
# requests to the host-based indexer.  Heavy deps (pymilvus, neo4j,
# chonkie, tree-sitter) live on the host — this container only needs
# httpx, mcp, fastapi, uvicorn, and pyyaml (for config module imports).
#
# The specifiers MUST stay quoted: an unquoted `mcp>=1.0` is parsed by
# `sh -c` as a redirection to a file named `=1.0`, which silently installs
# an unpinned `mcp`. The mcp pin mirrors pyproject.toml (`mcp>=1.27,<2`);
# a rebuild once jumped mcp 1.6 → 1.27 and broke the SSE transport.
RUN pip install --no-cache-dir \
    'httpx>=0.27' \
    'mcp>=1.27,<2' \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.32' \
    'pyyaml>=6.0'

COPY src/ src/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "treeloom.main:app", "--host", "0.0.0.0", "--port", "8000"]
