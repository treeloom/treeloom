FROM python:3.12-slim

WORKDIR /app

# ── Minimal MCP server dependencies ────────────────────────────────
# The MCP server is a pure API consumer that proxies all search/index
# requests to the host-based indexer.  Heavy deps (pymilvus, neo4j,
# chonkie, tree-sitter) live on the host — this container only needs
# httpx, mcp, fastapi, uvicorn, and pyyaml (for config module imports).
RUN pip install --no-cache-dir \
    httpx>=0.27 \
    mcp>=1.0 \
    fastapi>=0.115 \
    uvicorn[standard]>=0.32 \
    pyyaml>=6.0

COPY src/ src/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "treeloom.main:app", "--host", "0.0.0.0", "--port", "8000"]
