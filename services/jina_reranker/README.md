# jina-reranker service

A standalone GPU reranker that speaks TEI's `/rerank` wire format, for reranker
models TEI can't host. Built specifically for
`jinaai/jina-reranker-v2-base-multilingual`, which TEI refuses (custom modeling
code, no standard `model_type` — see huggingface/text-embeddings-inference issue #571).

## Contract (drop-in for TEI)

```
POST /rerank   {"query": str, "texts": [str, ...], "raw_scores": bool}
   ->          [{"index": int, "score": float}, ...]   # one per input text
GET  /info     {"model_id": ..., "max_input_length": ...}
GET  /health   {"status": "ok"|"loading", "model_id": ...}
```

`retriever.rerank` calls exactly this, so no treeloom code changes — just point
`RERANKER_URL` here.

## Run

```bash
# build + start (downloads the model on first boot, ~minutes)
docker compose --profile jina up -d --build jina-reranker

# verify it loaded the right model
curl -s localhost:8085/info        # -> {"model_id":"jinaai/jina-reranker-v2-base-multilingual",...}
docker logs --tail 20 treeloom-jina-reranker-1   # ends in "reranker ready"
```

Then point treeloom's reranker at it and restart the indexer:

```bash
# host indexer (.env)
RERANKER_URL=http://localhost:8085
# in-Docker indexer
DOCKER_RERANKER_URL=http://jina-reranker:80
```

To switch back to TEI/bge, set `RERANKER_URL=http://localhost:8081` and restart
the indexer. (No need to stop this service — leave it for A/B.)

## Config (env, all optional)

| var | default | meaning |
|---|---|---|
| `JINA_RERANKER_PORT` | `8085` | host port |
| `JINA_RERANKER_MODEL` / `MODEL_ID` | `jinaai/jina-reranker-v2-base-multilingual` | model to serve |
| `JINA_RERANKER_GPU` | `0` | GPU device id |
| `RERANK_MAX_LENGTH` | `1024` | max query+text tokens (jina v2 limit) |

## Notes

- Scores are the model's `compute_score` (sigmoid, `[0,1]`). treeloom only sorts
  by score and then min-max normalizes in `graph_rescore` (`RERANK_NORMALIZE`),
  so bounded scores are fine; the `raw_scores` request field is ignored.
- `flash-attn` is **not** installed (slow/fragile build); jina v2 falls back to
  eager attention. If logs show a `flash_attn` ImportError, add
  `RUN pip install flash-attn --no-build-isolation` to the Dockerfile.
- First boot downloads the model to the `jina-reranker-data` volume; later boots
  are fast.
