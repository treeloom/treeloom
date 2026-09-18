"""Application layer search service — orchestrates retriever, community, graph_store."""
import os

from treeloom import embedder, graph_store, llm

USE_HYBRID = os.environ.get("USE_HYBRID", "1") == "1"
USE_HYDE = os.environ.get("USE_HYDE", "1") == "1"
USE_SUMMARY_VECTOR = os.environ.get("USE_SUMMARY_VECTOR", "1") == "1"
# Graph RESCORING defaults ON (kept on because it helps rank-1
# under the production reranker; off is a Cohere-class opt-in) — kept aligned
# with application/retrieval.py. NB: this module is legacy/unwired; the live
# /search path goes through application/retrieval.graph_search.
USE_GRAPH_SCORING = os.environ.get("USE_GRAPH_SCORING", "1") == "1"


async def search_graphs(query_text, query_embedding, top_k=10, language=None, source_id=None):
    """Application-layer search — delegates to vector_store + graph_store + community."""
    from treeloom import retriever

    vec_results = retriever.search(
        query_embedding,
        top_k=retriever.MILVUS_TOP_K_PRE_RERANK,
        language=language,
        source_id=source_id,
    )
    if not vec_results:
        return {"chunks": [], "neighbors": [], "community_summaries": {}}

    file_paths = set()
    for r in vec_results:
        fp = r["entity"].get("file_path", "")
        if fp:
            file_paths.add(fp)

    entities_by_file = {}
    matched_entity_ids = set()
    neighbor_entities = []
    for fp in file_paths:
        entities = await graph_store.get_entities_by_file(fp, source_id=source_id)
        entities_by_file[fp] = entities
        for e in entities:
            matched_entity_ids.add(e["id"])
            traversed = await graph_store.traverse(e["id"], depth=1, max_nodes=30)
            for tn in traversed:
                nbs = tn.pop("_neighbors", [])
                for nb in nbs:
                    if nb["id"] not in matched_entity_ids:
                        neighbor_entities.append(nb)
                        matched_entity_ids.add(nb["id"])

    community_ids = await graph_store.get_community_ids(list(matched_entity_ids))
    from treeloom.adapters.graph.community_adapter import get_community_summaries
    summaries = await get_community_summaries(community_ids)

    reranked = await retriever.rerank(query_text, vec_results, top_k=top_k)
    if USE_GRAPH_SCORING and reranked:
        community_embs = await retriever._get_community_embeddings()
        reranked = await retriever.graph_rescore(
            reranked, query_embedding, query_text, entities_by_file, community_embs
        )

    return {
        "chunks": reranked,
        "neighbors": neighbor_entities[:20],
        "community_summaries": summaries,
    }
