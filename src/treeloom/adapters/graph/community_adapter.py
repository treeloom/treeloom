from collections import defaultdict
from pathlib import Path

import networkx as nx

from treeloom import embedder, graph_store


def _build_nx_graph(nodes: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        G.add_node(node["id"], **node)
    for node in nodes:
        for rel in node.get("_rels", []):
            target = rel.get("target_id")
            if target:
                G.add_edge(node["id"], target, type=rel["type"])
    return G


async def detect_communities(
    source_id: str, resolution: float = 1.0
) -> dict[str, str]:
    """Louvain community detection scoped to one source.

    Community ids are namespaced ``"<source_id>#<n>"`` so partitions from
    different sources never collide on the same Community node. Only the
    source's own entities get a `community_id` written (edge targets that
    Louvain pulls in — external modules, other sources — are left alone).
    """
    nodes = await graph_store.pull_graph(source_id=source_id)
    if not nodes:
        return {}
    G = _build_nx_graph(nodes)

    import community as community_louvain

    partition = community_louvain.best_partition(G, resolution=resolution)
    source_node_ids = {n["id"] for n in nodes}
    namespaced = {
        nid: f"{source_id}#{cid}"
        for nid, cid in partition.items()
        if nid in source_node_ids
    }
    await graph_store.set_community_ids_batch(namespaced)
    return namespaced


def summarize_community(entities: list[dict]) -> str:
    names = []
    for e in entities:
        name = e.get("name", "?")
        if e.get("type") == "Module":
            name = Path(name).name
        names.append(f"{e.get('type', '?')}:{name}")
    name_list = ", ".join(names[:10])
    entity_count = len(entities)
    file_count = len({e.get("file_path", "") for e in entities})
    return f"{entity_count} entities across {file_count} files: {name_list}"


async def get_community_summaries(community_ids: set[str]) -> dict[str, str]:
    if not community_ids:
        return {}
    return await graph_store.get_community_summaries_map(list(community_ids))


async def get_all_community_summaries() -> dict[str, str]:
    return await graph_store.get_all_community_summaries()


async def load_community_embeddings() -> dict[str, list[float]]:
    return await graph_store.load_community_embeddings()


async def run_post_index_signals(source_id: str) -> int:
    """Compute and persist graph signals for one source.

    Pulls the source subgraph once, runs Louvain + degree centrality in
    memory, then persists everything with a handful of batched UNWIND
    writes (community ids, community summaries + embeddings, centrality).
    Per-source and batched, so this runs in seconds — not the hours the
    old whole-graph, one-write-per-node version took (it never finished as
    a fire-and-forget task, leaving the graph with zero signals).

    Returns the number of communities detected.
    """
    nodes = await graph_store.pull_graph(source_id=source_id)
    if not nodes:
        return 0
    G = _build_nx_graph(nodes)
    source_node_ids = {n["id"] for n in nodes}
    nodes_by_id = {n["id"]: n for n in nodes}

    import community as community_louvain

    # ── communities (Louvain), namespaced per source ──
    partition = community_louvain.best_partition(G)
    entity_community = {
        nid: f"{source_id}#{cid}"
        for nid, cid in partition.items()
        if nid in source_node_ids
    }
    await graph_store.delete_source_communities(source_id)
    await graph_store.set_community_ids_batch(entity_community)

    # ── per-community summaries + embeddings ──
    groups: dict[str, list[dict]] = defaultdict(list)
    for nid, cid in entity_community.items():
        groups[cid].append(nodes_by_id[nid])
    summaries = {
        cid: summarize_community(ents) for cid, ents in groups.items() if ents
    }
    await graph_store.upsert_community_summaries_batch(summaries)
    if summaries:
        cids = list(summaries)
        embs = await embedder.embed([summaries[c] for c in cids])
        await graph_store.set_community_embeddings_batch(dict(zip(cids, embs)))

    # ── degree centrality on the source subgraph ──
    deg = nx.degree_centrality(G)
    centralities = {nid: deg[nid] for nid in source_node_ids if nid in deg}
    await graph_store.set_centralities_batch(centralities)

    # Drop the in-process community-embedding cache so this worker picks up
    # the embeddings just written. (The separate MCP search process has its
    # own cache and still needs a restart — see the lazy-cache note in
    # CLAUDE.md.) Lazy import avoids a community<->retrieval import cycle;
    # the cache lives in the store-agnostic retrieval module, NOT the milvus
    # adapter (importing that here would require Milvus env in simple mode).
    try:
        from treeloom.application.retrieval import invalidate_graph_caches

        invalidate_graph_caches()
    except Exception:
        pass

    return len(summaries)


async def build_all_sources() -> dict[str, int]:
    """Backfill graph signals across every indexed source.

    Used by `POST /build-community` to populate sources indexed before the
    per-source pipeline existed. Returns {source_id: community_count}.
    """
    sources = await graph_store.list_sources()
    result: dict[str, int] = {}
    for s in sources:
        sid = s.get("id")
        if not sid:
            continue
        result[sid] = await run_post_index_signals(sid)
    return result