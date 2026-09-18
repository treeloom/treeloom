"""Throughput model — predict per-file and total indexing wall time."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from treeloom.preflight.coefficients import Coefficients
from treeloom.preflight.scanner import FileStat

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))


def _estimate_tokens(text_or_bytes: int) -> int:
    """Cheap token estimate: bytes // 3. Good enough at the +/-20% level
    that the model already targets; matches the per-chunk heuristic in
    embedding_proxy when the real tokenizer isn't usable."""
    return max(text_or_bytes // 3, 1)


def _chunks_for(stat: FileStat) -> int:
    tokens = _estimate_tokens(stat.size_bytes)
    return max(math.ceil(tokens / CHUNK_SIZE), 1)


def _entities_for(stat: FileStat, coefs: Coefficients) -> int:
    return max(int(stat.loc * coefs.entities_per_loc(stat.language)), 1)


@dataclass
class FileEstimate:
    rel_path: str
    size_bytes: int
    chunks: int
    entities: int
    seconds_summary_on: float
    seconds_summary_off: float

    @property
    def language(self) -> str:
        # Not stored; recover from FileStat at the call-site if needed
        return ""


@dataclass
class TotalEstimate:
    wall_seconds_summary_on: float
    wall_seconds_summary_off: float
    files: int
    chunks: int
    file_concurrency: int

    @property
    def files_per_min_summary_on(self) -> float:
        return self.files / (self.wall_seconds_summary_on / 60) if self.wall_seconds_summary_on > 0 else 0

    @property
    def files_per_min_summary_off(self) -> float:
        return self.files / (self.wall_seconds_summary_off / 60) if self.wall_seconds_summary_off > 0 else 0


def estimate_file(stat: FileStat, coefs: Coefficients) -> FileEstimate:
    chunks = _chunks_for(stat)
    entities = _entities_for(stat, coefs)
    size_mb = stat.size_bytes / (1024 * 1024)

    parse_s = coefs.get("parse_s_per_mb") * size_mb
    embed_chunks_s = chunks * coefs.get("embed_s_per_chunk")
    llm_s = chunks * coefs.get("llm_s_per_chunk")
    embed_summaries_s = chunks * coefs.get("embed_s_per_chunk")
    milvus_s = coefs.get("milvus_s_per_file")
    graph_s = coefs.get("graph_s_per_file_const") + entities * coefs.get("graph_s_per_entity")

    summary_off = parse_s + embed_chunks_s + milvus_s + graph_s
    summary_on = summary_off + llm_s + embed_summaries_s
    return FileEstimate(
        rel_path=stat.rel_path,
        size_bytes=stat.size_bytes,
        chunks=chunks,
        entities=entities,
        seconds_summary_on=summary_on,
        seconds_summary_off=summary_off,
    )


def estimate_total(
    files: list[FileStat],
    coefs: Coefficients,
    *,
    file_concurrency: int = 8,
) -> tuple[list[FileEstimate], TotalEstimate]:
    if file_concurrency < 1:
        file_concurrency = 1
    per_file = [estimate_file(f, coefs) for f in files]
    total_on = sum(f.seconds_summary_on for f in per_file) / file_concurrency
    total_off = sum(f.seconds_summary_off for f in per_file) / file_concurrency
    warmup = coefs.get("neo4j_warmup_s")
    total_chunks = sum(f.chunks for f in per_file)
    return per_file, TotalEstimate(
        wall_seconds_summary_on=total_on + warmup,
        wall_seconds_summary_off=total_off + warmup,
        files=len(per_file),
        chunks=total_chunks,
        file_concurrency=file_concurrency,
    )
