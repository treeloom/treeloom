"""Throughput-model coefficients, loaded from Postgres with hard-coded fallback.

Operators recalibrate via `python -m treeloom.preflight --calibrate <source_id>`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from treeloom.adapters.postgresql.connection import get_pool

logger = logging.getLogger(__name__)

# Hard-coded fallback used when Postgres is unavailable (CLI run without
# DATABASE_URL, tests, etc.). Same defaults as the seed migration.
_FALLBACK_GLOBAL: dict[str, float] = {
    "parse_s_per_mb":         1.0,
    "embed_s_per_chunk":      0.04,
    "llm_s_per_chunk":        2.0,
    "milvus_s_per_file":      0.05,
    "graph_s_per_file_const": 0.3,
    "graph_s_per_entity":     0.005,
    "neo4j_warmup_s":         5.0,
}

_FALLBACK_ENTITIES_PER_LOC: dict[str, float] = {
    "typescript": 0.50, "tsx": 0.50,
    "javascript": 0.30, "jsx": 0.30,
    "python":     0.20, "rust": 0.25, "go": 0.25,
    "java":       0.30, "csharp": 0.30, "cpp": 0.25, "c": 0.20,
}
_FALLBACK_ENTITIES_PER_LOC_DEFAULT = 0.20


@dataclass
class Coefficients:
    """Loaded coefficient set. Use `entities_per_loc(lang)` for per-language lookup."""
    global_: dict[str, float] = field(default_factory=dict)
    entities_per_loc_by_lang: dict[str, float] = field(default_factory=dict)
    entities_per_loc_default: float = _FALLBACK_ENTITIES_PER_LOC_DEFAULT

    def get(self, key: str) -> float:
        v = self.global_.get(key)
        if v is None:
            v = _FALLBACK_GLOBAL.get(key, 0.0)
        return v

    def entities_per_loc(self, language: str) -> float:
        return self.entities_per_loc_by_lang.get(language, self.entities_per_loc_default)


def fallback_coefficients() -> Coefficients:
    return Coefficients(
        global_=dict(_FALLBACK_GLOBAL),
        entities_per_loc_by_lang=dict(_FALLBACK_ENTITIES_PER_LOC),
        entities_per_loc_default=_FALLBACK_ENTITIES_PER_LOC_DEFAULT,
    )


async def load_coefficients() -> Coefficients:
    """Read from `preflight_coefficients`. Falls back if PG isn't available."""
    pool = await get_pool()
    if pool is None:
        logger.info("preflight: no PG pool, using fallback coefficients")
        return fallback_coefficients()

    coefs = fallback_coefficients()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT metric_key, language, value FROM preflight_coefficients"
            )
        for r in rows:
            key = r["metric_key"]
            lang = r["language"] or ""
            val = float(r["value"])
            if key == "entities_per_loc":
                if lang:
                    coefs.entities_per_loc_by_lang[lang] = val
                else:
                    coefs.entities_per_loc_default = val
            elif lang == "":
                coefs.global_[key] = val
            # per-language overrides for global metrics not implemented in v1
    except Exception:
        logger.exception("preflight: failed to read coefficients; using fallback")
        return fallback_coefficients()
    return coefs


async def write_coefficients(
    updates: dict[tuple[str, str], float],
    *,
    source_id: str | None = None,
) -> int:
    """Upsert coefficient rows from a calibration run.

    `updates` keyed by (metric_key, language). Returns rows written. Raises
    if PG isn't available.
    """
    pool = await get_pool()
    if pool is None:
        raise RuntimeError(
            "Calibration requires DATABASE_URL — preflight_coefficients lives in Postgres."
        )
    count = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for (key, lang), val in updates.items():
                await conn.execute(
                    """
                    INSERT INTO preflight_coefficients
                        (metric_key, language, value, calibrated_from_source_id, updated_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (metric_key, language)
                    DO UPDATE SET value = EXCLUDED.value,
                                  calibrated_from_source_id = EXCLUDED.calibrated_from_source_id,
                                  updated_at = NOW()
                    """,
                    key, lang, val, source_id,
                )
                count += 1
    return count
