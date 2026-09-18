"""Recommendation rules + report assembly."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from treeloom.preflight.model import FileEstimate, TotalEstimate
from treeloom.preflight.scanner import FileStat, NOISY_PATTERNS

Severity = Literal["low", "medium", "high"]
Verdict = Literal["green", "yellow", "red"]


@dataclass
class Warning:
    code: str
    severity: Severity
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class Recommendations:
    env: dict[str, str] = field(default_factory=dict)
    skip_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"env": dict(self.env), "skip_patterns": list(self.skip_patterns)}


@dataclass
class PreflightReport:
    path: str
    files_summary: dict
    estimates: dict
    warnings: list[Warning]
    recommendations: Recommendations
    verdict: Verdict

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "files": self.files_summary,
            "estimates": self.estimates,
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendations": self.recommendations.to_dict(),
            "verdict": self.verdict,
        }


# Severity thresholds (seconds)
_RED_WALL_S = 24 * 3600
_YELLOW_WALL_S = 6 * 3600


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def build_report(
    path: str,
    stats: list[FileStat],
    unsupported_count: int,
    per_file: list[FileEstimate],
    total: TotalEstimate,
    *,
    feature_skip_patterns: bool | None = None,
) -> PreflightReport:
    if feature_skip_patterns is None:
        feature_skip_patterns = os.environ.get("TREELOOM_FEATURE_SKIP_PATTERNS", "") == "1"

    warnings: list[Warning] = []
    rec = Recommendations()

    if not stats:
        warnings.append(Warning(
            code="NO_SUPPORTED_FILES", severity="high",
            message=f"No supported files found at {path!r} — nothing to index.",
        ))
        return PreflightReport(
            path=path,
            files_summary={
                "total_supported": 0, "total_unsupported": unsupported_count,
                "by_language": {}, "size_total_bytes": 0,
                "max_file_bytes": 0, "long_tail_count_100kb": 0,
                "hot_files": [],
            },
            estimates={
                "summary_on": {"wall_time_s": 0, "files_per_min": 0},
                "summary_off": {"wall_time_s": 0, "files_per_min": 0},
            },
            warnings=warnings,
            recommendations=rec,
            verdict="red",
        )

    # --- file-shape facts ---
    by_language = Counter(s.language for s in stats)
    size_total = sum(s.size_bytes for s in stats)
    max_size = max(s.size_bytes for s in stats)
    long_tail = sum(1 for s in stats if s.size_bytes > 100 * 1024)

    # Hot files = sorted by estimated summary_on time, top 10
    by_est = sorted(per_file, key=lambda e: e.seconds_summary_on, reverse=True)[:10]
    by_path = {s.rel_path: s for s in stats}
    hot_files = []
    for est in by_est:
        s = by_path.get(est.rel_path)
        if s is None:
            continue
        hot_files.append({
            "path": s.rel_path,
            "size_bytes": s.size_bytes,
            "loc": s.loc,
            "language": s.language,
            "estimated_seconds_summary_on": round(est.seconds_summary_on, 1),
        })

    # --- rules ---
    # Two-pass strategy: when even summary_off (chunks + graph, no LLM)
    # is over the yellow threshold, recommend splitting indexing into
    # skip_graph=true followed by /index-graph so search becomes
    # usable in a fraction of the wall time.
    if total.wall_seconds_summary_off > _YELLOW_WALL_S:
        warnings.append(Warning(
            code="STRATEGY_TWO_PASS", severity="high",
            message=(
                f"Even chunks+graph alone takes {_format_duration(total.wall_seconds_summary_off)}. "
                f"Split into two passes: 1) /index-repo with skip_graph=true, "
                f"2) /index-graph with the resulting source_id during off-hours."
            ),
        ))
        rec.env["strategy"] = "two_pass"

    if total.wall_seconds_summary_on > _RED_WALL_S and total.wall_seconds_summary_off < _YELLOW_WALL_S:
        warnings.append(Warning(
            code="SUMMARY_VECTOR_BLOWS_BUDGET", severity="high",
            message=(f"Estimated {_format_duration(total.wall_seconds_summary_on)} with "
                     f"USE_SUMMARY_VECTOR=1; setting it to 0 drops the estimate to "
                     f"{_format_duration(total.wall_seconds_summary_off)}. Strongly recommended."),
        ))
        rec.env["USE_SUMMARY_VECTOR"] = "0"
    elif total.wall_seconds_summary_off > _RED_WALL_S:
        warnings.append(Warning(
            code="ESTIMATED_TIME_OVER_24H", severity="high",
            message=(f"Estimated {_format_duration(total.wall_seconds_summary_off)} even without "
                     f"summary vectors. Consider splitting the source or reducing scope."),
        ))
    elif total.wall_seconds_summary_on > _YELLOW_WALL_S:
        warnings.append(Warning(
            code="LONG_INDEX_TIME", severity="medium",
            message=(f"Estimated {_format_duration(total.wall_seconds_summary_on)} with summaries; "
                     f"{_format_duration(total.wall_seconds_summary_off)} without."),
        ))

    if max_size > 1_000_000:
        n_big = sum(1 for s in stats if s.size_bytes > 1_000_000)
        warnings.append(Warning(
            code="HOT_FILES", severity="medium",
            message=f"{n_big} file(s) > 1 MB — these dominate runtime. See `hot_files` in the report.",
        ))

    if long_tail > 20:
        warnings.append(Warning(
            code="LONG_TAIL", severity="medium",
            message=f"{long_tail} files > 100 KB — long tail will dominate even with concurrency.",
        ))

    # Tree-sitter crash risk: files with extremely long lines (>10KB)
    # can cause UnboundLocalError in the parser (observed on TypeScript
    # test baselines with 16KB+ single lines). Falls back to character
    # chunking, but flag them so operators can pre-emptively skip.
    _TS_CRASH_LINE_BYTES = int(os.environ.get("PREFLIGHT_TS_CRASH_LINE_BYTES", "10240"))
    tree_sitter_risks = [
        s for s in stats
        if s.max_line_length > _TS_CRASH_LINE_BYTES
    ]
    if tree_sitter_risks:
        risk_paths = sorted(s.rel_path for s in tree_sitter_risks[:5])
        tail = f" (+{len(tree_sitter_risks) - 5} more)" if len(tree_sitter_risks) > 5 else ""
        warnings.append(Warning(
            code="TREE_SITTER_CRASH_RISK", severity="medium",
            message=(
                f"{len(tree_sitter_risks)} file(s) have lines >{_TS_CRASH_LINE_BYTES//1024}KB — "
                f"tree-sitter may crash on these (falls back to character chunking): "
                f"{', '.join(risk_paths)}{tail}"
            ),
        ))

    # Noisy-file detection: suggest skip_patterns
    suggested_patterns: list[str] = []
    for pat in NOISY_PATTERNS:
        if any(s.rel_path.endswith(pat.lstrip("*")) or s.rel_path == pat for s in stats):
            suggested_patterns.append(pat)
    # Also suggest skipping .d.ts LIBRARY files (lib.*.d.ts) — generated
    if any(s.rel_path.endswith(".d.ts") and ("/lib/lib." in s.rel_path or "/lib.dom" in s.rel_path) for s in stats):
        suggested_patterns.append("**/lib.*.d.ts")
    if suggested_patterns:
        warnings.append(Warning(
            code="NOISY_FILES_DETECTED", severity="medium",
            message=f"Generated/bundle files detected. Suggest skipping: {', '.join(suggested_patterns)}.",
        ))
        if feature_skip_patterns:
            rec.skip_patterns = suggested_patterns

    if total.files > 20_000:
        warnings.append(Warning(
            code="LARGE_REPO", severity="low",
            message=f"{total.files} files — plan to run overnight; resume works on crash.",
        ))

    if by_language and max(by_language.values()) / total.files > 0.90:
        dominant = max(by_language, key=by_language.get)
        warnings.append(Warning(
            code="SINGLE_LANGUAGE_DOMINANT", severity="low",
            message=f"{by_language[dominant]}/{total.files} files are {dominant}.",
        ))

    # --- env defaults always recommended ---
    rec.env.setdefault("INDEX_FILE_CONCURRENCY", str(total.file_concurrency))

    # --- verdict ---
    has_high = any(w.severity == "high" for w in warnings)
    has_medium = any(w.severity == "medium" for w in warnings)
    if has_high or total.wall_seconds_summary_off > _RED_WALL_S:
        verdict: Verdict = "red"
    elif has_medium or total.wall_seconds_summary_on > _YELLOW_WALL_S:
        verdict = "yellow"
    else:
        verdict = "green"

    return PreflightReport(
        path=path,
        files_summary={
            "total_supported": len(stats),
            "total_unsupported": unsupported_count,
            "by_language": dict(by_language),
            "size_total_bytes": size_total,
            "max_file_bytes": max_size,
            "long_tail_count_100kb": long_tail,
            "hot_files": hot_files,
        },
        estimates={
            "summary_on": {
                "wall_time_s": int(total.wall_seconds_summary_on),
                "files_per_min": round(total.files_per_min_summary_on, 2),
            },
            "summary_off": {
                "wall_time_s": int(total.wall_seconds_summary_off),
                "files_per_min": round(total.files_per_min_summary_off, 2),
            },
        },
        warnings=warnings,
        recommendations=rec,
        verdict=verdict,
    )
