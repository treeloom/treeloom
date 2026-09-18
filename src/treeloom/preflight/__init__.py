"""Preflight repo analysis — predict indexing time + surface tuning hints."""
from treeloom.preflight.scanner import FileStat, walk_repo  # noqa: F401
from treeloom.preflight.model import (  # noqa: F401
    Coefficients,
    FileEstimate,
    TotalEstimate,
    estimate_total,
)
from treeloom.preflight.recommender import (  # noqa: F401
    PreflightReport,
    Warning as PreflightWarning,
    build_report,
)
