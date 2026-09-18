"""Backward-compatible re-export of config module."""
from treeloom.infrastructure.config import (
    ConfigError,
    require_env,
    require_int_env,
)  # noqa: F401
