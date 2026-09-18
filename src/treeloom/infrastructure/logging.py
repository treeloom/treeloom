"""Structured logging for Treeloom — JSON-formatted fatal error logs."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog


def _add_trace_context(_logger, _method, event_dict):
    """structlog processor: stamp the current OTel span's IDs onto every
    log event so a log line jumps to its trace in Grafana Explore.

    Uses lowercase hex (no leading 0x) to match the trace IDs emitted by
    the OTel SDK and the OTLP exporter. Skips silently when the SDK
    isn't installed or there's no active span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            event_dict.setdefault("otelTraceID", f"{ctx.trace_id:032x}")
            event_dict.setdefault("otelSpanID", f"{ctx.span_id:016x}")
    except Exception:
        pass
    return event_dict


def configure_logging(json_file: str | None = None) -> None:
    """Set up structlog for JSON output. If json_file is provided, append to it."""
    if json_file:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        out = open(json_file, "a")
    else:
        out = sys.stderr

    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_context,
        structlog.dev.ConsoleRenderer()
        if out is sys.stderr and sys.stderr.isatty()
        else structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=out),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (logging.getLogger(__name__) across the adapters)
    # to the same sink. Without this, app warnings are INVISIBLE locally:
    # OTel's LoggingInstrumentor leaves a LoggingHandler on the root logger,
    # which satisfies "has handlers" and suppresses Python's last-resort
    # stderr printing — records ship to the OTLP collector (or nowhere) and
    # never reach the console/log file. Concretely: every "rerank failed"
    # warning during a reranker outage vanished while search silently
    # degraded to raw vector order.
    root = logging.getLogger()
    if not any(getattr(h, "_treeloom_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(out)
        handler._treeloom_handler = True
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        root.addHandler(handler)
    # Root stays at WARNING so third-party INFO (httpx per-request lines,
    # neo4j notifications) doesn't flood; treeloom's own hierarchy logs INFO.
    if root.level in (logging.NOTSET, 0) or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    logging.getLogger("treeloom").setLevel(logging.INFO)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or "treeloom")


# Pre-configured logger for the indexer
log = get_logger("treeloom.indexer")
