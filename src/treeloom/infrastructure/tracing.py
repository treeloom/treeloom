"""OpenTelemetry tracing for the Treeloom indexer.

Standard OTel env vars are honored:
    OTEL_EXPORTER_OTLP_ENDPOINT (default: http://localhost:4317)
    OTEL_TRACES_SAMPLER          (default: parentbased_traceidratio)
    OTEL_TRACES_SAMPLER_ARG      (default: 0.1)
    OTEL_SERVICE_NAME            (defaults to argument to init_tracer)

`init_tracer()` is idempotent — calling it twice does not re-install the
provider, so test imports and uvicorn's reload behavior don't double-wire
instrumentation.

Auto-instrumentation we enable:
    - FastAPI: request spans on the indexer service.
    - HTTPX: outbound calls to TEI, LLM API, Milvus REST, etc.
    - AsyncPG: every queried statement.
    - logging: stdlib logging records carry otelTraceID / otelSpanID.

Manual spans live inside the indexer hot path and the adapter wrappers.
The default tracer name used everywhere is `treeloom.indexer`; callers
that want a sub-namespace should ask for one explicitly via
`trace.get_tracer("treeloom.<adapter>")`.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialized: bool = False


def init_tracer(service_name: str = "treeloom-indexer", app=None) -> None:
    """Build the global TracerProvider and install auto-instrumentation.

    `app` is the FastAPI instance; when provided, FastAPI auto-instrumentation
    is attached to it. Safe to omit (e.g. for non-HTTP callers like the CLI).
    """
    global _initialized
    if _initialized:
        if app is not None:
            _instrument_fastapi(app)
        return

    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        logger.info("OTel disabled via OTEL_SDK_DISABLED — skipping tracer init")
        _initialized = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as exc:
        logger.warning("OTel SDK not installed (%s); tracing disabled", exc)
        _initialized = True
        return

    resource_attrs = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
    }
    version = os.environ.get("TREELOOM_VERSION", "")
    if version:
        resource_attrs["service.version"] = version
    resource = Resource.create(resource_attrs)

    provider = TracerProvider(resource=resource)
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _instrument_libraries(app)
    _initialized = True
    logger.info("OTel tracer initialized: service=%s endpoint=%s", service_name, endpoint)


def _instrument_libraries(app) -> None:
    if app is not None:
        _instrument_fastapi(app)
    _safe_instrument("HTTPXClient", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor")
    _safe_instrument("AsyncPG", "opentelemetry.instrumentation.asyncpg", "AsyncPGInstrumentor")
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception as exc:
        logger.warning("LoggingInstrumentor not installed (%s)", exc)


def _instrument_fastapi(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("FastAPIInstrumentor not installed (%s)", exc)


def _safe_instrument(label: str, module: str, attr: str) -> None:
    try:
        mod = __import__(module, fromlist=[attr])
        instrumentor = getattr(mod, attr)()
        instrumentor.instrument()
    except Exception as exc:
        logger.warning("%s instrumentation skipped (%s)", label, exc)


def get_tracer(name: str = "treeloom.indexer"):
    """Return a tracer for the named scope.

    Returns a no-op tracer when the OTel SDK isn't installed.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return _NoopTracer()
    return trace.get_tracer(name)


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_attribute(self, *a, **kw):
        pass

    def set_attributes(self, *a, **kw):
        pass

    def set_status(self, *a, **kw):
        pass

    def record_exception(self, *a, **kw):
        pass

    def add_event(self, *a, **kw):
        pass


class _NoopTracer:
    def start_as_current_span(self, *a, **kw):
        return _NoopSpan()

    def start_span(self, *a, **kw):
        return _NoopSpan()
