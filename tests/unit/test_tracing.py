"""Smoke tests for the OTel tracing init module.

Most of the tracing surface is verified end-to-end via Tempo + Grafana.
The unit tests here just confirm that `init_tracer()` is idempotent, respects
`OTEL_SDK_DISABLED`, and that `get_tracer()` always returns something
callable so tracer.start_as_current_span(...) sites never KeyError out.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_module():
    """Reload the tracing module between tests so `_initialized` resets."""
    import treeloom.infrastructure.tracing as tracing_mod
    importlib.reload(tracing_mod)
    yield
    importlib.reload(tracing_mod)


def test_init_tracer_is_idempotent(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from treeloom.infrastructure import tracing

    tracing.init_tracer("test-service")
    assert tracing._initialized is True
    # A second call must not raise (and must not re-init).
    tracing.init_tracer("test-service")
    assert tracing._initialized is True


def test_get_tracer_returns_a_context_manager(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from treeloom.infrastructure import tracing

    tracing.init_tracer("test-service")
    tracer = tracing.get_tracer("treeloom.test")
    # The plan's callsites assume `with tracer.start_as_current_span(...)` works.
    # The no-op tracer fallback supports it too.
    with tracer.start_as_current_span("smoke") as span:
        span.set_attribute("treeloom.test", "ok")
        span.add_event("hi")


def test_sdk_disabled_skips_provider(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from treeloom.infrastructure import tracing

    # Should not raise even though no exporter endpoint is reachable.
    tracing.init_tracer("test-service")
