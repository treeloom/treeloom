"""The indexer app wires startup/shutdown through the ASGI lifespan protocol.

Starlette 1.6 removed ``Starlette.add_event_handler`` / ``on_event`` (they had
been deprecated since 0.26 in favour of ``lifespan=``). ``pyproject.toml``
pins ``fastapi>=0.115`` with no upper bound, so a fresh install — a rebuilt
Docker image, a clean CI runner — resolves the newest Starlette and the
indexer would fail at *import time* with
``AttributeError: 'FastAPI' object has no attribute 'add_event_handler'``
while a developer venv on an older pin keeps working. These tests pin the
lifespan wiring so that regression can't come back silently.
"""
import ast
import asyncio
import inspect
from unittest.mock import AsyncMock, patch

from treeloom.application import indexer_service
from treeloom.application import lifecycle


def _calls_named(module, *attrs: str) -> list[str]:
    """Return every ``<expr>.<attr>(...)`` call in the module source."""
    tree = ast.parse(inspect.getsource(module))
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in attrs
    ]


def test_indexer_service_uses_no_removed_event_handler_api():
    """No ``add_event_handler``/``on_event`` calls anywhere in the module.

    Asserted against the source so the check is independent of which
    Starlette happens to be installed in the venv running the tests.
    """
    assert _calls_named(indexer_service, "add_event_handler", "on_event") == []


def test_lifespan_runs_lifecycle_startup_then_shutdown():
    """Entering the lifespan calls ``lifecycle.startup(app)``; leaving it calls
    ``lifecycle.shutdown(app)`` — the same coroutines the old event handlers
    bound, with the app passed through unchanged."""
    order: list[tuple[str, object | None]] = []

    async def fake_startup(app):
        order.append(("startup", app))

    async def fake_shutdown(app):
        order.append(("shutdown", app))

    async def run():
        with patch.object(lifecycle, "startup", AsyncMock(side_effect=fake_startup)), \
             patch.object(lifecycle, "shutdown", AsyncMock(side_effect=fake_shutdown)):
            async with indexer_service.app.router.lifespan_context(indexer_service.app):
                order.append(("running", None))

    asyncio.run(run())
    assert [name for name, _ in order] == ["startup", "running", "shutdown"]
    assert all(app is indexer_service.app for _, app in order if app is not None)


def test_lifespan_still_runs_shutdown_when_body_raises():
    """A crash while serving must not skip teardown (pool close, task cancel)."""
    called = []

    async def fake_shutdown(app):
        called.append(app)

    async def run():
        with patch.object(lifecycle, "startup", AsyncMock()), \
             patch.object(lifecycle, "shutdown", AsyncMock(side_effect=fake_shutdown)):
            try:
                async with indexer_service.app.router.lifespan_context(indexer_service.app):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass

    asyncio.run(run())
    assert called == [indexer_service.app]
