"""Transport selection for the MCP entry point.

stdio is the default: it has no network surface, and the MCP spec prescribes
credentials-from-environment for it. HTTP+SSE is deprecated upstream and must
be an explicit opt-in that announces itself.
"""

import pytest

from treeloom.application.mcp_stdio import DEPRECATED_TRANSPORTS, resolve_transport


class TestResolveTransport:
    def test_default_is_stdio(self):
        assert resolve_transport(None) == "stdio"
        assert resolve_transport("") == "stdio"

    def test_case_and_whitespace_insensitive(self):
        assert resolve_transport("  STDIO  ") == "stdio"
        assert resolve_transport("Streamable-HTTP") == "streamable-http"

    @pytest.mark.parametrize(
        "raw,expected",
        [("stdio", "stdio"), ("sse", "sse"), ("streamable-http", "streamable-http")],
    )
    def test_supported_transports(self, raw, expected):
        assert resolve_transport(raw) == expected

    def test_unknown_transport_is_rejected_loudly(self):
        """Fail fast rather than silently falling back to a network transport."""
        with pytest.raises(ValueError) as exc:
            resolve_transport("carrier-pigeon")
        assert "carrier-pigeon" in str(exc.value)
        assert "stdio" in str(exc.value)

    def test_sse_is_marked_deprecated(self):
        assert "sse" in DEPRECATED_TRANSPORTS
        assert "stdio" not in DEPRECATED_TRANSPORTS
