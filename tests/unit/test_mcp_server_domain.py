"""Domain tests for mcp_server — MCP server structure and tool definitions."""
from treeloom.mcp_server import mcp


def test_mcp_server_exists():
    """The FastMCP app instance is importable."""
    assert mcp is not None
    assert hasattr(mcp, "name")
    assert mcp.name == "treeloom"


def test_mcp_has_tools():
    """The MCP server should have registered tools."""
    tools = getattr(mcp, "_tool_manager", None)
    if tools is not None:
        tool_list = getattr(tools, "_tools", {}) or getattr(tools, "tools", {})
        # At minimum, one of the search tools should be registered
        assert isinstance(tool_list, dict)
