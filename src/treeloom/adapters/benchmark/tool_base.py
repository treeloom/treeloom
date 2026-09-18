"""Benchmark adapter: shared Tool container for the agentic harness."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """One tool the agent can call.

    `run` is an async callable taking keyword args (the parsed ACTION args)
    and returning a JSON-serializable dict observation. `parameters` is a JSON
    Schema used only for the optional `--native-tools` (OpenAI tool_calls) path.
    """

    name: str
    description: str
    run: Callable[..., Awaitable[dict]]
    parameters: dict = field(default_factory=dict)

    def describe(self) -> str:
        """One-line description injected into the ReAct system prompt."""
        return f"- {self.name}: {self.description}"

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}},
            },
        }
