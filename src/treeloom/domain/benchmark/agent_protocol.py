"""Benchmark domain: ReAct agent protocol parsing — pure, no I/O.

The agent emits, per turn, either a single tool action:

    THOUGHT: <reasoning>
    ACTION: tool_name({"arg": "value"})

or a final answer:

    THOUGHT: <reasoning>
    FINAL_ANSWER: <prose>

`parse_action` turns raw model output into a structured `ParsedAction`. It is
deliberately forgiving about surrounding prose, code fences, and reasoning
(`<think>`) blocks — the harness nudges the model once on a parse failure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)
# ACTION: name( ...args... )  — name is an identifier; args captured lazily up
# to the LAST closing paren on/after the action so trailing prose is tolerated.
_ACTION_RE = re.compile(
    r"ACTION\s*:\s*([A-Za-z_]\w*)\s*\((.*)\)", re.DOTALL | re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks left over from reasoning models."""
    return _THINK_TAG_RE.sub("", text or "").strip()


@dataclass
class ParsedAction:
    kind: str  # "final" | "tool" | "none"
    final_answer: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    error: str = ""


def _extract_json_object(blob: str) -> tuple[dict | None, str]:
    """Extract the first brace-balanced JSON object from `blob`.

    Returns (obj, error). Tolerates trailing junk after the closing brace.
    """
    start = blob.find("{")
    if start == -1:
        # No JSON object — allow empty-arg actions like tool().
        stripped = blob.strip()
        if stripped == "":
            return {}, ""
        return None, "no JSON object found"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(blob)):
        ch = blob[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = blob[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if not isinstance(obj, dict):
                        return None, "not a JSON object"
                    return obj, ""
                except json.JSONDecodeError as e:
                    return None, f"invalid JSON: {e}"
    return None, "unbalanced braces"


def parse_action(text: str) -> ParsedAction:
    """Parse one model turn into a ParsedAction.

    Precedence: a FINAL_ANSWER (even alongside an ACTION) wins, so the agent
    can stop cleanly. Then a tool ACTION. Otherwise `kind="none"` with an error
    the harness surfaces back to the model.
    """
    cleaned = strip_thinking(text)

    final = _FINAL_RE.search(cleaned)
    action = _ACTION_RE.search(cleaned)
    # If both appear, prefer whichever comes first in the text.
    if final and (not action or final.start() <= action.start()):
        answer = final.group(1).strip()
        if not answer:
            return ParsedAction(kind="none", error="empty FINAL_ANSWER")
        return ParsedAction(kind="final", final_answer=answer)

    if action:
        tool = action.group(1)
        obj, err = _extract_json_object(action.group(2))
        if obj is None:
            return ParsedAction(kind="none", tool=tool, error=err)
        return ParsedAction(kind="tool", tool=tool, args=obj)

    return ParsedAction(kind="none", error="no ACTION or FINAL_ANSWER found")
