"""Benchmark domain: LLM-judge output parsing — pure, no I/O."""
from __future__ import annotations

import json
from dataclasses import dataclass

from treeloom.domain.benchmark.agent_protocol import _extract_json_object, strip_thinking


@dataclass
class JudgeScore:
    correctness: int
    completeness: int
    rationale: str
    parse_ok: bool

    def as_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "rationale": self.rationale,
            "parse_ok": self.parse_ok,
        }


def _clamp_score(value, default: int = 1) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(1, min(5, n))


def parse_judge_json(raw: str) -> JudgeScore:
    """Parse the judge's reply into a JudgeScore.

    Robust to <think> blocks, code fences, and trailing prose: extracts the
    first brace-balanced JSON object and clamps scores to 1-5. On any failure
    returns a 1/1 score flagged `parse_ok=False` so bad rows are visible, not
    silently averaged in as zeros.
    """
    cleaned = strip_thinking(raw or "")
    obj, err = _extract_json_object(cleaned)
    if obj is None:
        return JudgeScore(1, 1, f"unparseable judge output: {err}", parse_ok=False)
    rationale = obj.get("rationale") or obj.get("reason") or ""
    return JudgeScore(
        correctness=_clamp_score(obj.get("correctness")),
        completeness=_clamp_score(obj.get("completeness")),
        rationale=str(rationale)[:500],
        parse_ok=True,
    )
