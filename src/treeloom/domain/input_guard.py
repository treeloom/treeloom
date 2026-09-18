"""Input guard — validates user input before it reaches the LLM.

OWASP LLM Top 10 injection pattern detection. Catches prompt override,
jailbreak, token smuggling, persona hijacking, and indirect injection
attempts. Blocks before any LLM call is made — zero token cost for
blocked input.

Empty inputs and length overflows are caught first. Injection patterns
run second. All checks return structured ValidationResult — never
throws an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InputVerdict(str, Enum):
    """What the InputGuard decided about an input."""

    PASSED = "passed"
    EMPTY = "empty"
    TOKEN_OVERFLOW = "token_overflow"
    PROMPT_INJECTION = "prompt_injection"


@dataclass
class InputValidation:
    """Result of input guard validation."""

    verdict: InputVerdict
    reason: str = ""
    matched_pattern: str = ""
    input_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == InputVerdict.PASSED


# ── injection patterns (OWASP LLM Top 10) ──────────────────────────
# These are compiled once at module load for performance.
# Patterns run in microseconds — not an LLM call.

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Direct override attempts
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above|instructions?|prompts?)", re.I),
     "prompt override: ignore instructions"),
    (re.compile(r"disregard\s+(your|the)\s+(system\s+)?(prompt|instructions?)", re.I),
     "prompt override: disregard system prompt"),
    (re.compile(r"forget\s+(all\s+)?(previous|prior|your\s+(previous|prior)|your)\s+(instructions?|training)", re.I),
     "prompt override: forget instructions"),
    (re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new)\s+(AI|model|assistant|system)", re.I),
     "persona hijacking: role reassignment"),
    (re.compile(r"pretend\s+(you\s+(are|have)\s+)?(no\s+restrictions?|unlimited|no\s+limits?)", re.I),
     "jailbreak: pretend no restrictions"),
    (re.compile(r"act\s+as\s+(if\s+)?(you\s+are\s+)?(DAN|jailbroken|unfiltered|uncensored)", re.I),
     "jailbreak: DAN-style persona"),

    # System prompt extraction
    (re.compile(r"(reveal|tell\s+me|show|display|print|repeat|output)\s+(your|the)\s+(system\s+)?(prompt|instructions?)", re.I),
     "system prompt extraction"),
    (re.compile(r"(what\s+(is|are)\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)\??$", re.I),
     "system prompt inquiry"),

    # Token smuggling / encoding tricks
    (re.compile(r"(base64|rot13|hex|unicode)\s*(encode|decode|escap)", re.I),
     "token smuggling: encoding request"),
    (re.compile(r"(convert|translate)\s+(to|from)\s+(base64|rot13|hex|unicode)", re.I),
     "token smuggling: encoding conversion"),

    # Indirect injection patterns
    (re.compile(r"(the\s+)?(following|below)\s+(text|document|article|search\s+result)\s+(contains|says|states)\s+(new\s+)?instructions?", re.I),
     "indirect injection: embedded instructions"),
    (re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
     "injection: instruction tokens"),

    # Role/context manipulation
    (re.compile(r"you\s+are\s+(now\s+)?(a\s+)?(developer|hacker|cracker|malicious|evil)", re.I),
     "persona hijacking: malicious role"),
    (re.compile(r"(switch|change|flip)\s+(your\s+)?(role|persona|alignment)\s+to", re.I),
     "persona hijacking: role switch"),
    (re.compile(r"(bypass|override|circumvent)\s+(the\s+)?(system|safety|security|filter|guard)", re.I),
     "jailbreak: bypass attempt"),

    # Output format manipulation
    (re.compile(r"(do\s+not|don't|never)\s+(follow|obey|listen\s+to)\s+(the\s+)?(system|above|previous)", re.I),
     "prompt override: instruction negation"),
    (re.compile(r"from\s+now\s+on\s+(you\s+are|respond\s+as)", re.I),
     "persona hijacking: behavioral reset"),

    # Additional OWASP patterns
    (re.compile(r"respond\s+with\s+\\\"[^\\\"]*\\\"\s+no\s+matter\s+what", re.I),
     "injection: forced response"),
    (re.compile(r"everything\s+(above|below)\s+(is|was)\s+(a\s+)?(lie|joke|test|simulation)", re.I),
     "injection: context invalidation"),
    (re.compile(r"(\b|^)sudo\b.*\b(execute|run|do)\b", re.I),
     "injection: sudo-style command"),
]


# ── configuration ──────────────────────────────────────────────────


@dataclass
class InputGuardConfig:
    """Configuration for the input guard."""

    max_input_length: int = 2000  # Characters (not tokens)
    min_input_length: int = 1  # Minimum non-whitespace chars
    enable_injection_check: bool = True
    custom_patterns: list[tuple[str, str]] = field(default_factory=list)


# ── the guard ──────────────────────────────────────────────────────


class InputGuard:
    """Validates user input before any LLM call.

    Usage:
        guard = InputGuard()
        result = guard.validate(user_query)
        if not result.passed:
            return error_response(result)
        # Proceed to LLM call
    """

    def __init__(self, config: Optional[InputGuardConfig] = None):
        self._config = config or InputGuardConfig()
        self._patterns = list(_INJECTION_PATTERNS)
        # Register custom patterns
        for pattern_str, label in self._config.custom_patterns:
            self._patterns.append((re.compile(pattern_str, re.I), label))

    def validate(self, text: str) -> InputValidation:
        """Run all checks on user input. Returns on first failure."""

        # 1. Empty check (whitespace-only counts as empty)
        stripped = text.strip()
        if not stripped:
            return InputValidation(
                verdict=InputVerdict.EMPTY,
                reason="input is empty",
            )

        # 2. Minimum length check
        if len(stripped) < self._config.min_input_length:
            return InputValidation(
                verdict=InputVerdict.EMPTY,
                reason=f"input too short (min {self._config.min_input_length} chars)",
            )

        # 3. Length overflow check
        if len(stripped) > self._config.max_input_length:
            return InputValidation(
                verdict=InputVerdict.TOKEN_OVERFLOW,
                reason=f"input exceeds max length ({self._config.max_input_length} chars)",
            )

        # 4. Injection pattern check
        if self._config.enable_injection_check:
            for pattern, label in self._patterns:
                if pattern.search(stripped):
                    return InputValidation(
                        verdict=InputVerdict.PROMPT_INJECTION,
                        reason=f"injection pattern detected: {label}",
                        matched_pattern=label,
                    )

        # All checks passed
        return InputValidation(
            verdict=InputVerdict.PASSED,
            reason="all checks passed",
        )

    def is_safe(self, text: str) -> bool:
        """Quick check: is this input safe to pass to the LLM?"""
        return self.validate(text).passed

    @property
    def pattern_count(self) -> int:
        """Number of injection patterns registered."""
        return len(self._patterns)
