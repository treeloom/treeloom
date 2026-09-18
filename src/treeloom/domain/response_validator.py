"""Response validator — five checks on every LLM output.

Validates structured outputs before they reach downstream code.
Handles markdown fence stripping (models often wrap JSON in ```).
Maps failures to FailureMode enum for retry engine consumption.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from treeloom.domain.audit import FailureMode

# Matches ```json ... ``` or ``` ... ``` markdown fencing
_MD_FENCE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL
)


@dataclass
class ResponseSchema:
    """What a valid response must satisfy.

    All checks are optional — only configured checks run.
    """

    required_keys: list[str] = field(default_factory=list)
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    forbidden_phrases: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_be_json: bool = False


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    failure_mode: FailureMode = FailureMode.NONE
    cleaned_output: str = ""  # Output after fence stripping
    error_detail: str = ""

    @staticmethod
    def success(cleaned: str) -> "ValidationResult":
        return ValidationResult(
            passed=True, failure_mode=FailureMode.NONE, cleaned_output=cleaned
        )

    @staticmethod
    def fail(
        mode: FailureMode, raw: str, detail: str
    ) -> "ValidationResult":
        return ValidationResult(
            passed=False,
            failure_mode=mode,
            cleaned_output=raw,
            error_detail=detail,
        )


class ResponseValidator:
    """Five checks on every LLM response.

    Usage:
        schema = ResponseSchema(must_be_json=True, required_keys=["answer", "confidence"])
        validator = ResponseValidator(schema)
        result = validator.validate(llm_response_text)
        if result.passed:
            data = json.loads(result.cleaned_output)
    """

    def __init__(self, schema: ResponseSchema):
        self._schema = schema

    def validate(self, raw_output: str) -> ValidationResult:
        """Run all configured checks. Returns on first failure.

        Checks run in this order:
        1. Empty output
        2. Markdown fence stripping
        3. JSON structure (if must_be_json)
        4. Length boundaries
        5. Forbidden phrases
        6. Must-contain keywords
        """
        # 1. Empty output
        stripped = raw_output.strip()
        if not stripped:
            return ValidationResult.fail(
                FailureMode.EMPTY_RESPONSE,
                raw_output,
                "output is empty",
            )

        # 2. Strip markdown fencing
        cleaned = self._strip_fence(stripped)

        # 3. JSON validation
        if self._schema.must_be_json:
            json_result = self._validate_json(cleaned)
            if json_result is not None:
                return json_result

        # 4. Length boundaries
        if self._schema.max_length is not None and len(cleaned) > self._schema.max_length:
            return ValidationResult.fail(
                FailureMode.TOKEN_OVERFLOW,
                cleaned,
                f"output length {len(cleaned)} exceeds max {self._schema.max_length}",
            )
        if self._schema.min_length is not None and len(cleaned) < self._schema.min_length:
            return ValidationResult.fail(
                FailureMode.CONSTRAINT_VIOLATION,
                cleaned,
                f"output length {len(cleaned)} below min {self._schema.min_length}",
            )

        # 5. Forbidden phrases
        for phrase in self._schema.forbidden_phrases:
            if phrase.lower() in cleaned.lower():
                return ValidationResult.fail(
                    FailureMode.CONSTRAINT_VIOLATION,
                    cleaned,
                    f"forbidden phrase found: {phrase!r}",
                )

        # 6. Must-contain keywords
        for phrase in self._schema.must_contain:
            if phrase.lower() not in cleaned.lower():
                return ValidationResult.fail(
                    FailureMode.CONSTRAINT_VIOLATION,
                    cleaned,
                    f"required phrase not found: {phrase!r}",
                )

        # All checks passed
        return ValidationResult.success(cleaned)

    # ── internal ────────────────────────────────────────────────────

    @staticmethod
    def _strip_fence(text: str) -> str:
        """Strip markdown code fencing (```json ... ```) from output.

        Models frequently wrap JSON in markdown backticks even when
        explicitly told not to. This avoids wasting a retry on a
        purely cosmetic issue.
        """
        m = _MD_FENCE.match(text)
        if m:
            return m.group(1).strip()
        # Also try stripping leading/trailing ``` without full match
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text)
        cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
        return cleaned.strip()

    def _validate_json(self, text: str) -> Optional[ValidationResult]:
        """Validate JSON structure and required keys.

        Returns ValidationResult on failure, None on success.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return ValidationResult.fail(
                FailureMode.SCHEMA_VIOLATION,
                text,
                f"invalid JSON: {e}",
            )

        # Check required keys
        if not isinstance(data, dict):
            return ValidationResult.fail(
                FailureMode.SCHEMA_VIOLATION,
                text,
                "JSON root must be an object",
            )

        for key in self._schema.required_keys:
            if key not in data:
                return ValidationResult.fail(
                    FailureMode.SCHEMA_VIOLATION,
                    text,
                    f"missing required key: {key!r}",
                )

        return None  # Success
