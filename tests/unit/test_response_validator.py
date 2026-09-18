"""Tests for ResponseValidator domain model."""

from treeloom.domain.audit import FailureMode
from treeloom.domain.response_validator import ResponseSchema, ResponseValidator


class TestResponseValidator:
    """Five validation checks on LLM output."""

    # ── empty output ────────────────────────────────────────────────

    def test_empty_output_fails(self):
        schema = ResponseSchema()
        validator = ResponseValidator(schema)
        result = validator.validate("   ")
        assert result.passed is False
        assert result.failure_mode == FailureMode.EMPTY_RESPONSE

    # ── fence stripping ─────────────────────────────────────────────

    def test_strips_markdown_json_fence(self):
        schema = ResponseSchema(must_be_json=True, required_keys=["key"])
        validator = ResponseValidator(schema)
        result = validator.validate('```json\n{"key": "value"}\n```')
        assert result.passed is True
        assert '"key": "value"' in result.cleaned_output
        assert "```" not in result.cleaned_output

    def test_strips_markdown_fence_no_language(self):
        schema = ResponseSchema(must_be_json=True, required_keys=["x"])
        validator = ResponseValidator(schema)
        result = validator.validate('```\n{"x": 1}\n```')
        assert result.passed is True

    def test_strips_trailing_fence_only(self):
        """Some models wrap only the end in ```"""
        schema = ResponseSchema()
        validator = ResponseValidator(schema)
        result = validator.validate('{"key": "val"}\n```')
        assert "```" not in result.cleaned_output

    # ── JSON validation ─────────────────────────────────────────────

    def test_valid_json_passes(self):
        schema = ResponseSchema(must_be_json=True, required_keys=["answer"])
        validator = ResponseValidator(schema)
        result = validator.validate('{"answer": "it works", "confidence": 0.9}')
        assert result.passed is True

    def test_invalid_json_fails(self):
        schema = ResponseSchema(must_be_json=True)
        validator = ResponseValidator(schema)
        result = validator.validate("not json at all")
        assert result.passed is False
        assert result.failure_mode == FailureMode.SCHEMA_VIOLATION

    def test_missing_required_key_fails(self):
        schema = ResponseSchema(
            must_be_json=True, required_keys=["answer", "confidence"]
        )
        validator = ResponseValidator(schema)
        result = validator.validate('{"answer": "hi"}')
        assert result.passed is False
        assert result.failure_mode == FailureMode.SCHEMA_VIOLATION
        assert "confidence" in result.error_detail

    def test_json_root_must_be_object(self):
        schema = ResponseSchema(must_be_json=True, required_keys=["x"])
        validator = ResponseValidator(schema)
        result = validator.validate("[1, 2, 3]")
        assert result.passed is False
        assert result.failure_mode == FailureMode.SCHEMA_VIOLATION

    # ── length boundaries ───────────────────────────────────────────

    def test_exceeds_max_length_fails(self):
        schema = ResponseSchema(max_length=10)
        validator = ResponseValidator(schema)
        result = validator.validate("this is too long")
        assert result.passed is False
        assert result.failure_mode == FailureMode.TOKEN_OVERFLOW

    def test_below_min_length_fails(self):
        schema = ResponseSchema(min_length=100)
        validator = ResponseValidator(schema)
        result = validator.validate("short")
        assert result.passed is False
        assert result.failure_mode == FailureMode.CONSTRAINT_VIOLATION

    def test_within_length_bounds_passes(self):
        schema = ResponseSchema(min_length=3, max_length=50)
        validator = ResponseValidator(schema)
        result = validator.validate("valid length text")
        assert result.passed is True

    # ── forbidden phrases ───────────────────────────────────────────

    def test_forbidden_phrase_fails(self):
        schema = ResponseSchema(
            forbidden_phrases=["I don't know", "as an AI"]
        )
        validator = ResponseValidator(schema)
        result = validator.validate("I don't know the answer")
        assert result.passed is False
        assert result.failure_mode == FailureMode.CONSTRAINT_VIOLATION

    def test_forbidden_phrase_case_insensitive(self):
        schema = ResponseSchema(forbidden_phrases=["AS AN AI"])
        validator = ResponseValidator(schema)
        result = validator.validate("As an ai, I think...")
        assert result.passed is False

    def test_no_forbidden_phrase_passes(self):
        schema = ResponseSchema(forbidden_phrases=["bad"])
        validator = ResponseValidator(schema)
        result = validator.validate("all good here")
        assert result.passed is True

    # ── must-contain keywords ───────────────────────────────────────

    def test_must_contain_found_passes(self):
        schema = ResponseSchema(must_contain=["function", "return"])
        validator = ResponseValidator(schema)
        result = validator.validate("This function must return a value")
        assert result.passed is True

    def test_must_contain_missing_fails(self):
        schema = ResponseSchema(must_contain=["def", "class"])
        validator = ResponseValidator(schema)
        result = validator.validate("Here is a function")
        assert result.passed is False
        assert result.failure_mode == FailureMode.CONSTRAINT_VIOLATION

    # ── combined checks ─────────────────────────────────────────────

    def test_all_checks_pass(self):
        schema = ResponseSchema(
            must_be_json=True,
            required_keys=["summary"],
            min_length=15,
            max_length=500,
            forbidden_phrases=["as an AI"],
            must_contain=["code"],
        )
        validator = ResponseValidator(schema)
        result = validator.validate(
            '{"summary": "This code parses AST trees for analysis"}'
        )
        assert result.passed is True

    def test_fail_on_first_violation(self):
        """First failure stops checking — order: empty → JSON → length → phrases → keywords."""
        schema = ResponseSchema(
            must_be_json=True,
            required_keys=["x"],
            max_length=50,
        )
        validator = ResponseValidator(schema)
        # This is too long but also not JSON. Should fail on JSON first.
        result = validator.validate("valid JSON but exceeds max length limit test" * 10)
        assert result.failure_mode == FailureMode.SCHEMA_VIOLATION
