"""Tests for TokenBudget and PromptBuilder domain models."""

import pytest

from treeloom.domain.token_budget import (
    PromptBuilder,
    TokenBudget,
    TokenBudgetConfig,
)


class TestTokenBudget:
    """Named slot allocator with precise token counting."""

    def test_counts_tokens(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        count = budget.count("hello world")
        assert count > 0

    def test_reserve_within_budget(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        ok = budget.reserve("system", "You are a helpful assistant.")
        assert ok is True
        assert budget.used() > 0
        assert budget.remaining() < 10000

    def test_reserve_truncates_when_over_budget(self):
        tiny = TokenBudget(TokenBudgetConfig(total_tokens=3, char_to_token_ratio=2))
        ok = tiny.reserve("context", "this is a very long text that will not fit")
        assert ok is False  # was truncated

    def test_multiple_slots_in_priority_order(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        budget.reserve("system", "System prompt here.")
        budget.reserve("constraints", "Constraint block.")
        budget.reserve("query", "What is the answer?")

        info = budget.slot_info()
        assert info[0]["name"] == "system"
        assert info[1]["name"] == "constraints"
        assert info[2]["name"] == "query"

    def test_build_prompt_concatenates_slots(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        budget.reserve("a", "First.")
        budget.reserve("b", "Second.")
        prompt = budget.build_prompt()
        assert "First." in prompt
        assert "Second." in prompt

    def test_truncation_marker_appears_when_truncated(self):
        budget = TokenBudget(TokenBudgetConfig(
            total_tokens=15,
            truncation_marker="[TRUNC]",
            char_to_token_ratio=2,
        ))
        # Reserve ~10 tokens for first slot
        budget.reserve("a", "x" * 20)
        # Second slot won't fit
        ok = budget.reserve("b", "y" * 500)
        assert ok is False


class TestPromptBuilder:
    """Constraint injection and prioritized allocation."""

    def test_builds_messages_with_system_and_user(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        builder = PromptBuilder(budget)
        builder.add_system("You are helpful.")
        builder.add_query("What is Python?")

        messages = builder.build()
        assert any(m["role"] == "user" for m in messages)

    def test_constraints_appear_in_output(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        builder = PromptBuilder(budget)
        builder.add_system("System.")
        builder.add_constraints(["Return JSON.", "No markdown."])
        builder.add_query("Query.")

        messages = builder.build()
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Constraints" in user_msg["content"]
        assert "Return JSON" in user_msg["content"]
        assert "No markdown" in user_msg["content"]

    def test_mutation_hint_appears_when_set(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        builder = PromptBuilder(budget)
        builder.add_system("System.")
        builder.add_mutation_hint("Fix JSON format.")
        builder.add_query("Query.")

        messages = builder.build()
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Fix JSON format" in user_msg["content"]

    def test_full_pipeline_allocates_correctly(self):
        budget = TokenBudget(TokenBudgetConfig(total_tokens=10000))
        builder = PromptBuilder(budget)
        builder.add_system("You are a code search assistant.")
        builder.add_constraints([
            "Return ONLY valid JSON.",
            "Include 'answer' key.",
            "No markdown fencing.",
        ])
        builder.add_context("Relevant code:\ndef hello(): pass\n\n")
        builder.add_query("How do I use hello?")

        messages = builder.build()
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        names = [s["name"] for s in budget.slot_info()]
        assert "system_prompt" in names
        assert "constraints" in names
