"""Token budget — named slot allocator with precise tiktoken counting.

Priority-ordered allocation ensures critical content (system prompts,
constraints) gets tokens before optional context. Graceful truncation
when the budget runs out. Falls back to character-count estimation
if tiktoken is unavailable (air-gapped / secure environments).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── tiktoken import (optional — graceful fallback) ─────────────────

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
    _DEFAULT_ENCODING = "cl100k_base"
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _DEFAULT_ENCODING = "cl100k_base"
    logger.warning(
        "tiktoken not available — falling back to char/4 token estimation. "
        "Install tiktoken for precise counting."
    )


# ── configuration ──────────────────────────────────────────────────


@dataclass
class TokenBudgetConfig:
    """Configuration for the token budget."""

    total_tokens: int = 8192  # Default: 8K context window
    encoding_name: str = _DEFAULT_ENCODING
    char_to_token_ratio: int = 4  # Fallback: 1 token ≈ 4 chars
    truncation_marker: str = "\n[...truncated...]"  # Marker for truncated content


# ── slot definition ────────────────────────────────────────────────


@dataclass
class Slot:
    """A named allocation slot in the token budget."""

    name: str
    content: str = ""
    tokens_reserved: int = 0
    truncated: bool = False


# ── the budget ─────────────────────────────────────────────────────


class TokenBudget:
    """Named slot allocator with priority-ordered reservation.

    Usage:
        budget = TokenBudget(total_tokens=4096)
        budget.reserve("system", system_prompt)
        budget.reserve("constraints", constraints_text)
        budget.reserve("context", context_text)    # may be truncated
        budget.reserve("query", user_query)

        prompt = budget.build_prompt()  # concatenated in priority order
    """

    def __init__(self, config: Optional[TokenBudgetConfig] = None):
        self._config = config or TokenBudgetConfig()
        self._slots: list[Slot] = []
        self._enc = self._get_encoder()

    # ── public API ──────────────────────────────────────────────────

    def count(self, text: str) -> int:
        """Count tokens in text using tiktoken (or fallback)."""
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # Fallback: character-count division
        return max(1, len(text) // self._config.char_to_token_ratio)

    def remaining(self) -> int:
        """Tokens remaining in the budget."""
        used = sum(s.tokens_reserved for s in self._slots)
        return max(0, self._config.total_tokens - used)

    def used(self) -> int:
        """Tokens already allocated."""
        return sum(s.tokens_reserved for s in self._slots)

    def reserve(self, name: str, text: str) -> bool:
        """Reserve tokens for a named slot.

        Returns True if the full text fit, False if truncated.
        Late slots (lower priority) may be truncated.
        """
        raw_tokens = self.count(text)
        available = self.remaining()

        if raw_tokens <= available:
            # Full allocation
            self._slots.append(Slot(
                name=name,
                content=text,
                tokens_reserved=raw_tokens,
                truncated=False,
            ))
            return True

        # Not enough tokens — truncate
        if available <= len(self._config.truncation_marker):
            # Not even room for truncation marker
            return False

        # Estimate chars we can fit, accounting for marker
        marker_tokens = self.count(self._config.truncation_marker)
        content_budget = available - marker_tokens
        if content_budget <= 0:
            return False

        # Truncate text to fit
        chars_per_token = self._config.char_to_token_ratio
        max_chars = content_budget * chars_per_token
        truncated_text = text[:max_chars] + self._config.truncation_marker

        self._slots.append(Slot(
            name=name,
            content=truncated_text,
            tokens_reserved=available,
            truncated=True,
        ))
        return False

    def build_prompt(self) -> str:
        """Build the full prompt by concatenating slots in priority order."""
        return "\n\n".join(s.content for s in self._slots if s.content)

    def slot_info(self) -> list[dict]:
        """Debug info: what each slot got."""
        return [
            {
                "name": s.name,
                "tokens": s.tokens_reserved,
                "truncated": s.truncated,
            }
            for s in self._slots
        ]

    # ── internal ────────────────────────────────────────────────────

    def _get_encoder(self):
        """Get tiktoken encoder, or None if unavailable."""
        if not _TIKTOKEN_AVAILABLE:
            return None
        try:
            import tiktoken as _tk
            return _tk.get_encoding(self._config.encoding_name)
        except Exception:
            logger.warning(
                f"Failed to get tiktoken encoding '{self._config.encoding_name}'"
            )
            return None


# ── prompt builder ─────────────────────────────────────────────────
# Uses TokenBudget to assemble prompts with constraint injection.


class PromptBuilder:
    """Builds prompts using a TokenBudget with constraint injection.

    Constraints are injected directly above the user's query, labeled
    explicitly as hard requirements — not buried in the system prompt.

    Usage:
        builder = PromptBuilder(budget)
        builder.add_system("You are a code search assistant.")
        builder.add_constraints([
            "Return ONLY valid JSON.",
            "Start with { and end with }.",
            "No markdown fencing.",
        ])
        builder.add_context(retrieved_docs_text)
        builder.add_query("How does auth middleware work?")
        messages = builder.build()  # list of {"role": ..., "content": ...}
    """

    def __init__(
        self,
        budget: TokenBudget,
        constraint_header: str = "Constraints (hard requirements, not suggestions)",
    ):
        self._budget = budget
        self._constraint_header = constraint_header
        self._system: str = ""
        self._constraints: list[str] = []
        self._mutation_hint: str = ""
        self._context: str = ""
        self._query: str = ""
        self._messages: list[dict[str, str]] = []

    def add_system(self, text: str) -> None:
        self._system = text

    def add_constraints(self, items: list[str]) -> None:
        self._constraints = items

    def add_mutation_hint(self, hint: str) -> None:
        """Inject a mutation hint from a previous retry failure."""
        self._mutation_hint = hint

    def add_context(self, text: str) -> None:
        self._context = text

    def add_query(self, text: str) -> None:
        self._query = text

    def build(self) -> list[dict[str, str]]:
        """Assemble the final messages with strict token allocation.

        Priority order (article-proven):
        1. System prompt (fixed overhead)
        2. Constraints (hard requirements, numbered)
        3. Mutation hint (retry correction)
        4. Context (truncated if tight)
        5. User query
        """
        # 1. System prompt
        if self._system:
            self._budget.reserve("system_prompt", self._system)

        # 2. Constraints block
        if self._constraints:
            constraint_block = self._constraint_header + "\n"
            for i, c in enumerate(self._constraints, 1):
                constraint_block += f"{i}. {c}\n"
            self._budget.reserve("constraints", constraint_block.strip())

        # 3. Mutation hint (retry guidance)
        if self._mutation_hint:
            self._budget.reserve(
                "mutation_hint",
                f"[Hint from previous attempt: {self._mutation_hint}]",
            )

        # 4. Context (lowest priority — truncated if tight)
        if self._context:
            self._budget.reserve("context", self._context)

        # 5. User query (always last)
        if self._query:
            self._budget.reserve("user_input", self._query)

        # Build the full prompt
        full_prompt = self._budget.build_prompt()

        # Return as messages list for _chat() compatibility
        messages: list[dict[str, str]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": full_prompt})
        return messages
