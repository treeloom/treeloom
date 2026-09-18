"""Backward-compatible re-export of prompt enhancer module."""
from treeloom.domain.prompt_enhancer import (
    _camel_to_words,
    _detect_language,
    ALL_STRATEGIES,
    DIFFICULTY_MAP,
    Enhancer,
    PromptEnhancerConfig,
)  # noqa: F401
