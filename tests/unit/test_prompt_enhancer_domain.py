"""Domain tests for prompt_enhancer — deterministic query variant generation."""
import pytest
from treeloom.prompt_enhancer import (
    Enhancer,
    PromptEnhancerConfig,
    _camel_to_words,
    DIFFICULTY_MAP,
    ALL_STRATEGIES,
)


# ── camel_to_words ──────────────────────────────────────────────────────

def test_camel_to_words_simple():
    assert _camel_to_words("helloWorld") == "hello world"

def test_camel_to_words_pascal():
    assert _camel_to_words("HelloWorld") == "hello world"

def test_camel_to_words_acronym():
    assert _camel_to_words("parseHTML") == "parse html"

def test_camel_to_words_single_word():
    assert _camel_to_words("hello") == "hello"

def test_camel_to_words_empty_string():
    assert _camel_to_words("") == ""


# ── Config defaults ─────────────────────────────────────────────────────

def test_config_defaults():
    config = PromptEnhancerConfig()
    assert len(config.strategies) >= 4
    assert config.max_queries_per_entity == 3
    assert config.include_docstrings is True
    assert config.traversal_depth == 1

def test_config_custom_strategies():
    config = PromptEnhancerConfig(strategies=["entity_context", "intent"])
    assert config.strategies == ["entity_context", "intent"]

def test_config_max_queries():
    config = PromptEnhancerConfig(max_queries_per_entity=5)
    assert config.max_queries_per_entity == 5

def test_config_custom_values():
    config = PromptEnhancerConfig(
        strategies=["namespaced"],
        max_queries_per_entity=5,
        include_docstrings=False,
        traversal_depth=2,
    )
    assert config.strategies == ["namespaced"]
    assert config.max_queries_per_entity == 5
    assert config.include_docstrings is False
    assert config.traversal_depth == 2


# ── Difficulty map ──────────────────────────────────────────────────────

def test_difficulty_map_covers_all_strategies():
    for s in ALL_STRATEGIES:
        assert s in DIFFICULTY_MAP, f"Missing difficulty for {s}"

def test_difficulty_values():
    assert DIFFICULTY_MAP["docstring"] == "easy"
    assert DIFFICULTY_MAP["namespaced"] == "easy"
    assert DIFFICULTY_MAP["entity_context"] == "medium"
    assert DIFFICULTY_MAP["intent"] == "medium"
    assert DIFFICULTY_MAP["cross_cutting"] == "hard"
    assert DIFFICULTY_MAP["problem_driven"] == "hard"


# ── Enhancer: entity_context strategy ───────────────────────────────────

@pytest.mark.asyncio
async def test_entity_context_produces_query():
    config = PromptEnhancerConfig(strategies=["entity_context"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/file.py::my_func",
        "name": "my_func",
        "type": "Function",
        "file_path": "/tmp/nonexistent_app_file.py",
        "signature": "def my_func(x: int) -> str",
        "language": "python",
    }
    queries = await enhancer.enhance(entity)
    assert len(queries) >= 1
    q = queries[0]
    assert "query" in q
    assert "my_func" in q["query"].lower()
    assert q["strategy"] == "entity_context"
    assert q["difficulty"] == "medium"


@pytest.mark.asyncio
async def test_entity_context_includes_signature():
    config = PromptEnhancerConfig(strategies=["entity_context"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/mod.py::set_tags",
        "name": "SetTagsAsync",
        "type": "Function",
        "file_path": "/tmp/nonexistent_controller.py",
        "signature": "Task SetTagsAsync(FlagModel model)",
        "language": "csharp",
    }
    queries = await enhancer.enhance(entity)
    q = queries[0]
    assert "set tags" in q["query"].lower() or "SetTagsAsync" in q["query"]


# ── Enhancer: intent strategy ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_produces_query():
    config = PromptEnhancerConfig(strategies=["intent"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/flags.py::delete_flag_async",
        "name": "DeleteFlagAsync",
        "type": "Function",
        "file_path": "/tmp/nonexistent_flags.py",
        "signature": "async def delete_flag_async(flag_id: str)",
        "language": "python",
    }
    queries = await enhancer.enhance(entity)
    assert len(queries) >= 1
    q = queries[0]
    assert q["strategy"] == "intent"
    assert q["difficulty"] == "medium"
    assert "delete flag async" in q["query"].lower()


# ── Enhancer: base strategy (entity-anchored NL question) ───────────────

@pytest.mark.asyncio
async def test_base_function_produces_question():
    config = PromptEnhancerConfig(strategies=["base"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/m.py::openCreationDrawer",
        "name": "openCreationDrawer",
        "type": "Function",
        "file_path": "/tmp/nonexistent_m.py",
        "language": "javascript",
    }
    queries = await enhancer.enhance(entity)
    assert len(queries) == 1
    q = queries[0]
    assert q["strategy"] == "base"
    assert q["difficulty"] == "easy"
    # camelCase expanded; one of the deterministic function phrasings
    assert "open creation drawer" in q["query"].lower()
    assert q["query"].lower().startswith(("how does the", "what does"))


@pytest.mark.asyncio
async def test_base_class_uses_class_phrasing():
    config = PromptEnhancerConfig(strategies=["base"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "cls://test/m.cs::AccessTokenVm",
        "name": "AccessTokenVm",
        "type": "Class",
        "file_path": "/tmp/nonexistent_m.cs",
        "language": "csharp",
    }
    queries = await enhancer.enhance(entity)
    assert queries[0]["query"] == "Explain the access token vm class"


def test_base_in_all_strategies():
    assert "base" in ALL_STRATEGIES
    assert DIFFICULTY_MAP["base"] == "easy"


# ── Enhancer: namespaced strategy ───────────────────────────────────────

@pytest.mark.asyncio
async def test_namespaced_produces_query():
    config = PromptEnhancerConfig(strategies=["namespaced"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/my_module.py::my_func",
        "name": "my_func",
        "type": "Function",
        "file_path": "/tmp/nonexistent_my_module.py",
        "signature": "def my_func(): pass",
        "language": "python",
    }
    queries = await enhancer.enhance(entity)
    assert len(queries) >= 1
    q = queries[0]
    assert q["strategy"] == "namespaced"
    assert q["difficulty"] == "easy"


# ── Enhancer: unknown strategy silently skipped ─────────────────────────

@pytest.mark.asyncio
async def test_unknown_strategy_skipped():
    """Unsupported strategies are silently ignored (no LLM call needed)."""
    config = PromptEnhancerConfig(strategies=["nonexistent_strategy"])
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/mod.py::f",
        "name": "f",
        "type": "Function",
        "file_path": "/tmp/nonexistent_mod.py",
        "signature": "def f(): pass",
        "language": "python",
    }
    queries = await enhancer.enhance(entity)
    assert queries == []


# ── max_queries_per_entity ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_respects_max_queries():
    config = PromptEnhancerConfig(
        strategies=["entity_context", "intent", "namespaced"],
        max_queries_per_entity=2,
    )
    enhancer = Enhancer(config=config)
    entity = {
        "id": "func://test/mod.py::f",
        "name": "f",
        "type": "Function",
        "file_path": "/tmp/nonexistent_mod2.py",
        "signature": "def f(): pass",
        "language": "python",
    }
    queries = await enhancer.enhance(entity)
    assert len(queries) <= 2
