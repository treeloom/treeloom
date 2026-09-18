"""Prompt enhancer for code search benchmarking.

Generates multiple query variants per code entity using deterministic
code analysis strategies (no LLM calls).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any

from tree_sitter_language_pack import get_parser as get_ts_parser

from treeloom.indexer import LANGUAGE_MAP


@dataclass
class PromptEnhancerConfig:
    strategies: list[str] = field(default_factory=lambda: [
        "docstring", "namespaced", "entity_context", "intent",
        "cross_cutting", "problem_driven",
    ])
    max_queries_per_entity: int = 3
    include_docstrings: bool = True
    traversal_depth: int = 1


DIFFICULTY_MAP = {
    "base": "easy",
    "docstring": "easy",
    "namespaced": "easy",
    "entity_context": "medium",
    "intent": "medium",
    "cross_cutting": "hard",
    "problem_driven": "hard",
}

ALL_STRATEGIES = list(DIFFICULTY_MAP.keys())

TREE_SITTER_TYPES = set(LANGUAGE_MAP.values())

_SOURCE_CACHE: dict[str, str] = {}


def _camel_to_words(name: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()


def _detect_language(file_path: str) -> str | None:
    ext = os.path.splitext(file_path)[1]
    return LANGUAGE_MAP.get(ext)


class Enhancer:
    """Generates enhanced query variants for a code entity.

    All strategies are deterministic — no LLM calls.
    Uses tree-sitter for docstring extraction and problem-driven analysis.
    """

    def __init__(self, config: PromptEnhancerConfig, graph_store_module: object = None):
        self.config = config
        self.graph_store = graph_store_module

    async def enhance(self, entity: dict) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = []
        file_path = entity.get("file_path", "")

        source = None
        if file_path and os.path.isfile(file_path):
            if file_path not in _SOURCE_CACHE:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    _SOURCE_CACHE[file_path] = f.read()
            source = _SOURCE_CACHE[file_path]

        for strategy in self.config.strategies:
            if strategy not in ALL_STRATEGIES:
                continue
            gen = getattr(self, f"_gen_{strategy}", None)
            if gen is None:
                continue

            result = await gen(entity, source)
            if result:
                result["strategy"] = strategy
                result["difficulty"] = DIFFICULTY_MAP[strategy]
                variants.append(result)

        return variants[:self.config.max_queries_per_entity]

    # ── helpers ──────────────────────────────────────────────

    def _read_source_text(self, source: str, start_byte: int, end_byte: int) -> str:
        return source.encode("utf-8")[start_byte:end_byte].decode("utf-8", errors="replace")

    DEFINITION_TYPES = {
        "function_definition", "method_definition", "function_declaration",
        "method_declaration", "class_definition", "class_declaration",
        "struct_specifier", "interface_declaration",
    }

    def _find_entity_node(self, root_node, start_line: int):
        """Find the tree-sitter definition node starting at start_line."""
        def search(node):
            if node.start_point[0] + 1 == start_line and node.type in self.DEFINITION_TYPES:
                return node
            for child in node.named_children:
                result = search(child)
                if result:
                    return result
            return None
        return search(root_node)

    async def _get_parent_class(self, entity: dict) -> str | None:
        if not self.graph_store:
            return None
        fp = entity.get("file_path", "")
        start_line = entity.get("start_line", 0)
        end_line = entity.get("end_line", 0)
        eid = entity.get("id", "")
        if not fp or not start_line:
            return None
        async with self.graph_store._get_session() as session:
            result = await session.run(
                """
                MATCH (parent:Entity {file_path: $file_path, type: "Class"})
                WHERE parent.start_line <= $start_line AND parent.end_line >= $end_line
                  AND parent.id <> $eid
                RETURN parent
                ORDER BY (parent.end_line - parent.start_line) ASC
                LIMIT 1
                """,
                file_path=fp,
                start_line=start_line,
                end_line=end_line,
                eid=eid,
            )
            records = await result.fetch(1)
            if records:
                return records[0]["parent"].get("name")
        return None

    # ── strategy: docstring (easy) ───────────────────────────

    async def _gen_docstring(self, entity: dict, source: str | None) -> dict | None:
        if not source:
            return None
        lang = _detect_language(entity.get("file_path", ""))
        if not lang:
            return None

        try:
            parser = get_ts_parser(lang)
            tree = parser.parse(source.encode("utf-8"))
        except Exception:
            return None

        start_line = entity.get("start_line", 1)
        entity_node = self._find_entity_node(tree.root_node, start_line)
        if not entity_node:
            return None

        docstring = self._extract_python_docstring(entity_node, source)
        if docstring:
            return {"query": docstring, "relevant_files": [entity.get("file_path", "")]}

        docstring = self._extract_comment_docstring(tree.root_node, entity_node, source)
        if docstring:
            return {"query": docstring, "relevant_files": [entity.get("file_path", "")]}

        return None

    def _extract_python_docstring(self, entity_node, source: str) -> str | None:
        body = None
        for child in entity_node.named_children:
            if child.type == "block":
                body = child
                break
        if not body or not body.named_children:
            return None
        first = body.named_children[0]
        if first.type == "expression_statement" and first.named_children:
            string_node = first.named_children[0]
            if string_node.type == "string":
                text = source[string_node.start_byte:string_node.end_byte]
                text = text.strip("\"'")
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    return text
        return None

    def _extract_comment_docstring(self, root_node, entity_node, source: str) -> str | None:
        comments = []

        def visit(node):
            if "comment" in node.type.lower():
                comments.append(node)
            for child in node.children:
                visit(child)

        visit(root_node)

        entity_start_line = entity_node.start_point[0] + 1

        # Group comments by their end line (each node is one line in C# /// style)
        by_end_line: dict[int, str] = {}
        for c in comments:
            cl = c.end_point[0] + 1
            if cl < entity_start_line and cl >= entity_start_line - 20:
                text = source[c.start_byte:c.end_byte].strip()
                cleaned = re.sub(r'^[/\*\s!-]+|[/\*\s]+$', '', text).strip()
                by_end_line[cl] = cleaned

        if not by_end_line:
            return None

        # Walk backwards from entity_start_line - 1, collecting consecutive lines
        collected: list[str] = []
        line = entity_start_line - 1
        while line in by_end_line:
            collected.insert(0, by_end_line[line])
            line -= 1

        if not collected:
            return None

        text = " ".join(c for c in collected if c).strip()
        return text if text else None

    # ── strategy: base (easy) ────────────────────────────────

    async def _gen_base(self, entity: dict, source: str | None = None) -> dict | None:
        """Plain natural-language question anchored to the entity (deterministic).

        Reconstructs the original 'base' query bucket that seeded
        featbit-clean-100 — entity-anchored templates like "How does the X
        function work?" / "What does X do?" / "Explain the X class". Type-aware;
        the function phrasing rotates deterministically (by name length) so the
        bucket isn't monotonous.
        """
        name = entity.get("name", "")
        if not name:
            return None
        words = _camel_to_words(name)
        etype = (entity.get("type") or "Function").lower()
        if etype in ("class", "interface", "struct", "enum"):
            query = f"Explain the {words} {etype}"
        elif len(name) % 2 == 0:
            query = f"How does the {words} function work?"
        else:
            query = f"What does {words} do?"
        return {"query": query, "relevant_files": [entity.get("file_path", "")]}

    # ── strategy: namespaced (easy) ──────────────────────────

    async def _gen_namespaced(self, entity: dict, source: str | None = None) -> dict | None:
        name = entity.get("name", "")
        if not name:
            return None
        words = _camel_to_words(name)

        parent_name = await self._get_parent_class(entity)
        if parent_name:
            parent_words = _camel_to_words(parent_name)
            query = f"{words} in {parent_words}"
        else:
            stem = os.path.splitext(os.path.basename(entity.get("file_path", "")))[0]
            query = f"{words} ({stem})"

        return {"query": query, "relevant_files": [entity.get("file_path", "")]}

    # ── strategy: entity_context (medium) ────────────────────

    async def _gen_entity_context(self, entity: dict, source: str | None = None) -> dict | None:
        name = entity.get("name", "")
        etype = entity.get("type", "Function")
        sig = entity.get("signature", "")
        if not name and not sig:
            return None

        parent_name = await self._get_parent_class(entity)
        sig_abbrev = sig[:100].strip() if sig else ""

        parts = [f"Find the {etype.lower()} {name}"]
        if parent_name:
            parts.append(f"in {parent_name}")
        if sig_abbrev:
            parts.append(f"— {sig_abbrev}")

        return {"query": " ".join(parts), "relevant_files": [entity.get("file_path", "")]}

    # ── strategy: intent (medium) ────────────────────────────

    async def _gen_intent(self, entity: dict, source: str | None = None) -> dict | None:
        name = entity.get("name", "")
        if not name:
            return None
        words = _camel_to_words(name)
        return {"query": f"What happens when {words}?", "relevant_files": [entity.get("file_path", "")]}

    # ── strategy: cross_cutting (hard) ───────────────────────

    async def _gen_cross_cutting(self, entity: dict, source: str | None = None) -> dict | None:
        if not self.graph_store:
            return None
        entity_id = entity.get("id", "")
        if not entity_id:
            return None

        traversed = await self.graph_store.traverse(entity_id, depth=1, max_nodes=15)

        callers = []
        callees = []
        for tn in traversed:
            nbs = tn.pop("_neighbors", [])
            for nb in nbs:
                rel = nb.get("_rel_type", "")
                direction = nb.get("_rel_direction", "")
                nb_name = nb.get("name", "")
                if rel == "CALLS" and direction == "in" and nb_name:
                    callers.append(nb_name)
                elif rel == "CALLS" and direction == "out" and nb_name:
                    callees.append(nb_name)

        name = entity.get("name", "")
        addl_files = set()

        if callers and callees:
            query = f"How does {callers[0]} call {name} and what does {callees[0]} do in relation to it?"
        elif callers:
            query = f"Which entities call {name} and why?"
        elif callees:
            query = f"What does {name} call and how does that affect its behavior?"
        else:
            return None

        return {
            "query": query,
            "relevant_files": [entity.get("file_path", "")],
            "additional_relevant_files": list(addl_files),
        }

    # ── strategy: problem_driven (hard) ──────────────────────

    async def _gen_problem_driven(self, entity: dict, source: str | None) -> dict | None:
        if not source:
            return None
        name = entity.get("name", "")
        if not name:
            return None

        lang = _detect_language(entity.get("file_path", ""))
        if not lang:
            return None

        nullable_params = self._find_nullable_params(entity, source, lang)
        if nullable_params:
            query = f"What happens if {nullable_params[0]} is null when calling {name}?"
        else:
            query = f"What errors could occur when calling {name} with invalid arguments?"

        return {"query": query, "relevant_files": [entity.get("file_path", "")]}

    def _find_nullable_params(self, entity: dict, source: str, lang: str) -> list[str]:
        try:
            parser = get_ts_parser(lang)
            tree = parser.parse(source.encode("utf-8"))
        except Exception:
            return []

        start_line = entity.get("start_line", 1)
        entity_node = self._find_entity_node(tree.root_node, start_line)
        if not entity_node:
            return []

        # Only check the first line (signature), not the whole body
        sig_start = entity_node.start_byte
        sig_end = source.find("\n", sig_start)
        if sig_end == -1:
            sig_end = entity_node.end_byte
        sig_text = source[sig_start:sig_end]

        patterns = {
            "c_sharp": r"\?\s+(\w+)\s*(?:[,\)])",
            "typescript": r"(\w+)\s*\?\s*[,\):]",
            "javascript": r"(\w+)\s*\?\s*[,\):]",
            "python": r"(\w+)\s*=\s*None",
            "java": r"@Nullable\s+\w+\s+(\w+)",
        }

        pat = patterns.get(lang)
        if not pat:
            return []

        matches = re.findall(pat, sig_text)
        return [m.strip() for m in matches if m.strip()]
