"""Unit tests for domain/benchmark/query_hygiene.py — Detroit-style, no mocks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from treeloom.domain.benchmark.query_hygiene import (
    apply_quota,
    camel_tokens,
    dedup,
    filter_answerable,
    is_answerable,
    mentions,
    name_to_files,
    path_hint,
)


# ── camel_tokens ──────────────────────────────────────────────────────────────

class TestCamelTokens:
    def test_pascal_case(self):
        assert camel_tokens("GetEmailAsync") == ["get", "email", "async"]

    def test_single_word_lowercase(self):
        # single char tokens are dropped
        assert camel_tokens("handle") == ["handle"]

    def test_single_char_is_dropped(self):
        assert camel_tokens("A") == []

    def test_empty_string(self):
        assert camel_tokens("") == []

    def test_none_treated_as_empty(self):
        assert camel_tokens(None) == []  # type: ignore[arg-type]

    def test_acronym_followed_by_word(self):
        # "HTML" split from "Parser"
        tokens = camel_tokens("HTMLParser")
        assert "html" in tokens
        assert "parser" in tokens


# ── mentions ──────────────────────────────────────────────────────────────────

class TestMentions:
    def test_verbatim_match(self):
        assert mentions("find getuser in the service", "GetUser") is True

    def test_all_camel_tokens_match(self):
        # "OidcClient" → ["oidc", "client"]; both appear in query
        assert mentions("configure oidc client authentication", "OidcClient") is True

    def test_partial_camel_tokens_no_match(self):
        # "oidc" appears but "client" does not
        assert mentions("configure oidc authentication", "OidcClient") is False

    def test_not_mentioned(self):
        assert mentions("how does logging work", "GetEmailAsync") is False

    def test_case_insensitive_verbatim(self):
        assert mentions("call getemailasync here", "GetEmailAsync") is True

    def test_single_char_name_no_tokens(self):
        # camel_tokens drops len<=1 tokens, so toks is empty → False
        assert mentions("find X here", "X") is False


# ── path_hint ─────────────────────────────────────────────────────────────────

class TestPathHint:
    def _row(self, query: str, files: list[str]) -> dict:
        return {"query": query, "relevant_files": files}

    def test_discriminating_path_part_found(self):
        row = self._row(
            "how does oidcclient work",
            ["/home/user/myrepo/auth/OidcClient.cs"],
        )
        # "oidcclient" appears in query and is not a stop part
        assert path_hint(row, stop_parts={"home", "user", "myrepo"}) is True

    def test_only_stop_parts_no_hint(self):
        row = self._row(
            "find source code",
            ["/home/user/myrepo/src/app.ts"],
        )
        # every non-trivial part ("home","user","myrepo") is in stop_parts; "src","app","ts" in BASE
        assert path_hint(row, stop_parts={"home", "user", "myrepo"}) is False

    def test_short_tokens_ignored(self):
        # "api" has len==3, filtered by len>3 guard
        row = self._row(
            "api endpoint",
            ["/repo/api/main.py"],
        )
        assert path_hint(row, stop_parts=set()) is False

    def test_no_relevant_files(self):
        row = self._row("some query", [])
        assert path_hint(row, stop_parts=set()) is False

    def test_relevant_files_none(self):
        row = {"query": "some query"}
        assert path_hint(row, stop_parts=set()) is False


# ── name_to_files ─────────────────────────────────────────────────────────────

class TestNameToFiles:
    def test_builds_map(self):
        rows = [
            {"entity": {"name": "Foo"}, "relevant_files": ["/a.py", "/b.py"]},
            {"entity": {"name": "foo"}, "relevant_files": ["/c.py"]},  # same name, lowercased
            {"entity": {"name": "Bar"}, "relevant_files": ["/d.py"]},
        ]
        nf = name_to_files(rows)
        assert nf["foo"] == {"/a.py", "/b.py", "/c.py"}
        assert nf["bar"] == {"/d.py"}

    def test_missing_entity_skipped(self):
        rows = [{"relevant_files": ["/x.py"]}]
        nf = name_to_files(rows)
        assert nf == {}

    def test_empty_name_skipped(self):
        rows = [{"entity": {"name": ""}, "relevant_files": ["/x.py"]}]
        nf = name_to_files(rows)
        assert nf == {}


# ── is_answerable ─────────────────────────────────────────────────────────────

class TestIsAnswerable:
    def _make_name_files(self, name: str, files: list[str]) -> dict:
        return {name.lower(): set(files)}

    def test_discriminating_entity_answerable(self):
        row = {
            "query": "how does OidcClient validate tokens",
            "entity": {"name": "OidcClient"},
            "relevant_files": ["/auth/OidcClient.cs"],
        }
        nf = self._make_name_files("OidcClient", ["/auth/OidcClient.cs"])
        assert is_answerable(row, nf, stop_parts=set()) is True

    def test_non_discriminating_but_path_hint(self):
        # name maps to 5 files (>3), but query contains a path segment
        row = {
            "query": "find handle in oidcclient module",
            "entity": {"name": "Handle"},
            "relevant_files": ["/auth/OidcClient.cs"],
        }
        files = ["/a.cs", "/b.cs", "/c.cs", "/d.cs", "/e.cs"]
        nf = self._make_name_files("Handle", files)
        # "oidcclient" is in the query and in the file path
        assert is_answerable(row, nf, stop_parts=set()) is True

    def test_non_discriminating_no_path_hint_not_answerable(self):
        row = {
            "query": "how does handle work",
            "entity": {"name": "Handle"},
            "relevant_files": ["/a.cs"],
        }
        many_files = [f"/file{i}.cs" for i in range(10)]
        nf = self._make_name_files("Handle", many_files)
        assert is_answerable(row, nf, stop_parts=set()) is False

    def test_not_mentioned_not_answerable(self):
        row = {
            "query": "how does authentication work",
            "entity": {"name": "OidcClient"},
            "relevant_files": ["/auth/OidcClient.cs"],
        }
        nf = self._make_name_files("OidcClient", ["/auth/OidcClient.cs"])
        # "oidcclient" not in query (only "authentication")
        assert is_answerable(row, nf, stop_parts=set()) is False

    def test_missing_entity_not_answerable(self):
        row = {"query": "how does it work", "relevant_files": ["/a.cs"]}
        assert is_answerable(row, {}, stop_parts=set()) is False


# ── dedup ─────────────────────────────────────────────────────────────────────

class TestDedup:
    def test_drops_duplicate_query_text(self):
        rows = [
            {"query": "Find GetUser", "id": "1"},
            {"query": "find getuser", "id": "2"},   # same when stripped+lowered
            {"query": "find getuser  ", "id": "3"},  # trailing space
            {"query": "Another query", "id": "4"},
        ]
        result = dedup(rows)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "4"

    def test_keeps_first_occurrence(self):
        rows = [{"query": "Q", "id": "a"}, {"query": "Q", "id": "b"}]
        assert dedup(rows)[0]["id"] == "a"

    def test_empty_input(self):
        assert dedup([]) == []

    def test_no_duplicates_unchanged(self):
        rows = [{"query": "A"}, {"query": "B"}, {"query": "C"}]
        assert len(dedup(rows)) == 3


# ── apply_quota ───────────────────────────────────────────────────────────────

class TestApplyQuota:
    def _rows(self, strategies: list[str]) -> list[dict]:
        return [{"query": f"q{i}", "strategy": s} for i, s in enumerate(strategies)]

    def test_none_passthrough(self):
        rows = self._rows(["base", "intent", "base"])
        assert apply_quota(rows, None) == rows

    def test_per_strategy_cap(self):
        rows = self._rows(["base", "base", "base", "intent", "intent"])
        result = apply_quota(rows, {"base": 2, "intent": 1})
        base_rows = [r for r in result if r["strategy"] == "base"]
        intent_rows = [r for r in result if r["strategy"] == "intent"]
        assert len(base_rows) == 2
        assert len(intent_rows) == 1

    def test_missing_strategy_defaults_to_base(self):
        rows = [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]
        result = apply_quota(rows, {"base": 2})
        assert len(result) == 2

    def test_stops_early_when_all_quotas_exhausted(self):
        rows = self._rows(["base"] * 10)
        result = apply_quota(rows, {"base": 3})
        assert len(result) == 3

    def test_empty_quota_keeps_nothing(self):
        rows = self._rows(["base", "intent"])
        result = apply_quota(rows, {"base": 0, "intent": 0})
        assert result == []

    def test_preserves_input_order(self):
        rows = self._rows(["intent", "base", "intent"])
        result = apply_quota(rows, {"base": 1, "intent": 2})
        assert [r["strategy"] for r in result] == ["intent", "base", "intent"]


# ── filter_answerable integration ─────────────────────────────────────────────

class TestFilterAnswerable:
    def test_keeps_answerable_rows(self):
        rows = [
            {
                "query": "how does OidcClient validate tokens",
                "entity": {"name": "OidcClient"},
                "relevant_files": ["/auth/OidcClient.cs"],
                "strategy": "entity_context",
            },
            {
                "query": "how does authentication work",
                "entity": {"name": "Handle"},
                "relevant_files": ["/a.cs", "/b.cs", "/c.cs", "/d.cs"],
                "strategy": "intent",
            },
        ]
        result = filter_answerable(rows, stop_parts=set())
        # OidcClient: name in query + maps to 1 file → answerable
        # Handle: name NOT in query → not answerable
        assert len(result) == 1
        assert result[0]["entity"]["name"] == "OidcClient"

    def test_stop_parts_suppress_generic_path_tokens(self):
        # "myrepo" in stop_parts, "oidcclient" not in stop_parts.
        # Entity "Handle" — camel_tokens gives ["handle"]; "handle" is a substring
        # of "handler" in the query, so mentions() returns True.
        # name_to_files maps "handle" → 1 file (≤3) → discriminating → answerable.
        rows = [
            {
                "query": "find oidcclient auth handler",
                "entity": {"name": "Handle"},
                "relevant_files": ["/home/user/myrepo/oidcclient.cs"],
                "strategy": "base",
            },
        ]
        result = filter_answerable(rows, stop_parts={"home", "user", "myrepo"})
        assert len(result) == 1

    def test_stop_parts_block_non_discriminating_path_hint(self):
        # Entity "Widget" maps to 5 files (>3), so path_hint is the only gate.
        # If every non-trivial path token is in stop_parts, path_hint is False.
        rows = [
            {
                "query": "how does widget work",
                "entity": {"name": "Widget"},
                "relevant_files": ["/home/user/myrepo/src/main.py"],
                "strategy": "base",
            },
        ]
        many = [f"/file{i}.py" for i in range(5)]
        # Patch name_files externally by adding extra rows with same entity name
        extra = [
            {"query": f"q{i}", "entity": {"name": "Widget"}, "relevant_files": [f"/file{i}.py"]}
            for i in range(5)
        ]
        result = filter_answerable(rows + extra, stop_parts={"home", "user", "myrepo"})
        # "widget" in "how does widget work" → mentions = True
        # name maps to 6 files (>3); path_hint: parts of /home/user/myrepo/src/main.py
        # — "home","user","myrepo","main" → "main" not in stop_parts and len>3 → check
        # "main" in "how does widget work"? No → path_hint = False → not answerable
        widget_rows = [r for r in result if (r.get("entity") or {}).get("name") == "Widget"
                       and "how does widget work" in r["query"]]
        assert widget_rows == []


class TestSampleEntities:
    """Seeded entity sampling for query generation (enhanced_query_gen)."""

    def _entities(self, n):
        return [{"id": i, "file_path": f"/repo/{chr(97 + i % 26)}/f{i}.ts"} for i in range(n)]

    def test_population_smaller_than_k_returns_every_entity(self):
        """The whole population is returned -- as a SET. Order is now seeded
        random rather than input order (see test below for why)."""
        from treeloom.application.benchmark.enhanced_query_gen import sample_entities

        ents = self._entities(10)
        out = sample_entities(ents, 20, seed=42)
        assert len(out) == len(ents)
        assert {e["id"] for e in out} == {e["id"] for e in ents}

    def test_deterministic_for_same_seed(self):
        from treeloom.application.benchmark.enhanced_query_gen import sample_entities

        ents = self._entities(1000)
        a = sample_entities(ents, 50, seed=42)
        b = sample_entities(ents, 50, seed=42)
        assert a == b
        assert len(a) == 50

    def test_different_seed_differs(self):
        from treeloom.application.benchmark.enhanced_query_gen import sample_entities

        ents = self._entities(1000)
        assert sample_entities(ents, 50, seed=42) != sample_entities(ents, 50, seed=7)

    def test_random_order_and_spreads_beyond_prefix(self):
        """Contract changed deliberately: this used to assert input
        (file_path) order was preserved. That order made every PREFIX of a
        generation run a biased subset -- on featbit v5.4.9 the first 64% of a
        symbol-free run held zero TypeScript queries. It was harmless while
        runs could not be interrupted; checkpointing made stopping early a real
        option. See tests/unit/test_sample_entities_order.py.
        """
        from treeloom.application.benchmark.enhanced_query_gen import sample_entities

        ents = self._entities(1000)
        picked = sample_entities(ents, 50, seed=42)
        ids = [e["id"] for e in picked]
        assert ids != sorted(ids)  # seeded random order, not file_path order
        # Unchanged guard: a LIMIT-slice would give ids 0..49; a sample must
        # reach past the prefix.
        assert max(ids) > 49


class TestOffDomainPrescreen:
    """v1 off-domain-term pre-screen (lemma-normalized)."""

    def _vocab(self):
        from treeloom.domain.benchmark.query_hygiene import build_vocab_lemmas

        return build_vocab_lemmas(
            ["wrap", "wrapped", "policy", "authorization", "flag", "toggle",
             "workspace", "listed", "sandbox", "lease"]
        )

    def test_lemma_strips_gerund_and_collapses_doubling(self):
        from treeloom.domain.benchmark.query_hygiene import lemma

        assert lemma("wrapping") == "wrap"
        assert lemma("policies") == "policy"
        assert lemma("listing") == "list"
        assert lemma("falls") == "fall"  # no false doubling collapse

    def test_true_positive_off_domain_noun_flagged(self):
        from treeloom.domain.benchmark.query_hygiene import off_domain_terms

        flags = off_domain_terms(
            "How do I retrieve insurance policies for a workspace?", self._vocab()
        )
        assert flags == ["insurance"]

    def test_gerund_of_vocab_word_not_flagged(self):
        from treeloom.domain.benchmark.query_hygiene import off_domain_terms

        assert off_domain_terms(
            "function wrapping a listed toggle in a sandbox lease", self._vocab()
        ) == []

    def test_generic_words_never_flagged(self):
        from treeloom.domain.benchmark.query_hygiene import off_domain_terms

        assert off_domain_terms(
            "how does returning multiple values determine the specific result",
            frozenset(),
        ) == []
