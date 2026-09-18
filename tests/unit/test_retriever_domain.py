"""Domain tests for retriever — scoring formulas, filter building, batch logic."""
import pytest
from treeloom.adapters.milvus.vector_store import _build_filter, _cosine, _name_in_query


# ── Filter Building ─────────────────────────────────────────────────────
#
# `_build_filter` returns (expression, params). Values that Milvus can bind
# server-side are bound; only the `like` operand is interpolated, because
# Milvus 2.5.4 cannot template one.
#
# The tests these replace asserted `"my_source" in f` — that the value was
# interpolated into the expression. They passed on the injectable version and
# would have kept passing on any variant of it, because interpolation was the
# thing they were checking for. The assertions below are the inverse.


def _expr(**kw):
    kw.setdefault("language", None)
    kw.setdefault("path_prefix", None)
    kw.setdefault("source_id", None)
    return _build_filter(**kw)


def test_build_filter_no_args():
    expr, params = _expr()
    assert expr == ""
    assert params == {}


def test_build_filter_language_is_bound_not_interpolated():
    expr, params = _expr(language="python")
    assert "language" in expr
    assert "python" not in expr, "the value must not reach the expression text"
    assert params == {"f_language": "python"}


def test_build_filter_source_id_is_bound_not_interpolated():
    expr, params = _expr(source_id="my_source")
    assert "source_id" in expr
    assert "my_source" not in expr
    assert params == {"f_source_id": "my_source"}


def test_build_filter_exclusions_are_bound_as_a_list():
    expr, params = _expr(exclude_source_ids=["a", "b"])
    assert "not in" in expr
    assert "a" not in expr.replace("f_exclude", "")
    assert params == {"f_exclude": ["a", "b"]}


def test_build_filter_path_prefix_is_interpolated_with_a_wildcard():
    """The one clause that must stay inline — Milvus cannot template `like`."""
    expr, params = _expr(path_prefix="/src/")
    assert 'file_path like "/src/%"' in expr
    assert params == {}


def test_build_filter_all_combined():
    expr, params = _expr(
        language="go", path_prefix="/pkg/", source_id="src1",
        exclude_source_ids=["denied"],
    )
    assert expr.count(" and ") == 3
    assert "/pkg/" in expr
    assert "go" not in expr.replace("f_language", "")
    assert "src1" not in expr
    assert params == {
        "f_language": "go", "f_source_id": "src1", "f_exclude": ["denied"],
    }


class TestFilterInjection:
    """CWE-943. `path_prefix` is caller-supplied on /search, and the
    `source_id not in [...]` clause is the per-repo ACL prefilter.

    With all four values interpolated unescaped, a prefix of
    `%" or source_id == "secret-repo` produced

        file_path like "%" or source_id == "secret-repo%"
          and source_id not in ["secret-repo"]

    and because `and` binds tighter than `or`, the left disjunct matched every
    row while the exclusion never applied. Confirmed against a live Milvus
    2.5.4 before the fix: the denied source's rows came back.
    """

    EVIL = '%" or source_id == "secret-repo'

    @staticmethod
    def _unescaped_quotes(expr: str) -> int:
        """Count quotes that actually delimit a literal, i.e. are not preceded
        by a backslash. A payload that broke out would raise this above the
        two that bound the `like` operand."""
        n = 0
        i = 0
        while i < len(expr):
            if expr[i] == "\\":
                i += 2
                continue
            if expr[i] == '"':
                n += 1
            i += 1
        return n

    def test_injected_prefix_cannot_break_out_of_its_literal(self):
        expr, _ = _expr(path_prefix=self.EVIL, exclude_source_ids=["secret-repo"])
        # The payload's ` or ` survives as TEXT inside the operand — that is
        # the point; it is data, not syntax. What matters is that its quotes
        # are escaped, so exactly two quotes still delimit the one literal.
        assert self._unescaped_quotes(expr) == 2
        assert ' or ' in expr, "the payload text is still there, inertly"

    def test_a_benign_prefix_has_the_same_quote_structure(self):
        """Control for the count above: two quotes is the normal shape, so the
        assertion is measuring escaping rather than an accident of the payload."""
        expr, _ = _expr(path_prefix="/srv/mine")
        assert self._unescaped_quotes(expr) == 2

    def test_the_acl_clause_survives_the_payload(self):
        expr, params = _expr(path_prefix=self.EVIL, exclude_source_ids=["secret-repo"])
        assert "(source_id not in {f_exclude})" in expr
        assert params["f_exclude"] == ["secret-repo"]

    def test_clauses_are_parenthesised_so_precedence_cannot_detach_the_acl(self):
        """Insurance beyond the escaping: even a future slip in the `like`
        operand cannot re-associate the conjunction away from the ACL."""
        expr, _ = _expr(language="python", path_prefix="/a/", source_id="s",
                        exclude_source_ids=["d"])
        for clause in expr.split(" and "):
            assert clause.startswith("(") and clause.endswith(")")

    def test_trailing_backslash_cannot_swallow_the_closing_quote(self):
        """`\\` is an escape character in Milvus literals, so escaping only the
        quote leaves `sid\\` rendering as `"sid\\"` — an unterminated string.
        Backslash is escaped first, which is why this holds."""
        from treeloom.adapters.milvus.vector_store import _escape_literal

        assert _escape_literal("sid\\") == "sid\\\\"
        assert _escape_literal('a"b') == 'a\\"b'
        assert _escape_literal('a\\"b') == 'a\\\\\\"b'

    def test_a_literal_percent_is_a_known_wildcard_not_an_escape(self):
        """Milvus `like` has no way to escape `%` (`\\%` is a parse error on
        2.5.4), so a `%` in a real path widens the match. Pinned as known
        behaviour: it cannot cross the ACL, because the exclusion is a
        separate templated conjunct, so it only widens a caller's query
        inside the scope they already hold."""
        expr, _ = _expr(path_prefix="/srv/x%y/")
        assert 'file_path like "/srv/x%y/%"' in expr


# ── Cosine Similarity ───────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = [0.5, 0.5, 0.5]
    sim = _cosine(v, v)
    assert abs(sim - 1.0) < 0.001


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    sim = _cosine(a, b)
    assert abs(sim - 0.0) < 0.001


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    sim = _cosine(a, b)
    assert abs(sim - (-1.0)) < 0.001


def test_cosine_zero_vector_returns_zero():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    sim = _cosine(a, b)
    assert sim == 0.0


def test_cosine_different_lengths():
    """_cosine returns 0.0 for different length vectors (no crash)."""
    a = [0.1, 0.2]
    b = [0.1, 0.2, 0.3]
    sim = _cosine(a, b)
    assert sim == 0.0


# ── Name in Query Scoring ───────────────────────────────────────────────

def test_name_in_query_exact_match():
    """Name-in-query is graded by identifier length (full weight at 16 chars).

    "SetTagsAsync" is 12 chars -> 12/16 = 0.75 with the default
    GRAPH_NAME_FULL_WEIGHT_LEN; a >=16-char identifier caps at 1.0.
    """
    entities = [{"name": "SetTagsAsync"}]
    score = _name_in_query("what does settagsasync do", entities)
    assert score == pytest.approx(0.75)

    long_entities = [{"name": "PostgresMessageConsumer"}]
    score = _name_in_query("where is postgresmessageconsumer defined", long_entities)
    assert score == 1.0


def test_name_in_query_no_match():
    entities = [{"name": "RenderPage"}]
    score = _name_in_query("where is set tags async", entities)
    assert score == 0.0


def test_name_in_query_partial_match():
    """Name is substring of query; space-split is NOT used."""
    entities = [{"name": "SetTagsAsync"}]
    score = _name_in_query("the SetTagsAsync method handles", entities)
    assert score > 0.0


def test_name_in_query_multiple_entities():
    entities = [
        {"name": "SetTagsAsync"},
        {"name": "GetTags"},
    ]
    score = _name_in_query("use gettags for retrieval", entities)
    assert score > 0.0


def test_name_in_query_empty_entities():
    score = _name_in_query("anything", [])
    assert score == 0.0
