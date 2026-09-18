"""Unit tests for the Scope enum and scopes_for_role helper.

Detroit-style: exercises the real domain objects, no mocks needed.
"""

from treeloom.domain.authorization import Scope, Role, scopes_for_role


class TestScopeEnum:
    """Scope enum has the expected values."""

    def test_search_value(self):
        assert Scope.SEARCH.value == "search"

    def test_index_value(self):
        assert Scope.INDEX.value == "index"

    def test_admin_value(self):
        assert Scope.ADMIN.value == "admin"

    def test_scope_is_string_enum(self):
        assert isinstance(Scope.SEARCH, str)
        assert Scope.SEARCH == "search"


class TestScopesForRole:
    """scopes_for_role returns the correct scope sets."""

    def test_admin_gets_all_scopes(self):
        scopes = scopes_for_role(Role.ADMIN)
        assert Scope.SEARCH in scopes
        assert Scope.INDEX in scopes
        assert Scope.ADMIN in scopes

    def test_user_gets_search_and_index(self):
        scopes = scopes_for_role(Role.USER)
        assert Scope.SEARCH in scopes
        assert Scope.INDEX in scopes

    def test_user_does_not_get_admin_scope(self):
        scopes = scopes_for_role(Role.USER)
        assert Scope.ADMIN not in scopes

    def test_admin_scope_count(self):
        assert len(scopes_for_role(Role.ADMIN)) == 3

    def test_user_scope_count(self):
        assert len(scopes_for_role(Role.USER)) == 2
