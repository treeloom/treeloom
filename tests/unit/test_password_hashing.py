"""Detroit-style unit tests for password hashing helpers.

hash_password/verify_password are pure functions — no mocks needed.
"""

from treeloom.adapters.authorization.user_store import hash_password, verify_password


class TestHashPasswordVerifyPasswordRoundTrip:
    """hash_password + verify_password round-trip correctly."""

    def test_round_trip(self):
        pw = "correct-horse-battery-staple"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_hash_is_not_plaintext(self):
        pw = "my-secret"
        hashed = hash_password(pw)
        assert hashed != pw

    def test_same_password_produces_different_hashes(self):
        """bcrypt generates a random salt per call."""
        pw = "same-password"
        h1 = hash_password(pw)
        h2 = hash_password(pw)
        assert h1 != h2

    def test_both_hashes_verify(self):
        """Two different hashes for the same password both verify."""
        pw = "same-password"
        h1 = hash_password(pw)
        h2 = hash_password(pw)
        assert verify_password(pw, h1) is True
        assert verify_password(pw, h2) is True


class TestVerifyPasswordWrongPassword:
    """Wrong password returns False."""

    def test_wrong_password_returns_false(self):
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_empty_password_returns_false(self):
        hashed = hash_password("real-password")
        assert verify_password("", hashed) is False


class TestVerifyPasswordMalformedHash:
    """Malformed or missing hash returns False without raising."""

    def test_empty_hash_returns_false(self):
        assert verify_password("any", "") is False

    def test_garbage_hash_returns_false(self):
        assert verify_password("any", "not-a-bcrypt-hash") is False

    def test_truncated_hash_returns_false(self):
        hashed = hash_password("pw")
        assert verify_password("pw", hashed[:10]) is False
