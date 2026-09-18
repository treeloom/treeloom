"""Authorization bounded context.

Domain ports, entities, and value objects for authentication and authorization.
The domain defines what a User, Role, and Permission are; adapters implement
the storage and validation contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


# ── Value objects ──────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"


class Permission(str, Enum):
    INDEX = "index"
    SEARCH = "search"
    ADMIN = "admin"  # User management, system config


class Scope(str, Enum):
    """Fine-grained token scopes for PATs and API keys."""

    SEARCH = "search"
    INDEX = "index"
    ADMIN = "admin"


def scopes_for_role(role: Role) -> set[Scope]:
    """Return the set of scopes that a user with the given role may hold.

    Admin users may hold all scopes; regular users hold SEARCH and INDEX.
    """
    if role == Role.ADMIN:
        return {Scope.SEARCH, Scope.INDEX, Scope.ADMIN}
    return {Scope.SEARCH, Scope.INDEX}


# ── Entity ─────────────────────────────────────────────────────────────


@dataclass
class User:
    """An authenticated user with an API key and role.

    api_key_hash is stored, never the raw key. The raw key is generated
    once at creation time and returned to the caller — it cannot be
    recovered later.

    password_hash is populated for users with local login credentials.
    A null/empty value means the user can only authenticate via tokens
    (e.g. token-only users).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    username: str = ""
    email: str = ""
    api_key_hash: str = ""
    role: Role = Role.USER
    active: bool = True
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Effective authorization read-model, populated at authentication time.
    # `all_access` is the OR of the user's own users.all_access flag and the
    # all_access flag of any group they belong to (see PostgreSQLUserStore
    # hydration). `group_ids` is their group membership. Both default empty so
    # an unhydrated User (e.g. in older tests) authorizes only what it owns.
    all_access: bool = False
    group_ids: list[str] = field(default_factory=list)
    # Local-login password hash (bcrypt). Empty string = no local login.
    password_hash: str = ""

    def has_permission(self, permission: Permission) -> bool:
        """Check whether this user holds the given permission."""
        if self.role == Role.ADMIN:
            return True
        if permission == Permission.ADMIN:
            return False
        # Regular users have INDEX and SEARCH
        return permission in (Permission.INDEX, Permission.SEARCH)


@dataclass
class Group:
    """A named authorization principal that users belong to.

    `all_access` marks a group whose members may query the shared/cross-repo
    index (subject to explicit deny grants).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    all_access: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PersonalAccessToken:
    """A user-owned API token with per-token scopes.

    Scopes are stored as a list of strings (values from Scope enum).
    token_hash is an HMAC-SHA256 digest of the raw token; the raw token
    is returned once at creation time.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = ""
    name: str = ""
    token_hash: str = ""
    scopes: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked: bool = False


@dataclass
class ApiKey:
    """A service/admin-owned API key with per-token scopes.

    all_access mirrors the users.all_access flag: when True the key
    may run shared/cross-repo searches.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    token_hash: str = ""
    scopes: list[str] = field(default_factory=list)
    all_access: bool = False
    created_by: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked: bool = False


@dataclass
class Session:
    """A DB-backed browser session issued on successful /auth/login.

    token_hash is an HMAC-SHA256 digest of the raw cookie value.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = ""
    token_hash: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: Optional[datetime] = None


# ── Ports ──────────────────────────────────────────────────────────────


class UserStorePort(ABC):
    """Persistence contract for user accounts."""

    @abstractmethod
    async def create_user(
        self,
        username: str,
        email: str = "",
        role: Role = Role.USER,
        api_key_hash: str = "",
    ) -> Optional[User]:
        """Create a new user and return the User entity, or None on failure."""
        ...

    @abstractmethod
    async def get_by_api_key_hash(self, key_hash: str) -> Optional[User]:
        """Look up a user by their hashed API key."""
        ...

    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Look up a user by their id."""
        ...

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Look up a user by username (used to map external identities)."""
        ...

    @abstractmethod
    async def list_users(self) -> list[User]:
        """Return all users."""
        ...

    @abstractmethod
    async def update_role(self, user_id: str, role: Role) -> bool:
        """Update a user's role. Returns True if the user was found."""
        ...

    @abstractmethod
    async def update_api_key_hash(self, user_id: str, new_hash: str) -> bool:
        """Rotate a user's API key hash. Returns True if user found."""
        ...

    @abstractmethod
    async def update_all_access(self, user_id: str, all_access: bool) -> bool:
        """Set a user's all_access flag. Returns True if the user was found."""
        ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Soft-delete a user (set active=False). Returns True if found."""
        ...

    @abstractmethod
    async def count_users(self) -> int:
        """Return the total number of active users. Used for seed detection."""
        ...

    @abstractmethod
    async def update_password(self, user_id: str, password_hash: str) -> bool:
        """Update a user's bcrypt password hash. Returns True if user found."""
        ...


class AuthPort(ABC):
    """Authentication and authorization contract.

    Implementations validate credentials and enforce permissions.
    """

    @abstractmethod
    async def authenticate(self, api_key: str) -> Optional[User]:
        """Validate an API key and return the User if valid."""
        ...

    @abstractmethod
    async def authorize(self, user: User, permission: Permission) -> bool:
        """Check whether a user holds a specific permission."""
        ...


class GroupStorePort(ABC):
    """Persistence contract for authorization groups and membership."""

    @abstractmethod
    async def create_group(
        self, name: str, all_access: bool = False
    ) -> Optional["Group"]:
        """Create a group and return it, or None on failure/duplicate."""
        ...

    @abstractmethod
    async def get_by_id(self, group_id: str) -> Optional["Group"]:
        """Look up a group by id."""
        ...

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional["Group"]:
        """Look up a group by name (used to map external group claims)."""
        ...

    @abstractmethod
    async def list_groups(self) -> list["Group"]:
        """Return all groups."""
        ...

    @abstractmethod
    async def delete_group(self, group_id: str) -> bool:
        """Delete a group (and its memberships). Returns True if found."""
        ...

    @abstractmethod
    async def set_all_access(self, group_id: str, all_access: bool) -> bool:
        """Toggle a group's all_access flag. Returns True if found."""
        ...

    @abstractmethod
    async def add_member(self, group_id: str, user_id: str) -> bool:
        """Add a user to a group (idempotent). Returns True on success."""
        ...

    @abstractmethod
    async def remove_member(self, group_id: str, user_id: str) -> bool:
        """Remove a user from a group. Returns True if the row existed."""
        ...

    @abstractmethod
    async def list_members(self, group_id: str) -> list[str]:
        """Return the user ids in a group."""
        ...


class GrantStorePort(ABC):
    """Persistence contract for per-source access grants.

    A grant is (principal_type, principal_id, source_id, effect) where
    principal_type is 'user' or 'group' and effect is 'allow' or 'deny'.
    """

    @abstractmethod
    async def grant(
        self,
        principal_type: str,
        principal_id: str,
        source_id: str,
        effect: str,
    ) -> bool:
        """Upsert a grant. Returns True on success."""
        ...

    @abstractmethod
    async def revoke(
        self, principal_type: str, principal_id: str, source_id: str
    ) -> bool:
        """Remove a grant. Returns True if the row existed."""
        ...

    @abstractmethod
    async def list_for_source(self, source_id: str) -> list[dict]:
        """Return all grants on a source (for admin display)."""
        ...

    @abstractmethod
    async def effects_for_source(
        self, principals: list[tuple[str, str]], source_id: str
    ) -> set[str]:
        """Return the set of effects ('allow'/'deny') any of these principals
        holds on the given source."""
        ...

    @abstractmethod
    async def denied_sources(
        self, principals: list[tuple[str, str]]
    ) -> list[str]:
        """Return source_ids these principals are explicitly denied."""
        ...


class TokenStorePort(ABC):
    """Persistence contract for user-owned Personal Access Tokens."""

    @abstractmethod
    async def create(
        self,
        user_id: str,
        name: str,
        token_hash: str,
        scopes: list[str],
        expires_at: Optional[datetime] = None,
    ) -> Optional[PersonalAccessToken]:
        """Create a new PAT and return it, or None on failure."""
        ...

    @abstractmethod
    async def get_by_token_hash(self, token_hash: str) -> Optional[PersonalAccessToken]:
        """Look up an active, non-expired, non-revoked PAT by its hash."""
        ...

    @abstractmethod
    async def list_for_user(self, user_id: str) -> list[PersonalAccessToken]:
        """Return all PATs for a user (including revoked)."""
        ...

    @abstractmethod
    async def revoke(self, token_id: str, user_id: str) -> bool:
        """Revoke a PAT scoped to its owner. Returns True if found."""
        ...

    @abstractmethod
    async def touch_last_used(self, token_id: str) -> None:
        """Update last_used_at to NOW() — best-effort, never raises."""
        ...


class ApiKeyStorePort(ABC):
    """Persistence contract for service/admin-owned API keys."""

    @abstractmethod
    async def create(
        self,
        name: str,
        token_hash: str,
        scopes: list[str],
        all_access: bool = False,
        created_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[ApiKey]:
        """Create a new API key and return it, or None on failure."""
        ...

    @abstractmethod
    async def get_by_token_hash(self, token_hash: str) -> Optional[ApiKey]:
        """Look up an active, non-expired, non-revoked API key by its hash."""
        ...

    @abstractmethod
    async def list_keys(self) -> list[ApiKey]:
        """Return all API keys (admin view)."""
        ...

    @abstractmethod
    async def revoke(self, key_id: str) -> bool:
        """Revoke an API key by id. Returns True if found."""
        ...

    @abstractmethod
    async def touch_last_used(self, key_id: str) -> None:
        """Update last_used_at to NOW() — best-effort, never raises."""
        ...


class SessionStorePort(ABC):
    """Persistence contract for DB-backed browser sessions."""

    @abstractmethod
    async def create(
        self,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> Optional[Session]:
        """Create a new session and return it, or None on failure."""
        ...

    @abstractmethod
    async def get_by_token_hash(self, token_hash: str) -> Optional[Session]:
        """Look up a non-expired session by its hash."""
        ...

    @abstractmethod
    async def delete(self, token_hash: str) -> bool:
        """Delete a session (logout). Returns True if a row was removed."""
        ...

    @abstractmethod
    async def touch(self, token_hash: str) -> None:
        """Update last_seen_at to NOW() — best-effort, never raises."""
        ...

    @abstractmethod
    async def delete_expired(self) -> int:
        """Delete all expired sessions and return the count removed."""
        ...

    @abstractmethod
    async def delete_for_user(self, user_id: str) -> int:
        """Delete every session belonging to a user (e.g. after a password
        change). Returns the count removed."""
        ...
