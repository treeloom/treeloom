import { api } from "@/api/client";

/**
 * Users & Groups admin:
 * - Users: `/users` (GET/POST/DELETE) + `/users/{id}/role`, `/users/{id}`,
 *   `/users/{id}/set-password`, `/users/{id}/rotate-key`.
 * - Groups: `/groups` (GET/POST/DELETE) + `/groups/{id}` and
 *   `/groups/{id}/members` (GET/POST/DELETE).
 *
 * All endpoints are admin-only (403 for non-admins) when AUTH_ENABLED.
 * `POST /users` and `POST /users/{id}/rotate-key` return a raw `api_key` ONCE —
 * surface it via {@link SecretReveal}, never recoverable afterwards.
 */

/** Roles a user can hold. */
export const USER_ROLES = ["user", "admin"] as const;
export type UserRole = (typeof USER_ROLES)[number];

/** A user row (from `GET /users`). */
export interface UserRow {
  id: string;
  username: string;
  email: string | null;
  role: UserRole;
  active: boolean;
  all_access?: boolean;
  created_at: string;
}

/** User create response — carries the once-shown raw `api_key`. */
export interface UserCreated extends UserRow {
  api_key: string;
}

/** Rotate-key response — carries the once-shown raw `api_key`. */
export interface KeyRotated {
  api_key: string;
}

/** A group row (from `GET /groups`). */
export interface GroupRow {
  id: string;
  name: string;
  all_access: boolean;
  created_at: string;
}

/** User create body (matches the server's create payload). */
export interface CreateUserBody {
  username: string;
  email?: string;
  role?: UserRole;
}

/** Group create body. */
export interface CreateGroupBody {
  name: string;
  all_access?: boolean;
}

// --------------------------------------------------------------------------
// User fetch functions
// --------------------------------------------------------------------------

export function fetchUsers(): Promise<UserRow[]> {
  return api.get<UserRow[]>("/users");
}

export function createUser(body: CreateUserBody): Promise<UserCreated> {
  return api.post<UserCreated>("/users", body);
}

export function deactivateUser(userId: string): Promise<void> {
  return api.del<void>(`/users/${encodeURIComponent(userId)}`);
}

export function setUserRole(userId: string, role: UserRole): Promise<void> {
  return api.put<void>(`/users/${encodeURIComponent(userId)}/role`, { role });
}

export function setUserAllAccess(
  userId: string,
  allAccess: boolean,
): Promise<void> {
  return api.patch<void>(`/users/${encodeURIComponent(userId)}`, {
    all_access: allAccess,
  });
}

export function setUserPassword(
  userId: string,
  password: string,
): Promise<void> {
  return api.post<void>(`/users/${encodeURIComponent(userId)}/set-password`, {
    password,
  });
}

export function rotateUserKey(userId: string): Promise<KeyRotated> {
  return api.post<KeyRotated>(
    `/users/${encodeURIComponent(userId)}/rotate-key`,
  );
}

// --------------------------------------------------------------------------
// Group fetch functions
// --------------------------------------------------------------------------

export function fetchGroups(): Promise<GroupRow[]> {
  return api.get<GroupRow[]>("/groups");
}

export function createGroup(body: CreateGroupBody): Promise<GroupRow> {
  return api.post<GroupRow>("/groups", body);
}

export function deleteGroup(groupId: string): Promise<void> {
  return api.del<void>(`/groups/${encodeURIComponent(groupId)}`);
}

export function setGroupAllAccess(
  groupId: string,
  allAccess: boolean,
): Promise<void> {
  return api.patch<void>(`/groups/${encodeURIComponent(groupId)}`, {
    all_access: allAccess,
  });
}

/** GET /groups/{id}/members returns `{group_id, members: string[]}` — unwrap to
 * the member-id array the UI expects (a bare list). */
export async function fetchGroupMembers(groupId: string): Promise<string[]> {
  const resp = await api.get<{ group_id: string; members: string[] }>(
    `/groups/${encodeURIComponent(groupId)}/members`,
  );
  return resp?.members ?? [];
}

export function addGroupMember(
  groupId: string,
  userId: string,
): Promise<void> {
  return api.post<void>(
    `/groups/${encodeURIComponent(groupId)}/members`,
    { user_id: userId },
  );
}

export function removeGroupMember(
  groupId: string,
  userId: string,
): Promise<void> {
  return api.del<void>(
    `/groups/${encodeURIComponent(groupId)}/members/${encodeURIComponent(userId)}`,
  );
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/** Validate a username; returns an error string or null. */
export function validateUsername(name: string): string | null {
  if (!name.trim()) return "Enter a username.";
  return null;
}

/** Validate a new password; returns an error string or null. */
export function validatePassword(password: string): string | null {
  if (password.length < 8) return "Use at least 8 characters.";
  return null;
}

/** Validate a group name; returns an error string or null. */
export function validateGroupName(name: string): string | null {
  if (!name.trim()) return "Enter a group name.";
  return null;
}

/** Normalize the `role` value, defaulting unknown values to "user". */
export function normalizeRole(role: string): UserRole {
  return role === "admin" ? "admin" : "user";
}

/** Build the `POST /users` body, omitting a blank email. */
export function buildCreateUserBody(
  username: string,
  email: string,
  role: UserRole,
): CreateUserBody {
  const body: CreateUserBody = { username: username.trim() };
  const trimmedEmail = email.trim();
  if (trimmedEmail) body.email = trimmedEmail;
  if (role === "admin") body.role = "admin";
  else body.role = "user";
  return body;
}
