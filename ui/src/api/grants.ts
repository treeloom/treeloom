import { api } from "@/api/client";

/**
 * Per-source access grants (— access-grants UI).
 *
 * Admin-only CRUD over the `source_grants` ACL table:
 * - `GET    /sources/{id}/grants` → SourceGrant[]
 * - `POST   /sources/{id}/grants`  (201) ← AddGrantBody
 * - `DELETE /sources/{id}/grants/{principal_type}/{principal_id}` (204)
 *
 * Principal labels are resolved client-side against `GET /users` and
 * `GET /groups` (both admin-only, may return empty lists).
 */

export type PrincipalType = "user" | "group";
export type GrantEffect = "allow" | "deny";

/** One ACL row, as returned by `GET /sources/{id}/grants`. */
export interface SourceGrant {
  principal_type: PrincipalType;
  principal_id: string;
  effect: GrantEffect;
}

/** Create-grant request body (matches the POST endpoint). */
export interface AddGrantBody {
  principal_type: PrincipalType;
  principal_id: string;
  effect: GrantEffect;
}

/** One user, as returned by `GET /users`. */
export interface UserSummary {
  id: string;
  username: string;
  role: string;
}

/** One group, as returned by `GET /groups`. */
export interface GroupSummary {
  id: string;
  name: string;
}

// --------------------------------------------------------------------------
// Fetch functions
// --------------------------------------------------------------------------

/** Fetch the grants for one source. */
export function fetchGrants(sourceId: string): Promise<SourceGrant[]> {
  return api.get<SourceGrant[]>(
    `/sources/${encodeURIComponent(sourceId)}/grants`,
  );
}

/** Create (or upsert) a grant for one source. */
export function addGrant(
  sourceId: string,
  body: AddGrantBody,
): Promise<SourceGrant> {
  return api.post<SourceGrant>(
    `/sources/${encodeURIComponent(sourceId)}/grants`,
    body,
  );
}

/** Revoke a grant by principal. */
export function deleteGrant(
  sourceId: string,
  principalType: PrincipalType,
  principalId: string,
): Promise<void> {
  return api.del<void>(
    `/sources/${encodeURIComponent(sourceId)}/grants/${encodeURIComponent(
      principalType,
    )}/${encodeURIComponent(principalId)}`,
  );
}

/** Fetch the user directory (admin-only). */
export function fetchUsers(): Promise<UserSummary[]> {
  return api.get<UserSummary[]>("/users");
}

/** Fetch the group directory (admin-only). */
export function fetchGroups(): Promise<GroupSummary[]> {
  return api.get<GroupSummary[]>("/groups");
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/**
 * Human label for a grant's principal: resolve the id to a username (user) or
 * group name (group) from the fetched directories. Falls back to the raw id
 * when the principal is unknown (e.g. a deleted user still holding a grant).
 */
export function principalLabel(
  principalType: PrincipalType,
  principalId: string,
  users: UserSummary[],
  groups: GroupSummary[],
): string {
  if (principalType === "user") {
    const u = users.find((x) => x.id === principalId);
    return u ? u.username : principalId;
  }
  const g = groups.find((x) => x.id === principalId);
  return g ? g.name : principalId;
}

/**
 * Validate an add-grant request before POSTing. Returns an error string, or
 * null when the request is well-formed.
 */
export function validateGrant(
  principalType: string,
  principalId: string,
  effect: string,
): string | null {
  if (principalType !== "user" && principalType !== "group") {
    return "Choose a principal type.";
  }
  if (!principalId.trim()) {
    return principalType === "user"
      ? "Choose a user."
      : "Choose a group.";
  }
  if (effect !== "allow" && effect !== "deny") {
    return "Choose allow or deny.";
  }
  return null;
}

/**
 * True when a grant for this principal already exists — used to disable the
 * picker option / prevent a duplicate POST. Effect is ignored (one grant per
 * principal per source).
 */
export function hasGrant(
  grants: SourceGrant[],
  principalType: PrincipalType,
  principalId: string,
): boolean {
  return grants.some(
    (g) =>
      g.principal_type === principalType && g.principal_id === principalId,
  );
}
