import { api } from "@/api/client";

/**
 * `GET /auth/me` — the current authenticated caller (user + scopes). Used by
 * admin-gated tabs (backends, api-keys) and to resolve the user id
 * for "My Tokens" (PATs).
 */

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  role: string; // "admin" | "user"
  active: boolean;
  created_at: string;
}

export interface AuthMe {
  user: AuthUser;
  scopes: string[];
  auth_method: string;
}

/** Fetch the current caller. 401 → ApiError (routes to the login gate). */
export function fetchAuthMe(): Promise<AuthMe> {
  return api.get<AuthMe>("/auth/me");
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/** True when the caller is an admin (role === "admin" or has the admin scope). */
export function isAdmin(me: AuthMe | undefined | null): boolean {
  if (!me) return false;
  if (me.user?.role === "admin") return true;
  return (me.scopes ?? []).includes("admin");
}

/** The caller's user id, or null when unknown. */
export function userId(me: AuthMe | undefined | null): string | null {
  return me?.user?.id ?? null;
}
