import { api } from "@/api/client";

/**
 * Access tokens:
 * - Personal Access Tokens (PATs): `/users/{user_id}/tokens` (GET/POST/DELETE).
 * - API keys (admin): `/api-keys` (GET/POST/DELETE).
 *
 * Both create endpoints return the raw secret ONCE — `token` for PATs,
 * `key` for API keys — never recoverable afterwards.
 */

/** Scopes selectable for a PAT / API key. */
export const TOKEN_SCOPES = ["search", "index", "admin"] as const;
export type TokenScope = (typeof TOKEN_SCOPES)[number];

/** A PAT row (from `_pat_to_response`). */
export interface PatRow {
  id: string;
  user_id: string;
  name: string;
  scopes: string[];
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  revoked: boolean;
}

/** PAT create response — extends the row with the once-shown `token`. */
export interface PatCreated extends PatRow {
  token: string;
}

/** An API-key row (from `_api_key_to_response`). */
export interface ApiKeyRow {
  id: string;
  name: string;
  scopes: string[];
  all_access: boolean;
  created_by: string | null;
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  revoked: boolean;
}

/** API-key create response — extends the row with the once-shown `key`. */
export interface ApiKeyCreated extends ApiKeyRow {
  key: string;
}

/** PAT create body (matches `CreateTokenRequest`). */
export interface CreatePatBody {
  name: string;
  scopes: string[];
  expires_at?: string | null;
}

/** API-key create body (matches `CreateApiKeyRequest`). */
export interface CreateApiKeyBody {
  name: string;
  scopes: string[];
  all_access: boolean;
  expires_at?: string | null;
}

// --------------------------------------------------------------------------
// PAT fetch functions
// --------------------------------------------------------------------------

export function fetchTokens(userId: string): Promise<PatRow[]> {
  return api.get<PatRow[]>(`/users/${encodeURIComponent(userId)}/tokens`);
}

export function createToken(
  userId: string,
  body: CreatePatBody,
): Promise<PatCreated> {
  return api.post<PatCreated>(
    `/users/${encodeURIComponent(userId)}/tokens`,
    body,
  );
}

export function revokeToken(userId: string, tokenId: string): Promise<void> {
  return api.del<void>(
    `/users/${encodeURIComponent(userId)}/tokens/${encodeURIComponent(tokenId)}`,
  );
}

// --------------------------------------------------------------------------
// API-key fetch functions
// --------------------------------------------------------------------------

export function fetchApiKeys(): Promise<ApiKeyRow[]> {
  return api.get<ApiKeyRow[]>("/api-keys");
}

export function createApiKey(body: CreateApiKeyBody): Promise<ApiKeyCreated> {
  return api.post<ApiKeyCreated>("/api-keys", body);
}

export function revokeApiKey(keyId: string): Promise<void> {
  return api.del<void>(`/api-keys/${encodeURIComponent(keyId)}`);
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/** Validate a token/key name; returns an error string or null. */
export function validateTokenName(name: string): string | null {
  if (!name.trim()) return "Enter a name.";
  return null;
}

/**
 * Toggle a scope in/out of a list, returning a new sorted-stable array
 * (preserving the canonical {@link TOKEN_SCOPES} order).
 */
export function toggleScope(scopes: string[], scope: string): string[] {
  const has = scopes.includes(scope);
  const next = has ? scopes.filter((s) => s !== scope) : [...scopes, scope];
  return TOKEN_SCOPES.filter((s) => next.includes(s));
}
