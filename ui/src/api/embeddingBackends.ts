import { api } from "@/api/client";

/**
 * Embedding-backend registry. Admin-only CRUD over the Postgres
 * `embedding_backends` table.
 *
 * - `GET  /embedding-backends` → { registry: BackendRow[], local, process_id }
 * - `POST /embedding-backends` (201) → BackendRow   ({ url, klass })
 * - `DELETE /embedding-backends/{id}` (204)
 */

/** One registry row (from `BackendRow.to_dict()`). */
export interface BackendRow {
  id: string;
  url: string;
  klass: string; // "gpu" | "cpu"
  enabled: boolean;
  created_at: string;
  created_by: string | null;
}

/** Per-URL live counters for the responding worker process. */
export interface BackendLocalCounters {
  in_flight_tokens: number;
  failures: number;
}

/** The `GET /embedding-backends` response. */
export interface BackendListResponse {
  registry: BackendRow[];
  local: Record<string, BackendLocalCounters>;
  process_id: string;
}

/** Create-backend request body (matches `AddBackendRequest`). */
export interface AddBackendBody {
  url: string;
  klass: string; // "gpu" | "cpu"
}

// --------------------------------------------------------------------------
// Fetch functions
// --------------------------------------------------------------------------

export function fetchBackends(): Promise<BackendListResponse> {
  return api.get<BackendListResponse>("/embedding-backends");
}

export function addBackend(body: AddBackendBody): Promise<BackendRow> {
  return api.post<BackendRow>("/embedding-backends", body);
}

export function deleteBackend(id: string): Promise<void> {
  return api.del<void>(`/embedding-backends/${encodeURIComponent(id)}`);
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/** Validate the add-backend form; returns an error string or null. */
export function validateBackend(url: string, klass: string): string | null {
  const u = url.trim();
  if (!u) return "Enter a backend URL.";
  if (!(u.startsWith("http://") || u.startsWith("https://"))) {
    return "URL must start with http:// or https://";
  }
  if (klass !== "gpu" && klass !== "cpu") return "Class must be gpu or cpu.";
  return null;
}
