import { api } from "@/api/client";

/**
 * The catalog of indexed sources, served by the public `GET /sources` endpoint.
 *
 * Sources catalog tab.
 */

/** One indexed source, as returned by `GET /sources`. */
export interface SourceRecord {
  id: string;
  path: string; // "" for url-only sources
  url: string; // "" for local-path sources
  branch: string;
  indexed_at: number; // UNIX epoch seconds (int)
  file_count: number;
  chunk_count: number;
  commit_sha: string;
  graph_indexed: boolean;
}

/**
 * Result of `GET /sources/{id}/staleness` — a LIVE git call server-side, so it
 * is fetched LAZILY (opt-in per row), never on catalog load. `is_stale` is
 * `null` for non-git sources.
 */
export interface SourceStaleness {
  indexed_sha: string;
  current_sha: string;
  is_stale: boolean | null;
}

// --------------------------------------------------------------------------
// Fetch functions
// --------------------------------------------------------------------------

/** Fetch the catalog of indexed sources (public endpoint). */
export function fetchSources(): Promise<SourceRecord[]> {
  return api.get<SourceRecord[]>("/sources");
}

/**
 * Fetch staleness for one source. Triggers a live git HEAD resolution
 * server-side — only call this on an explicit user action, never on load.
 */
export function fetchStaleness(id: string): Promise<SourceStaleness> {
  return api.get<SourceStaleness>(
    `/sources/${encodeURIComponent(id)}/staleness`,
  );
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/**
 * Best human label for a source: prefer `path`, else `url`, else `id`.
 * (`branch` is rendered separately so filtering can still match it.)
 */
export function sourceLabel(src: SourceRecord): string {
  return src.path || src.url || src.id || "(unknown)";
}

/**
 * Case-insensitive client-side filter over a source's path, url, and branch.
 * An empty / whitespace-only query returns the input unchanged (all rows).
 */
export function filterSources(
  sources: SourceRecord[],
  query: string,
): SourceRecord[] {
  const q = query.trim().toLowerCase();
  if (q === "") return sources;
  return sources.filter((s) => {
    const hay = `${s.path} ${s.url} ${s.branch}`.toLowerCase();
    return hay.includes(q);
  });
}

/**
 * Human label for a staleness result: `null` → "n/a" (non-git source),
 * `is_stale === true` → "stale", else "fresh".
 */
export function stalenessLabel(s: SourceStaleness): string {
  if (s.is_stale === null) return "n/a";
  return s.is_stale ? "stale" : "fresh";
}
