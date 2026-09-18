import { api } from "@/api/client";

/**
 * Job submission against the indexer `/index-*` endpoints.
 *
 * Three kinds, each its own endpoint + request body:
 * - repo: `POST /index-repo`   ({ path } OR { url, branch? })
 * - directory: `POST /index-directory` ({ directory, pattern? })
 * - file: `POST /index-file`   ({ file_path })
 *
 * All accept `force` + `skip_graph`. The response is a {@link JobAck} carrying
 * a per-Task `job_id` (NOT a job-group id — do NOT deep-link to /jobs/:id).
 */

export type SubmitKind = "repo" | "directory" | "file";

/** Repo source: exactly one of `path` (local) or `url` (remote). */
export type RepoTarget =
  | { mode: "path"; path: string }
  | { mode: "url"; url: string; branch: string };

/** Raw form state, as edited in the UI. */
export interface SubmitFormState {
  kind: SubmitKind;
  /** repo-path / file-path / directory path field (shared text input). */
  path: string;
  /** repo url (when repo + url mode). */
  url: string;
  /** repo branch (optional, url mode). */
  branch: string;
  /** directory glob, defaults to "**\/*" server-side when blank. */
  pattern: string;
  /** repo source toggle: index from a local path or a git url. */
  repoMode: "path" | "url";
  force: boolean;
  skipGraph: boolean;
}

/** A fresh, empty form. */
export function emptyForm(): SubmitFormState {
  return {
    kind: "repo",
    path: "",
    url: "",
    branch: "",
    pattern: "",
    repoMode: "path",
    force: false,
    skipGraph: false,
  };
}

/** The JobAck response model (job_id, status, source_id, optional advice). */
export interface JobAck {
  job_id: string;
  status: string;
  source_id: string;
  warnings?: Array<Record<string, unknown>> | null;
  recommendations?: Record<string, unknown> | null;
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/**
 * Validate a form. Returns an error message string when invalid, else null.
 * Checks only the fields relevant to the selected `kind`/`repoMode`.
 */
export function validateForm(f: SubmitFormState): string | null {
  if (f.kind === "repo") {
    if (f.repoMode === "path") {
      if (!f.path.trim()) return "Enter a repository path.";
    } else {
      if (!f.url.trim()) return "Enter a repository URL.";
    }
    return null;
  }
  if (f.kind === "directory") {
    if (!f.path.trim()) return "Enter a directory path.";
    return null;
  }
  // file
  if (!f.path.trim()) return "Enter a file path.";
  return null;
}

/** The endpoint path for a given kind. */
export function endpointFor(kind: SubmitKind): string {
  switch (kind) {
    case "repo":
      return "/index-repo";
    case "directory":
      return "/index-directory";
    case "file":
      return "/index-file";
  }
}

/**
 * Build the request body for a (validated) form. Only emits the fields the
 * matching backend model accepts, omitting blanks so server defaults apply.
 */
export function buildBody(f: SubmitFormState): Record<string, unknown> {
  const common = { force: f.force, skip_graph: f.skipGraph };
  if (f.kind === "repo") {
    if (f.repoMode === "url") {
      const body: Record<string, unknown> = { url: f.url.trim(), ...common };
      if (f.branch.trim()) body.branch = f.branch.trim();
      return body;
    }
    return { path: f.path.trim(), ...common };
  }
  if (f.kind === "directory") {
    const body: Record<string, unknown> = {
      directory: f.path.trim(),
      ...common,
    };
    if (f.pattern.trim()) body.pattern = f.pattern.trim();
    return body;
  }
  // file
  return { file_path: f.path.trim(), ...common };
}

// --------------------------------------------------------------------------
// Fetch function
// --------------------------------------------------------------------------

/** Submit an indexing job; resolves with the {@link JobAck} (HTTP 202). */
export function submitJob(f: SubmitFormState): Promise<JobAck> {
  return api.post<JobAck>(endpointFor(f.kind), buildBody(f));
}

// --------------------------------------------------------------------------
// Multi-repo job submission
// --------------------------------------------------------------------------

/**
 * One parsed repo entry from the multi-line textarea. Exactly one of a remote
 * `url` (with optional `branch`) or a local `path` — mirrors `ManifestEntry`.
 */
export type RepoEntry =
  | { url: string; branch?: string }
  | { path: string };

/** Options that apply to a whole multi-repo Job (no per-entry options in v1). */
export interface RepoSubmitOptions {
  force: boolean;
  skipGraph: boolean;
  groupId: string;
}

/** A line whose first token is one of these schemes is treated as a git URL. */
const URL_RE = /^(https?|git|ssh):\/\//i;

/** True when a (trimmed) line denotes a git URL rather than a local path. */
function isUrlLine(line: string): boolean {
  const first = line.split(/\s+/, 1)[0] ?? "";
  return URL_RE.test(first) || first.startsWith("git@");
}

/**
 * Parse the repos textarea into {@link RepoEntry}[] (pure — see spec §4).
 *
 * - Split on newlines; trim each line; drop blanks and `#` comments.
 * - URL line (first token matches a scheme or `git@`): first whitespace token is
 *   the `url`, an optional second token is the `branch`.
 * - Otherwise a path entry: the WHOLE trimmed line is the `path` (paths may
 *   contain spaces, so we do not split — branches are URL-only).
 * - Deduplicate identical entries (same url+branch, or same path).
 */
export function parseRepoLines(text: string): RepoEntry[] {
  const out: RepoEntry[] = [];
  const seen = new Set<string>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;

    let entry: RepoEntry;
    if (isUrlLine(line)) {
      const tokens = line.split(/\s+/);
      const url = tokens[0];
      const branch = tokens[1];
      entry = branch ? { url, branch } : { url };
    } else {
      entry = { path: line };
    }

    const key =
      "url" in entry ? `url\0${entry.url}\0${entry.branch ?? ""}` : `path\0${entry.path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(entry);
  }
  return out;
}

/**
 * Build the `/index-repo` body for one {@link RepoEntry} + Job options. The body
 * is `{path}` XOR `{url, branch?}`, plus `force`, `skip_graph`, and `group_id`.
 */
export function buildRepoEntryBody(
  entry: RepoEntry,
  opts: RepoSubmitOptions,
): Record<string, unknown> {
  const common = {
    force: opts.force,
    skip_graph: opts.skipGraph,
    group_id: opts.groupId,
  };
  if ("url" in entry) {
    const body: Record<string, unknown> = { url: entry.url, ...common };
    if (entry.branch) body.branch = entry.branch;
    return body;
  }
  return { path: entry.path, ...common };
}

/** Create a job group and return its id. `POST /job-groups {label, kind}`. */
export function createJobGroup(
  label: string,
  kind = "manual",
): Promise<{ id: string }> {
  return api.post<{ id: string }>("/job-groups", { label, kind });
}

/** The outcome of submitting a single repo entry within a multi-repo Job. */
export interface EntryResult {
  entry: RepoEntry;
  /** `queued` | `done` (already indexed) | `failed`. */
  outcome: "queued" | "done" | "failed";
  /** The JobAck on success. */
  ack?: JobAck;
  /** The error message when `outcome === "failed"`. */
  error?: string;
}

/** The result of a whole multi-repo submission. */
export interface MultiRepoResult {
  groupId: string;
  results: EntryResult[];
}

/** A short human label for a repo entry (its url or path). */
export function entryLabel(entry: RepoEntry): string {
  return "url" in entry ? entry.url : entry.path;
}

/** The basename of a url or path, used to default a single-entry Job label. */
function basename(s: string): string {
  const cleaned = s.replace(/[/\\]+$/, "").replace(/\.git$/i, "");
  const parts = cleaned.split(/[/\\]/);
  return parts[parts.length - 1] || cleaned;
}

/**
 * Default Job label: the single repo's name for one entry, else "N repositories".
 * Used when the operator leaves the label field blank.
 */
export function defaultJobLabel(entries: RepoEntry[]): string {
  if (entries.length === 1) return basename(entryLabel(entries[0]));
  return `${entries.length} repositories`;
}

/**
 * Orchestrate a multi-repo Job (spec §3):
 *   1) `POST /job-groups` → group_id (throws on failure — caller aborts).
 *   2) sequentially `POST /index-repo` per entry with the shared group_id,
 *      recording each per-entry outcome (queued | done | failed); per-entry
 *      failures are non-fatal.
 */
export async function submitMultiRepo(
  entries: RepoEntry[],
  opts: { label: string; force: boolean; skipGraph: boolean },
): Promise<MultiRepoResult> {
  const group = await createJobGroup(opts.label);
  const groupId = group.id;
  const results: EntryResult[] = [];
  for (const entry of entries) {
    const body = buildRepoEntryBody(entry, {
      force: opts.force,
      skipGraph: opts.skipGraph,
      groupId,
    });
    try {
      const ack = await api.post<JobAck>("/index-repo", body);
      results.push({
        entry,
        outcome: ack.status === "done" ? "done" : "queued",
        ack,
      });
    } catch (err) {
      const error =
        err instanceof Error ? err.message : "Submission failed.";
      results.push({ entry, outcome: "failed", error });
    }
  }
  return { groupId, results };
}
