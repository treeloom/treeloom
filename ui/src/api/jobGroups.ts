import { api } from "@/api/client";

/**
 * A "Job" in operator-UI terms is a **job group** — a batch of indexing tasks
 * (e.g. one fleet onboard from a manifest). The backend exposes them at
 * `/job-groups` (feed) and `/job-groups/{id}` (scoped detail with `tasks`).
 *
 * (feed) + (scoped overview).
 */

/** Per-status task tallies on a job group. */
export interface StatusCounts {
  queued: number;
  running: number;
  done: number;
  failed: number;
  dead_letter: number;
}

/** Aggregate file progress across a group's tasks. */
export interface GroupProgress {
  processed_files: number;
  total_files: number;
}

/** Rolled-up status of a whole job group. */
export type GroupStatus = "running" | "failed" | "done" | "queued";

/** Per-task status (one indexing job within a group). */
export type TaskStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "dead_letter";

/** A job-group summary row, as returned by `GET /job-groups` (newest-first). */
export interface JobGroupSummary {
  id: string;
  label: string;
  kind: string;
  created_at: number; // UNIX epoch seconds (float)
  created_by: string;
  task_count: number;
  status_counts: StatusCounts;
  progress: GroupProgress;
  status: GroupStatus;
}

/** A single indexing task within a group (`tasks[]` on the detail response). */
export interface JobTask {
  id: string;
  job_id: string;
  source: string;
  source_id: string;
  status: TaskStatus;
  kind: string;
  total_files: number;
  processed_files: number;
  committed_files: number;
  total_chunks: number;
  start_time: number | null;
  finished_at: number | null;
  error: string;
  message: string;
  current_file: string;
  source_path: string;
  source_url: string;
  source_branch: string;
  commit_sha: string;
  errors: number;
  payload: unknown;
  attempts: number;
  group_id: string;
}

/** The detail response: a summary plus its `tasks`. */
export interface JobGroupDetail extends JobGroupSummary {
  tasks: JobTask[];
}

// --------------------------------------------------------------------------
// Fetch functions
// --------------------------------------------------------------------------

/** Fetch the newest-first feed of job groups. */
export function fetchJobGroups(): Promise<JobGroupSummary[]> {
  return api.get<JobGroupSummary[]>("/job-groups");
}

/** Fetch one job group with its scoped tasks. 404 → ApiError(status 404). */
export function fetchJobGroup(id: string): Promise<JobGroupDetail> {
  return api.get<JobGroupDetail>(`/job-groups/${encodeURIComponent(id)}`);
}

// --------------------------------------------------------------------------
// Pure helpers (unit-tested)
// --------------------------------------------------------------------------

/**
 * Clamp `processed/total` to a 0–100 integer percentage. Guards `total <= 0`
 * (returns 0, never NaN) and never exceeds 100.
 */
export function percent(processed: number, total: number): number {
  if (!Number.isFinite(total) || total <= 0) return 0;
  if (!Number.isFinite(processed) || processed <= 0) return 0;
  const pct = (processed / total) * 100;
  if (pct >= 100) return 100;
  return Math.round(pct);
}

/** True when a group is actively indexing (running or queued tasks present). */
export function isGroupActive(group: { status: GroupStatus }): boolean {
  return group.status === "running";
}

/**
 * Overall live progress across ACTIVE groups only — sums `processed_files` and
 * `total_files` over groups whose `status === "running"`. Returns the summed
 * pair plus a clamped percentage. Inactive-only / empty input → all zeros.
 */
export function overallActiveProgress(groups: JobGroupSummary[]): {
  processed: number;
  total: number;
  percent: number;
} {
  let processed = 0;
  let total = 0;
  for (const g of groups) {
    if (g.status !== "running") continue;
    processed += g.progress?.processed_files ?? 0;
    total += g.progress?.total_files ?? 0;
  }
  return { processed, total, percent: percent(processed, total) };
}

/** Headline tally: number of Jobs (groups) in each rolled-up status. */
export function jobStatusCounts(groups: JobGroupSummary[]): {
  running: number;
  queued: number;
  done: number;
  failed: number;
} {
  const out = { running: 0, queued: 0, done: 0, failed: 0 };
  for (const g of groups) {
    if (g.status === "running") out.running += 1;
    else if (g.status === "queued") out.queued += 1;
    else if (g.status === "done") out.done += 1;
    else if (g.status === "failed") out.failed += 1;
  }
  return out;
}

/**
 * Bucket a group's tasks into in-progress (running/queued), done, and failed
 * (failed/dead_letter). Order within each bucket is preserved from input.
 */
export function taskBuckets(tasks: JobTask[]): {
  inprogress: JobTask[];
  done: JobTask[];
  failed: JobTask[];
} {
  const inprogress: JobTask[] = [];
  const done: JobTask[] = [];
  const failed: JobTask[] = [];
  for (const t of tasks) {
    if (t.status === "running" || t.status === "queued") inprogress.push(t);
    else if (t.status === "done") done.push(t);
    else if (t.status === "failed" || t.status === "dead_letter")
      failed.push(t);
  }
  return { inprogress, done, failed };
}

/** Status counts among a group's own tasks (for the detail header). */
export function taskStatusCounts(tasks: JobTask[]): {
  running: number;
  queued: number;
  done: number;
  failed: number;
} {
  const out = { running: 0, queued: 0, done: 0, failed: 0 };
  for (const t of tasks) {
    if (t.status === "running") out.running += 1;
    else if (t.status === "queued") out.queued += 1;
    else if (t.status === "done") out.done += 1;
    else if (t.status === "failed" || t.status === "dead_letter")
      out.failed += 1;
  }
  return out;
}

/** Best human label for a task: source > source_path > source_url > id. */
export function taskLabel(task: JobTask): string {
  return (
    task.source ||
    task.source_path ||
    task.source_url ||
    task.id ||
    task.job_id ||
    "(unknown)"
  );
}

/** First line of a (possibly multi-line) error string, trimmed. */
export function firstLine(text: string): string {
  if (!text) return "";
  const idx = text.indexOf("\n");
  return (idx === -1 ? text : text.slice(0, idx)).trim();
}

/**
 * Compact relative time, e.g. "just now", "2m ago", "3h ago", "5d ago".
 * `epochSeconds` is UNIX epoch seconds; `now` is millis (injectable for tests).
 * Future timestamps clamp to "just now".
 */
export function relativeTime(
  epochSeconds: number,
  now: number = Date.now(),
): string {
  if (!Number.isFinite(epochSeconds)) return "";
  const deltaSec = Math.floor(now / 1000 - epochSeconds);
  if (deltaSec < 5) return "just now";
  if (deltaSec < 60) return `${deltaSec}s ago`;
  const min = Math.floor(deltaSec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const days = Math.floor(hr / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  const years = Math.floor(days / 365);
  return `${years}y ago`;
}
