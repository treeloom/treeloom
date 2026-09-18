import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { ApiError } from "@/api/client";
import {
  defaultJobLabel,
  entryLabel,
  parseRepoLines,
  submitJob,
  submitMultiRepo,
  type JobAck,
  type MultiRepoResult,
  type RepoEntry,
  type SubmitFormState,
} from "@/api/submit";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";

type Mode = "repos" | "file";

const REPOS_PLACEHOLDER = [
  "~/source/myrepo",
  "https://github.com/org/repo",
  "https://github.com/org/repo main",
  "# comment lines and blanks are ignored",
].join("\n");

export function SubmitPage() {
  const [mode, setMode] = useState<Mode>("repos");

  return (
    <section>
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Submit</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Index one or many repositories as a single Job, or a single file.
        </p>
      </div>

      <Field label="Submit" htmlFor="submit-mode" className="mt-6 max-w-2xl">
        <Select
          id="submit-mode"
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
        >
          <option value="repos">Repositories</option>
          <option value="file">File</option>
        </Select>
      </Field>

      {mode === "repos" ? <ReposForm /> : <FileForm />}
    </section>
  );
}

// --------------------------------------------------------------------------
// Repos form (multi-repo Job)
// --------------------------------------------------------------------------

function ReposForm() {
  const [label, setLabel] = useState("");
  const [reposText, setReposText] = useState("");
  const [force, setForce] = useState(false);
  const [skipGraph, setSkipGraph] = useState(false);

  const entries = useMemo<RepoEntry[]>(
    () => parseRepoLines(reposText),
    [reposText],
  );

  const mutation = useMutation<MultiRepoResult, unknown, void>({
    mutationFn: () =>
      submitMultiRepo(entries, {
        label: label.trim() || defaultJobLabel(entries),
        force,
        skipGraph,
      }),
  });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (entries.length === 0) return;
    mutation.reset();
    mutation.mutate();
  }

  // A failed `POST /job-groups` aborts the whole submission (no group created).
  const groupError =
    mutation.error instanceof ApiError
      ? mutation.error.message
      : mutation.isError
        ? "Could not create the Job."
        : null;

  return (
    <>
      <form onSubmit={onSubmit} className="mt-6 max-w-2xl space-y-5">
        <Field label="Job label (optional)" htmlFor="submit-label">
          <Input
            id="submit-label"
            placeholder={
              entries.length > 0 ? defaultJobLabel(entries) : "My import"
            }
            value={label}
            onChange={(e) => setLabel(e.target.value)}
          />
        </Field>

        <Field
          label="Repositories (one per line)"
          htmlFor="submit-repos"
          hint="A local path, or a git URL with an optional branch: <url> [branch]. Blank lines and # comments are ignored."
        >
          <textarea
            id="submit-repos"
            rows={6}
            spellCheck={false}
            placeholder={REPOS_PLACEHOLDER}
            value={reposText}
            onChange={(e) => setReposText(e.target.value)}
            className="flex w-full rounded-md border border-border bg-transparent px-3 py-2 font-mono text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          />
        </Field>

        <div className="flex flex-wrap gap-6">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
            />
            <span>Force re-index</span>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={skipGraph}
              onChange={(e) => setSkipGraph(e.target.checked)}
            />
            <span>Skip graph (two-pass)</span>
          </label>
        </div>

        {groupError && (
          <p
            role="alert"
            className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
          >
            {groupError}
          </p>
        )}

        <div className="flex items-center gap-3">
          <Button
            type="submit"
            disabled={entries.length === 0 || mutation.isPending}
          >
            {mutation.isPending
              ? "Submitting…"
              : entries.length > 1
                ? `Submit Job (${entries.length} repos)`
                : "Submit Job"}
          </Button>
          {entries.length === 0 && reposText.trim() === "" && (
            <span className="text-sm text-muted-foreground">
              Add at least one repository.
            </span>
          )}
        </div>
      </form>

      {mutation.isSuccess && mutation.data && (
        <ReposResults result={mutation.data} />
      )}
    </>
  );
}

function ReposResults({ result }: { result: MultiRepoResult }) {
  const { groupId, results } = result;
  const failed = results.filter((r) => r.outcome === "failed").length;
  const queued = results.filter((r) => r.outcome === "queued").length;
  const done = results.filter((r) => r.outcome === "done").length;

  return (
    <div
      role="status"
      className="mt-6 max-w-2xl rounded-lg border border-success/40 bg-success/10 p-5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-success">Job submitted</span>
        {queued > 0 && <Badge variant="outline">{queued} queued</Badge>}
        {done > 0 && <Badge variant="outline">{done} already indexed</Badge>}
        {failed > 0 && <Badge variant="outline">{failed} failed</Badge>}
      </div>

      <ul className="mt-3 space-y-1.5 text-sm">
        {results.map((r, i) => (
          <li key={i} className="flex flex-wrap items-baseline gap-2">
            <OutcomeDot outcome={r.outcome} />
            <span className="font-mono text-xs">{entryLabel(r.entry)}</span>
            <span className="text-muted-foreground">
              {r.outcome === "queued"
                ? "queued"
                : r.outcome === "done"
                  ? "already indexed"
                  : `failed — ${r.error ?? "unknown error"}`}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-4 flex flex-wrap gap-4 text-sm">
        <Link
          to={`/jobs/${groupId}`}
          className="text-primary underline-offset-4 hover:underline"
        >
          View Job →
        </Link>
        <Link
          to="/jobs"
          className="text-primary underline-offset-4 hover:underline"
        >
          Jobs feed →
        </Link>
      </p>
    </div>
  );
}

function OutcomeDot({ outcome }: { outcome: "queued" | "done" | "failed" }) {
  const color =
    outcome === "failed"
      ? "bg-danger"
      : outcome === "done"
        ? "bg-muted-foreground"
        : "bg-success";
  return (
    <span
      aria-hidden
      className={`mt-1.5 inline-block size-2 shrink-0 rounded-full ${color}`}
    />
  );
}

// --------------------------------------------------------------------------
// File sub-form (single-file submit, unchanged endpoint)
// --------------------------------------------------------------------------

function fileForm(path: string, force: boolean, skipGraph: boolean): SubmitFormState {
  return {
    kind: "file",
    path,
    url: "",
    branch: "",
    pattern: "",
    repoMode: "path",
    force,
    skipGraph,
  };
}

function FileForm() {
  const [path, setPath] = useState("");
  const [force, setForce] = useState(false);
  const [skipGraph, setSkipGraph] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);

  const mutation = useMutation<JobAck, unknown, SubmitFormState>({
    mutationFn: (f) => submitJob(f),
  });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!path.trim()) {
      setClientError("Enter a file path.");
      return;
    }
    setClientError(null);
    mutation.reset();
    mutation.mutate(fileForm(path, force, skipGraph));
  }

  const apiError =
    mutation.error instanceof ApiError
      ? mutation.error.message
      : mutation.isError
        ? "Submission failed."
        : null;

  return (
    <>
      <form onSubmit={onSubmit} className="mt-6 max-w-2xl space-y-5">
        <Field label="File path" htmlFor="submit-file-path">
          <Input
            id="submit-file-path"
            placeholder="~/source/myrepo/src/main.py"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
        </Field>

        <div className="flex flex-wrap gap-6">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
            />
            <span>Force re-index</span>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={skipGraph}
              onChange={(e) => setSkipGraph(e.target.checked)}
            />
            <span>Skip graph (two-pass)</span>
          </label>
        </div>

        {clientError && (
          <p
            role="alert"
            className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
          >
            {clientError}
          </p>
        )}

        {apiError && (
          <p
            role="alert"
            className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
          >
            {apiError}
          </p>
        )}

        <div className="flex items-center gap-3">
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "Submitting…" : "Submit file"}
          </Button>
        </div>
      </form>

      {mutation.isSuccess && mutation.data && <FileAck ack={mutation.data} />}
    </>
  );
}

function FileAck({ ack }: { ack: JobAck }) {
  const warnings = ack.warnings ?? [];
  return (
    <div
      role="status"
      className="mt-6 max-w-2xl rounded-lg border border-success/40 bg-success/10 p-5"
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-success">File accepted</span>
        <Badge variant="outline">{ack.status}</Badge>
      </div>
      <dl className="mt-3 space-y-1 text-sm">
        <div className="flex gap-2">
          <dt className="text-muted-foreground">source_id</dt>
          <dd className="font-mono">{ack.source_id}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="text-muted-foreground">job_id</dt>
          <dd className="font-mono">{ack.job_id}</dd>
        </div>
      </dl>

      {warnings.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-medium text-warning">Preflight warnings</p>
          <ul className="mt-1 list-inside list-disc text-xs text-muted-foreground">
            {warnings.map((w, i) => (
              <li key={i}>{warningText(w)}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-4 text-sm">
        {/* A single-file JobAck carries a per-Task job_id, not a group id. */}
        <Link
          to="/jobs"
          className="text-primary underline-offset-4 hover:underline"
        >
          Watch it on the Jobs feed →
        </Link>
      </p>
    </div>
  );
}

// --------------------------------------------------------------------------
// Shared bits
// --------------------------------------------------------------------------

function Field({
  label,
  htmlFor,
  hint,
  className,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`space-y-1.5 ${className ?? ""}`}>
      <label htmlFor={htmlFor} className="text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/** Render a preflight warning object as a single readable line. */
function warningText(w: Record<string, unknown>): string {
  const msg = w.message ?? w.detail ?? w.warning;
  if (typeof msg === "string") return msg;
  return JSON.stringify(w);
}
