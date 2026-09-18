import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { ApiError } from "@/api/client";
import {
  fetchJobGroups,
  jobStatusCounts,
  overallActiveProgress,
  percent,
  relativeTime,
  type JobGroupSummary,
} from "@/api/jobGroups";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/jobs/StatusBadge";

/** Poll fast while anything is active, slow to a stop when idle. */
function refetchInterval(groups: JobGroupSummary[] | undefined): number | false {
  if (!groups) return 3000;
  return groups.some((g) => g.status === "running") ? 3000 : false;
}

export function JobsPage() {
  const query = useQuery({
    queryKey: ["job-groups"],
    queryFn: fetchJobGroups,
    refetchInterval: (q) => refetchInterval(q.state.data),
  });

  if (query.isLoading) {
    return (
      <section>
        <Header />
        <p className="mt-6 text-sm text-muted-foreground">Loading jobs…</p>
      </section>
    );
  }

  if (query.isError) {
    const message =
      query.error instanceof ApiError
        ? query.error.message
        : "Failed to load jobs.";
    return (
      <section>
        <Header />
        <p
          role="alert"
          className="mt-6 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {message}
        </p>
      </section>
    );
  }

  const groups = query.data ?? [];

  if (groups.length === 0) {
    return (
      <section>
        <Header />
        <p className="mt-8 text-center text-sm text-muted-foreground">
          No indexing jobs yet.
        </p>
      </section>
    );
  }

  const overall = overallActiveProgress(groups);
  const counts = jobStatusCounts(groups);

  return (
    <section>
      <Header />

      {/* Top summary: live progress across active jobs + headline counts. */}
      <div className="mt-6 rounded-lg border border-border bg-card p-5">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Active progress</span>
          <span className="text-sm tabular-nums text-muted-foreground">
            {overall.processed.toLocaleString()} /{" "}
            {overall.total.toLocaleString()} files ({overall.percent}%)
          </span>
        </div>
        <Progress value={overall.percent} className="mt-3" />

        <div className="mt-4 flex flex-wrap gap-2">
          <CountPill label="running" value={counts.running} variant="default" />
          <CountPill label="queued" value={counts.queued} variant="muted" />
          <CountPill label="done" value={counts.done} variant="success" />
          <CountPill label="failed" value={counts.failed} variant="danger" />
        </div>
      </div>

      {/* Feed rows (newest-first as returned by the API). */}
      <ul className="mt-6 space-y-3" aria-label="Jobs">
        {groups.map((g) => (
          <JobRow key={g.id} group={g} />
        ))}
      </ul>
    </section>
  );
}

function Header() {
  return (
    <div>
      <h1 className="text-xl font-semibold tracking-tight">Jobs</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Indexing job batches, newest first.
      </p>
    </div>
  );
}

function CountPill({
  label,
  value,
  variant,
}: {
  label: string;
  value: number;
  variant: "default" | "muted" | "success" | "danger";
}) {
  return (
    <Badge variant={variant} className="gap-1.5">
      <span className="tabular-nums font-semibold">{value}</span>
      <span className="font-normal opacity-80">{label}</span>
    </Badge>
  );
}

function JobRow({ group }: { group: JobGroupSummary }) {
  const pct = percent(
    group.progress?.processed_files ?? 0,
    group.progress?.total_files ?? 0,
  );
  const doneCount = group.status_counts?.done ?? 0;

  return (
    <li>
      <Link
        to={`/jobs/${group.id}`}
        className="block rounded-lg border border-border bg-card px-5 py-4 transition-colors hover:border-primary/50 hover:bg-accent/30"
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate font-medium">{group.label}</span>
            <Badge variant="outline">{group.kind}</Badge>
            <StatusBadge status={group.status} />
          </div>
          <span className="shrink-0 text-xs text-muted-foreground">
            {relativeTime(group.created_at)}
          </span>
        </div>

        <div className="mt-3 flex items-center gap-3">
          <Progress value={pct} className="h-1.5" />
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {doneCount}/{group.task_count} tasks
          </span>
        </div>
      </Link>
    </li>
  );
}
