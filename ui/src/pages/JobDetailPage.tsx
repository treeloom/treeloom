import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { ApiError } from "@/api/client";
import {
  fetchJobGroup,
  firstLine,
  percent,
  relativeTime,
  taskBuckets,
  taskLabel,
  taskStatusCounts,
  type JobGroupDetail,
  type JobTask,
} from "@/api/jobGroups";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/jobs/StatusBadge";

function refetchInterval(
  detail: JobGroupDetail | undefined,
): number | false {
  if (!detail) return 3000;
  const active = detail.tasks.some(
    (t) => t.status === "running" || t.status === "queued",
  );
  return active ? 3000 : false;
}

export function JobDetailPage() {
  const { id = "" } = useParams<{ id: string }>();

  const query = useQuery({
    queryKey: ["job-group", id],
    queryFn: () => fetchJobGroup(id),
    enabled: id.length > 0,
    refetchInterval: (q) => refetchInterval(q.state.data),
    retry: (failureCount, error) => {
      // Don't retry a 404 (unknown id) — surface "not found" immediately.
      if (error instanceof ApiError && error.status === 404) return false;
      return failureCount < 1;
    },
  });

  const backLink = (
    <Link
      to="/jobs"
      className="text-sm text-muted-foreground hover:text-foreground"
    >
      ← Back to Jobs
    </Link>
  );

  if (query.isLoading) {
    return (
      <section>
        {backLink}
        <p className="mt-6 text-sm text-muted-foreground">Loading job…</p>
      </section>
    );
  }

  if (query.isError) {
    const notFound =
      query.error instanceof ApiError && query.error.status === 404;
    return (
      <section>
        {backLink}
        <div className="mt-8 text-center">
          <h1 className="text-lg font-semibold">
            {notFound ? "Job not found" : "Failed to load job"}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {notFound
              ? `No job with id "${id}".`
              : query.error instanceof ApiError
                ? query.error.message
                : "Something went wrong."}
          </p>
        </div>
      </section>
    );
  }

  const detail = query.data!;
  const counts = taskStatusCounts(detail.tasks);
  const buckets = taskBuckets(detail.tasks);
  const overallPct = percent(
    detail.progress?.processed_files ?? 0,
    detail.progress?.total_files ?? 0,
  );

  return (
    <section>
      {backLink}

      <div className="mt-4 flex items-center gap-3">
        <h1 className="text-xl font-semibold tracking-tight">
          {detail.label}
        </h1>
        <Badge variant="outline">{detail.kind}</Badge>
        <StatusBadge status={detail.status} />
        <span className="text-xs text-muted-foreground">
          {relativeTime(detail.created_at)}
        </span>
      </div>

      {/* Overall progress + status counts among this job's tasks. */}
      <div className="mt-5 rounded-lg border border-border bg-card p-5">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Overall progress</span>
          <span className="text-sm tabular-nums text-muted-foreground">
            {(detail.progress?.processed_files ?? 0).toLocaleString()} /{" "}
            {(detail.progress?.total_files ?? 0).toLocaleString()} files (
            {overallPct}%)
          </span>
        </div>
        <Progress value={overallPct} className="mt-3" />

        <div className="mt-4 flex flex-wrap gap-2">
          <Badge variant="default" className="gap-1.5">
            <span className="tabular-nums font-semibold">
              {counts.running}
            </span>
            <span className="font-normal opacity-80">running</span>
          </Badge>
          <Badge variant="muted" className="gap-1.5">
            <span className="tabular-nums font-semibold">{counts.queued}</span>
            <span className="font-normal opacity-80">queued</span>
          </Badge>
          <Badge variant="success" className="gap-1.5">
            <span className="tabular-nums font-semibold">{counts.done}</span>
            <span className="font-normal opacity-80">done</span>
          </Badge>
          <Badge variant="danger" className="gap-1.5">
            <span className="tabular-nums font-semibold">{counts.failed}</span>
            <span className="font-normal opacity-80">failed</span>
          </Badge>
        </div>
      </div>

      {/* In-progress tasks grid */}
      <TaskSection title="In progress" count={buckets.inprogress.length}>
        {buckets.inprogress.length === 0 ? (
          <EmptyNote>No tasks in progress.</EmptyNote>
        ) : (
          <div
            className="grid grid-cols-1 gap-3 sm:grid-cols-2"
            aria-label="In-progress tasks"
          >
            {buckets.inprogress.map((t) => (
              <InProgressCard key={t.id} task={t} />
            ))}
          </div>
        )}
      </TaskSection>

      {/* Completed tasks */}
      <TaskSection title="Completed" count={buckets.done.length}>
        {buckets.done.length === 0 ? (
          <EmptyNote>No completed tasks yet.</EmptyNote>
        ) : (
          <ul className="space-y-2" aria-label="Completed tasks">
            {buckets.done.map((t) => (
              <li
                key={t.id}
                className="flex items-center justify-between gap-3 rounded-md border border-border bg-card px-4 py-2.5"
              >
                <span className="truncate text-sm">{taskLabel(t)}</span>
                <StatusBadge status={t.status} />
              </li>
            ))}
          </ul>
        )}
      </TaskSection>

      {/* Failed tasks */}
      <TaskSection title="Failed" count={buckets.failed.length}>
        {buckets.failed.length === 0 ? (
          <EmptyNote>No failed tasks.</EmptyNote>
        ) : (
          <ul className="space-y-2" aria-label="Failed tasks">
            {buckets.failed.map((t) => (
              <li
                key={t.id}
                className="rounded-md border border-danger/40 bg-danger/5 px-4 py-2.5"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate text-sm">{taskLabel(t)}</span>
                  <StatusBadge status={t.status} />
                </div>
                {firstLine(t.error) && (
                  <p className="mt-1 truncate text-xs text-danger">
                    {firstLine(t.error)}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </TaskSection>
    </section>
  );
}

function TaskSection({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) {
  return (
    <div className="mt-6">
      <h2 className="mb-3 text-sm font-semibold">
        {title}{" "}
        <span className="text-muted-foreground tabular-nums">({count})</span>
      </h2>
      {children}
    </div>
  );
}

function EmptyNote({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted-foreground">{children}</p>;
}

function InProgressCard({ task }: { task: JobTask }) {
  const pct = percent(task.processed_files, task.total_files);
  return (
    <div className="rounded-md border border-border bg-card px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-medium">{taskLabel(task)}</span>
        <StatusBadge status={task.status} />
      </div>
      <div className="mt-2 flex items-center gap-3">
        <Progress value={pct} className="h-1.5" />
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {pct}%
        </span>
      </div>
    </div>
  );
}
