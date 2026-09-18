import { describe, expect, it } from "vitest";

import {
  firstLine,
  jobStatusCounts,
  overallActiveProgress,
  percent,
  relativeTime,
  taskBuckets,
  taskLabel,
  taskStatusCounts,
  type JobGroupSummary,
  type JobTask,
} from "@/api/jobGroups";

function group(over: Partial<JobGroupSummary>): JobGroupSummary {
  return {
    id: "grp_x",
    label: "repos.yaml",
    kind: "fleet",
    created_at: 1_750_000_000,
    created_by: "user_1",
    task_count: 0,
    status_counts: { queued: 0, running: 0, done: 0, failed: 0, dead_letter: 0 },
    progress: { processed_files: 0, total_files: 0 },
    status: "queued",
    ...over,
  };
}

function task(over: Partial<JobTask>): JobTask {
  return {
    id: "t1",
    job_id: "t1",
    source: "",
    source_id: "s1",
    status: "queued",
    kind: "repo",
    total_files: 0,
    processed_files: 0,
    committed_files: 0,
    total_chunks: 0,
    start_time: null,
    finished_at: null,
    error: "",
    message: "",
    current_file: "",
    source_path: "",
    source_url: "",
    source_branch: "",
    commit_sha: "",
    errors: 0,
    payload: null,
    attempts: 0,
    group_id: "grp_x",
    ...over,
  };
}

describe("percent", () => {
  it("returns 0 when total is 0 (no NaN)", () => {
    expect(percent(0, 0)).toBe(0);
    expect(percent(5, 0)).toBe(0);
  });

  it("returns 0 for negative / non-finite inputs", () => {
    expect(percent(-1, 10)).toBe(0);
    expect(percent(NaN, 10)).toBe(0);
    expect(percent(5, NaN)).toBe(0);
  });

  it("rounds to an integer percentage", () => {
    expect(percent(1, 3)).toBe(33);
    expect(percent(2, 3)).toBe(67);
  });

  it("clamps to 100 when processed >= total", () => {
    expect(percent(10, 10)).toBe(100);
    expect(percent(20, 10)).toBe(100);
  });
});

describe("overallActiveProgress", () => {
  it("returns all zeros when no groups are active", () => {
    const groups = [
      group({ status: "done", progress: { processed_files: 5, total_files: 5 } }),
      group({ status: "queued", progress: { processed_files: 0, total_files: 9 } }),
    ];
    expect(overallActiveProgress(groups)).toEqual({
      processed: 0,
      total: 0,
      percent: 0,
    });
  });

  it("sums only running groups and computes percent", () => {
    const groups = [
      group({ status: "running", progress: { processed_files: 30, total_files: 100 } }),
      group({ status: "running", progress: { processed_files: 20, total_files: 100 } }),
      group({ status: "done", progress: { processed_files: 999, total_files: 999 } }),
    ];
    expect(overallActiveProgress(groups)).toEqual({
      processed: 50,
      total: 200,
      percent: 25,
    });
  });

  it("handles an empty list", () => {
    expect(overallActiveProgress([])).toEqual({
      processed: 0,
      total: 0,
      percent: 0,
    });
  });
});

describe("jobStatusCounts", () => {
  it("tallies group statuses across mixed input", () => {
    const groups = [
      group({ status: "running" }),
      group({ status: "running" }),
      group({ status: "queued" }),
      group({ status: "done" }),
      group({ status: "failed" }),
    ];
    expect(jobStatusCounts(groups)).toEqual({
      running: 2,
      queued: 1,
      done: 1,
      failed: 1,
    });
  });

  it("is all zeros for an empty list", () => {
    expect(jobStatusCounts([])).toEqual({
      running: 0,
      queued: 0,
      done: 0,
      failed: 0,
    });
  });
});

describe("taskBuckets", () => {
  it("buckets running/queued into inprogress, done, and failed/dead_letter", () => {
    const tasks = [
      task({ id: "a", status: "running" }),
      task({ id: "b", status: "queued" }),
      task({ id: "c", status: "done" }),
      task({ id: "d", status: "failed" }),
      task({ id: "e", status: "dead_letter" }),
    ];
    const out = taskBuckets(tasks);
    expect(out.inprogress.map((t) => t.id)).toEqual(["a", "b"]);
    expect(out.done.map((t) => t.id)).toEqual(["c"]);
    expect(out.failed.map((t) => t.id)).toEqual(["d", "e"]);
  });

  it("handles an empty list", () => {
    expect(taskBuckets([])).toEqual({ inprogress: [], done: [], failed: [] });
  });
});

describe("taskStatusCounts", () => {
  it("counts dead_letter under failed", () => {
    const tasks = [
      task({ status: "running" }),
      task({ status: "done" }),
      task({ status: "dead_letter" }),
      task({ status: "failed" }),
    ];
    expect(taskStatusCounts(tasks)).toEqual({
      running: 1,
      queued: 0,
      done: 1,
      failed: 2,
    });
  });
});

describe("taskLabel", () => {
  it("prefers source, then source_path, then source_url, then id", () => {
    expect(taskLabel(task({ source: "/a", source_path: "/b" }))).toBe("/a");
    expect(taskLabel(task({ source: "", source_path: "/b" }))).toBe("/b");
    expect(
      taskLabel(task({ source: "", source_path: "", source_url: "http://x" })),
    ).toBe("http://x");
    expect(
      taskLabel(task({ id: "id7", source: "", source_path: "", source_url: "" })),
    ).toBe("id7");
  });
});

describe("firstLine", () => {
  it("returns the first line, trimmed", () => {
    expect(firstLine("boom\nstack\nframe")).toBe("boom");
    expect(firstLine("  single  ")).toBe("single");
    expect(firstLine("")).toBe("");
  });
});

describe("relativeTime", () => {
  const now = 1_750_000_000_000; // ms

  it("says 'just now' for very recent and future timestamps", () => {
    expect(relativeTime(now / 1000, now)).toBe("just now");
    expect(relativeTime(now / 1000 + 60, now)).toBe("just now"); // future
  });

  it("formats seconds / minutes / hours / days", () => {
    expect(relativeTime(now / 1000 - 30, now)).toBe("30s ago");
    expect(relativeTime(now / 1000 - 120, now)).toBe("2m ago");
    expect(relativeTime(now / 1000 - 3 * 3600, now)).toBe("3h ago");
    expect(relativeTime(now / 1000 - 5 * 86400, now)).toBe("5d ago");
  });

  it("formats months and years", () => {
    expect(relativeTime(now / 1000 - 60 * 86400, now)).toBe("2mo ago");
    expect(relativeTime(now / 1000 - 400 * 86400, now)).toBe("1y ago");
  });

  it("returns empty string for non-finite input", () => {
    expect(relativeTime(NaN, now)).toBe("");
  });
});
