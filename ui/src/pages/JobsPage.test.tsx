import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";

import { ApiError } from "@/api/client";
import type { JobGroupSummary } from "@/api/jobGroups";

// Mock the typed client; the real fetch functions + pure helpers still run.
const getMock = vi.fn();
vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>(
    "@/api/client",
  );
  return { ...actual, api: { get: (p: string) => getMock(p) } };
});

function testClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
}

async function renderJobsPage() {
  const { JobsPage } = await import("@/pages/JobsPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <MemoryRouter>
        <JobsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function group(over: Partial<JobGroupSummary>): JobGroupSummary {
  return {
    id: "grp_1",
    label: "repos.yaml",
    kind: "fleet",
    created_at: Date.now() / 1000 - 120,
    created_by: "user_1",
    task_count: 4,
    status_counts: { queued: 1, running: 1, done: 2, failed: 0, dead_letter: 0 },
    progress: { processed_files: 50, total_files: 100 },
    status: "running",
    ...over,
  };
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
});

describe("JobsPage feed", () => {
  it("renders empty state when there are no groups", async () => {
    getMock.mockResolvedValue([]);
    await renderJobsPage();
    expect(await screen.findByText("No indexing jobs yet.")).toBeTruthy();
  });

  it("renders a row per group with status badge and N/M tasks", async () => {
    getMock.mockResolvedValue([
      group({ id: "grp_1", label: "alpha", status: "running" }),
      group({
        id: "grp_2",
        label: "beta",
        status: "done",
        task_count: 3,
        status_counts: {
          queued: 0,
          running: 0,
          done: 3,
          failed: 0,
          dead_letter: 0,
        },
        progress: { processed_files: 30, total_files: 30 },
      }),
    ]);
    await renderJobsPage();

    // Scope to the feed list to avoid the top-summary count pills.
    const list = await screen.findByRole("list", { name: "Jobs" });
    expect(within(list).getByText("alpha")).toBeTruthy();
    expect(within(list).getByText("beta")).toBeTruthy();
    // status badges (one per row)
    expect(within(list).getByText("running")).toBeTruthy();
    expect(within(list).getByText("done")).toBeTruthy();
    // N/M tasks (done of task_count) for each row
    expect(within(list).getByText("2/4 tasks")).toBeTruthy();
    expect(within(list).getByText("3/3 tasks")).toBeTruthy();
  });

  it("renders an error message from ApiError", async () => {
    getMock.mockRejectedValue(new ApiError(500, "kaboom"));
    await renderJobsPage();
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "kaboom",
    );
  });
});
