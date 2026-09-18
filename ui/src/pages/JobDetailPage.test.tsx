import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { ApiError } from "@/api/client";
import type { JobGroupDetail, JobTask } from "@/api/jobGroups";

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

async function renderDetail(id: string) {
  const { JobDetailPage } = await import("@/pages/JobDetailPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <MemoryRouter initialEntries={[`/jobs/${id}`]}>
        <Routes>
          <Route path="/jobs/:id" element={<JobDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
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
    group_id: "grp_1",
    ...over,
  };
}

function detail(tasks: JobTask[]): JobGroupDetail {
  return {
    id: "grp_1",
    label: "repos.yaml",
    kind: "fleet",
    created_at: Date.now() / 1000 - 60,
    created_by: "user_1",
    task_count: tasks.length,
    status_counts: { queued: 0, running: 0, done: 0, failed: 0, dead_letter: 0 },
    progress: { processed_files: 6, total_files: 12 },
    status: "running",
    tasks,
  };
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
});

describe("JobDetailPage", () => {
  it("renders the in-progress, completed, and failed buckets", async () => {
    getMock.mockResolvedValue(
      detail([
        task({ id: "a", source: "/repo/active", status: "running", processed_files: 3, total_files: 6 }),
        task({ id: "b", source: "/repo/queued", status: "queued" }),
        task({ id: "c", source: "/repo/finished", status: "done" }),
        task({
          id: "d",
          source: "/repo/broken",
          status: "failed",
          error: "clone failed: transport 'file' not allowed\nat line 2",
        }),
      ]),
    );
    await renderDetail("grp_1");

    // In-progress grid
    const inprogress = await screen.findByLabelText("In-progress tasks");
    expect(within(inprogress).getByText("/repo/active")).toBeTruthy();
    expect(within(inprogress).getByText("/repo/queued")).toBeTruthy();

    // Completed list
    const completed = screen.getByLabelText("Completed tasks");
    expect(within(completed).getByText("/repo/finished")).toBeTruthy();

    // Failed list shows first line of the error only
    const failed = screen.getByLabelText("Failed tasks");
    expect(within(failed).getByText("/repo/broken")).toBeTruthy();
    expect(
      within(failed).getByText("clone failed: transport 'file' not allowed"),
    ).toBeTruthy();
    expect(within(failed).queryByText(/at line 2/)).toBeNull();
  });

  it("renders a friendly not-found on 404 instead of crashing", async () => {
    getMock.mockRejectedValue(new ApiError(404, "not found"));
    await renderDetail("nope");
    expect(await screen.findByText("Job not found")).toBeTruthy();
    expect(screen.getByText(/No job with id "nope"/)).toBeTruthy();
  });
});
