import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";

import { ApiError } from "@/api/client";

const postMock = vi.fn();
vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>(
    "@/api/client",
  );
  return { ...actual, api: { post: (p: string, b?: unknown) => postMock(p, b) } };
});

function testClient() {
  return new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
}

async function renderSubmit() {
  const { SubmitPage } = await import("@/pages/SubmitPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <MemoryRouter>
        <SubmitPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  postMock.mockReset();
});

describe("SubmitPage — repos mode", () => {
  it("creates one job group then posts each repo with the group_id", async () => {
    // 1st call: /job-groups -> {id}. Then 3x /index-repo, one of them fails.
    postMock.mockImplementation((path: string, body?: { url?: string }) => {
      if (path === "/job-groups") return Promise.resolve({ id: "grp_42" });
      if (body?.url === "https://x/bad")
        return Promise.reject(new ApiError(400, "unsafe url"));
      return Promise.resolve({
        job_id: "task",
        status: "queued",
        source_id: "src",
      });
    });

    await renderSubmit();

    fireEvent.change(screen.getByLabelText(/Repositories/i), {
      target: {
        value: ["/home/u/repo", "https://x/good main", "https://x/bad"].join(
          "\n",
        ),
      },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit job/i }));

    // 1 group + 3 repos
    await waitFor(() => expect(postMock).toHaveBeenCalledTimes(4));

    expect(postMock).toHaveBeenCalledWith("/job-groups", {
      label: "3 repositories",
      kind: "manual",
    });
    expect(postMock).toHaveBeenCalledWith("/index-repo", {
      path: "/home/u/repo",
      force: false,
      skip_graph: false,
      group_id: "grp_42",
    });
    expect(postMock).toHaveBeenCalledWith("/index-repo", {
      url: "https://x/good",
      branch: "main",
      force: false,
      skip_graph: false,
      group_id: "grp_42",
    });
    expect(postMock).toHaveBeenCalledWith("/index-repo", {
      url: "https://x/bad",
      force: false,
      skip_graph: false,
      group_id: "grp_42",
    });

    // Results panel: per-entry outcomes incl. the failure.
    expect(await screen.findByText("Job submitted")).toBeTruthy();
    expect(screen.getByText("/home/u/repo")).toBeTruthy();
    expect(screen.getByText(/failed — unsafe url/i)).toBeTruthy();

    // "View Job" deep-links to /jobs/{group_id}.
    const view = screen.getByRole("link", { name: /view job/i });
    expect(view.getAttribute("href")).toBe("/jobs/grp_42");
  });

  it("aborts (no /index-repo calls) when /job-groups fails", async () => {
    postMock.mockImplementation((path: string) => {
      if (path === "/job-groups")
        return Promise.reject(new ApiError(500, "group create failed"));
      return Promise.resolve({});
    });
    await renderSubmit();

    fireEvent.change(screen.getByLabelText(/Repositories/i), {
      target: { value: "/home/u/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit job/i }));

    expect(await screen.findByText(/group create failed/i)).toBeTruthy();
    // Only the job-groups call happened — nothing was submitted.
    expect(postMock).toHaveBeenCalledTimes(1);
    expect(postMock).toHaveBeenCalledWith("/job-groups", expect.anything());
  });

  it("disables submit when there are no parseable entries", async () => {
    await renderSubmit();
    const button = screen.getByRole("button", { name: /submit job/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    // comments/blanks only → still disabled
    fireEvent.change(screen.getByLabelText(/Repositories/i), {
      target: { value: "# just a comment\n\n" },
    });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    // a real line enables it
    fireEvent.change(screen.getByLabelText(/Repositories/i), {
      target: { value: "/r" },
    });
    expect((button as HTMLButtonElement).disabled).toBe(false);
  });
});

describe("SubmitPage — file mode", () => {
  it("posts a single-file body to /index-file", async () => {
    postMock.mockResolvedValue({
      job_id: "task_1",
      status: "queued",
      source_id: "src_file",
    });
    await renderSubmit();

    fireEvent.change(screen.getByLabelText("Submit"), {
      target: { value: "file" },
    });
    fireEvent.change(screen.getByLabelText("File path"), {
      target: { value: "/home/u/repo/main.py" },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit file/i }));

    await waitFor(() => expect(postMock).toHaveBeenCalledTimes(1));
    expect(postMock).toHaveBeenCalledWith("/index-file", {
      file_path: "/home/u/repo/main.py",
      force: false,
      skip_graph: false,
    });
    expect(await screen.findByText("File accepted")).toBeTruthy();
    expect(screen.getByText("src_file")).toBeTruthy();
  });
});
