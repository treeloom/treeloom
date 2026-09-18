import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import type { SourceRecord } from "@/api/sources";

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

async function renderSourcesPage() {
  const { SourcesPage } = await import("@/pages/SourcesPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <SourcesPage />
    </QueryClientProvider>,
  );
}

function src(over: Partial<SourceRecord>): SourceRecord {
  return {
    id: "sha_1",
    path: "/home/u/repo",
    url: "",
    branch: "main",
    indexed_at: Math.floor(Date.now() / 1000) - 120,
    file_count: 12,
    chunk_count: 34,
    commit_sha: "abc123",
    graph_indexed: true,
    ...over,
  };
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
});

describe("SourcesPage", () => {
  it("renders the empty state when there are no sources", async () => {
    getMock.mockResolvedValue([]);
    await renderSourcesPage();
    expect(await screen.findByText("No indexed sources.")).toBeTruthy();
  });

  it("renders an error message from ApiError", async () => {
    getMock.mockRejectedValue(new ApiError(500, "kaboom"));
    await renderSourcesPage();
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "kaboom",
    );
  });

  it("renders a row per source with graph badge", async () => {
    getMock.mockResolvedValue([
      src({ id: "1", path: "/home/u/treeloom", graph_indexed: true }),
      src({ id: "2", path: "", url: "https://github.com/acme/widgets", graph_indexed: false }),
    ]);
    await renderSourcesPage();

    expect(await screen.findByText("/home/u/treeloom")).toBeTruthy();
    expect(screen.getByText("https://github.com/acme/widgets")).toBeTruthy();
    expect(screen.getByText("yes")).toBeTruthy();
    expect(screen.getByText("no")).toBeTruthy();
  });

  it("narrows visible rows as the filter input changes", async () => {
    getMock.mockResolvedValue([
      src({ id: "1", path: "/home/u/treeloom" }),
      src({ id: "2", path: "/home/u/featbit" }),
    ]);
    await renderSourcesPage();

    // Both visible initially.
    expect(await screen.findByText("/home/u/treeloom")).toBeTruthy();
    expect(screen.getByText("/home/u/featbit")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Filter sources"), {
      target: { value: "featbit" },
    });

    expect(screen.queryByText("/home/u/treeloom")).toBeNull();
    expect(screen.getByText("/home/u/featbit")).toBeTruthy();
  });

  it("does NOT fetch staleness on load; fetches lazily on Check click", async () => {
    getMock.mockImplementation((path: string) => {
      if (path === "/sources") return Promise.resolve([src({ id: "1" })]);
      return Promise.resolve({
        indexed_sha: "abc",
        current_sha: "def",
        is_stale: true,
      });
    });
    await renderSourcesPage();

    await screen.findByText("/home/u/repo");
    // Only the catalog was fetched — no staleness call on load.
    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith("/sources");

    fireEvent.click(screen.getByRole("button", { name: "Check" }));

    await waitFor(() => expect(screen.getByText("stale")).toBeTruthy());
    expect(getMock).toHaveBeenCalledWith("/sources/1/staleness");
  });
});
