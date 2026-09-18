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
import type { BackendListResponse } from "@/api/embeddingBackends";

const getMock = vi.fn();
const postMock = vi.fn();
const delMock = vi.fn();
vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>(
    "@/api/client",
  );
  return {
    ...actual,
    api: {
      get: (p: string) => getMock(p),
      post: (p: string, b?: unknown) => postMock(p, b),
      del: (p: string) => delMock(p),
    },
  };
});

function testClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

async function renderBackends() {
  const { BackendsPage } = await import("@/pages/BackendsPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <BackendsPage />
    </QueryClientProvider>,
  );
}

function listResponse(): BackendListResponse {
  return {
    registry: [
      {
        id: "b1",
        url: "http://localhost:8082",
        klass: "gpu",
        enabled: true,
        created_at: "2026-06-01T00:00:00Z",
        created_by: null,
      },
    ],
    local: { "http://localhost:8082": { in_flight_tokens: 3, failures: 0 } },
    process_id: "host:1",
  };
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
  postMock.mockReset();
  delMock.mockReset();
});

describe("BackendsPage", () => {
  it("lists backends from GET /embedding-backends", async () => {
    getMock.mockResolvedValue(listResponse());
    await renderBackends();
    expect(await screen.findByText("http://localhost:8082")).toBeTruthy();
    expect(getMock).toHaveBeenCalledWith("/embedding-backends");
  });

  it("creates a backend via POST then refetches", async () => {
    getMock.mockResolvedValue(listResponse());
    postMock.mockResolvedValue({
      id: "b2",
      url: "http://localhost:9000",
      klass: "cpu",
      enabled: true,
      created_at: "2026-06-02T00:00:00Z",
      created_by: null,
    });
    await renderBackends();
    await screen.findByText("http://localhost:8082");

    fireEvent.change(screen.getByLabelText("Backend URL"), {
      target: { value: "http://localhost:9000" },
    });
    fireEvent.click(screen.getByRole("button", { name: /add backend/i }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/embedding-backends", {
        url: "http://localhost:9000",
        klass: "gpu",
      }),
    );
  });

  it("deletes a backend via DELETE /embedding-backends/{id}", async () => {
    getMock.mockResolvedValue(listResponse());
    delMock.mockResolvedValue(undefined);
    await renderBackends();
    await screen.findByText("http://localhost:8082");

    fireEvent.click(screen.getByRole("button", { name: /^delete$/i }));
    await waitFor(() =>
      expect(delMock).toHaveBeenCalledWith("/embedding-backends/b1"),
    );
  });

  it("shows an admin-required message on 403", async () => {
    getMock.mockRejectedValue(new ApiError(403, "forbidden"));
    await renderBackends();
    expect(await screen.findByText(/admin access required/i)).toBeTruthy();
  });
});
