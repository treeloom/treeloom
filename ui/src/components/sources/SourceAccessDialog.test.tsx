import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import type { SourceRecord } from "@/api/sources";

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

function source(): SourceRecord {
  return {
    id: "src-1",
    path: "/home/u/repo",
    url: "",
    branch: "main",
    indexed_at: 1_700_000_000,
    file_count: 1,
    chunk_count: 1,
    commit_sha: "abc",
    graph_indexed: true,
  };
}

async function renderDialog(onClose = vi.fn()) {
  const { SourceAccessDialog } = await import(
    "@/components/sources/SourceAccessDialog"
  );
  const result = render(
    <QueryClientProvider client={testClient()}>
      <SourceAccessDialog source={source()} onClose={onClose} />
    </QueryClientProvider>,
  );
  return { ...result, onClose };
}

/** Default GET router: grants + users + groups. */
function routeGet(grants: unknown[]) {
  getMock.mockImplementation((path: string) => {
    if (path === "/sources/src-1/grants") return Promise.resolve(grants);
    if (path === "/users")
      return Promise.resolve([
        { id: "u1", username: "alice", role: "admin" },
        { id: "u2", username: "bob", role: "user" },
      ]);
    if (path === "/groups")
      return Promise.resolve([{ id: "g1", name: "engineering" }]);
    return Promise.reject(new Error(`unexpected GET ${path}`));
  });
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
  postMock.mockReset();
  delMock.mockReset();
});

describe("SourceAccessDialog", () => {
  it("lists existing grants with resolved principal labels", async () => {
    routeGet([
      { principal_type: "user", principal_id: "u2", effect: "allow" },
      { principal_type: "group", principal_id: "g1", effect: "deny" },
    ]);
    await renderDialog();

    // Scope to the grants list (<span>) — "bob"/"engineering" also appear as
    // <option> labels in the principal picker.
    const list = await screen.findByRole("list");
    expect(within(list).getByText("bob")).toBeTruthy();
    expect(within(list).getByText("engineering")).toBeTruthy();
    expect(within(list).getByText("Allow")).toBeTruthy();
    expect(within(list).getByText("Deny")).toBeTruthy();
    expect(getMock).toHaveBeenCalledWith("/sources/src-1/grants");
  });

  it("posts the right body when adding a grant", async () => {
    routeGet([]);
    postMock.mockResolvedValue({
      principal_type: "user",
      principal_id: "u1",
      effect: "deny",
    });
    await renderDialog();

    // Wait for directories to load (the picker becomes selectable).
    await screen.findByText("No grants yet.");

    fireEvent.change(screen.getByLabelText("Principal"), {
      target: { value: "u1" },
    });
    fireEvent.change(screen.getByLabelText("Effect"), {
      target: { value: "deny" },
    });
    fireEvent.click(screen.getByRole("button", { name: /grant access/i }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/sources/src-1/grants", {
        principal_type: "user",
        principal_id: "u1",
        effect: "deny",
      }),
    );
  });

  it("revokes a grant via DELETE on the principal path", async () => {
    routeGet([{ principal_type: "user", principal_id: "u2", effect: "allow" }]);
    delMock.mockResolvedValue(undefined);
    await renderDialog();

    await screen.findByRole("list");
    fireEvent.click(screen.getByRole("button", { name: /^revoke$/i }));

    await waitFor(() =>
      expect(delMock).toHaveBeenCalledWith("/sources/src-1/grants/user/u2"),
    );
  });

  it("shows the empty-directory message when there are no users", async () => {
    getMock.mockImplementation((path: string) => {
      if (path === "/sources/src-1/grants") return Promise.resolve([]);
      if (path === "/users") return Promise.resolve([]);
      if (path === "/groups") return Promise.resolve([]);
      return Promise.reject(new Error(`unexpected GET ${path}`));
    });
    await renderDialog();

    expect(await screen.findByText(/no users to grant/i)).toBeTruthy();
  });

  it("closes on Escape and on backdrop click", async () => {
    routeGet([]);
    const { onClose } = await renderDialog();
    await screen.findByText("No grants yet.");

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByLabelText("Close dialog"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
