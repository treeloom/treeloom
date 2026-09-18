import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import type { AuthMe } from "@/api/auth";

const getMock = vi.fn();
const postMock = vi.fn();
const putMock = vi.fn();
const patchMock = vi.fn();
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
      put: (p: string, b?: unknown) => putMock(p, b),
      patch: (p: string, b?: unknown) => patchMock(p, b),
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

async function renderUsers() {
  const { UsersPage } = await import("@/pages/UsersPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <UsersPage />
    </QueryClientProvider>,
  );
}

function meBody(role: string): AuthMe {
  return {
    user: {
      id: "u1",
      username: "alice",
      email: "alice@example.com",
      role,
      active: true,
      created_at: "2026-01-01T00:00:00Z",
    },
    scopes: role === "admin" ? ["search", "index", "admin"] : ["search"],
    auth_method: "bearer",
  };
}

function userRow(over: Record<string, unknown> = {}) {
  return {
    id: "u1",
    username: "alice",
    email: "alice@example.com",
    role: "user",
    active: true,
    all_access: false,
    created_at: "2026-06-01T00:00:00Z",
    ...over,
  };
}

/** Route GET by path: /auth/me, /users, /groups. */
function routeGet(role: string, users: unknown[], groups: unknown[] = []) {
  getMock.mockImplementation((p: string) => {
    if (p === "/auth/me") return Promise.resolve(meBody(role));
    if (p === "/users") return Promise.resolve(users);
    if (p === "/groups") return Promise.resolve(groups);
    return Promise.reject(new Error(`unexpected GET ${p}`));
  });
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
  postMock.mockReset();
  putMock.mockReset();
  patchMock.mockReset();
  delMock.mockReset();
});

describe("UsersPage — admin gating", () => {
  it("shows 'Admin access required' for non-admins", async () => {
    routeGet("user", []);
    await renderUsers();
    expect(await screen.findByText(/Admin access required/i)).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Users$/ })).toBeNull();
  });
});

describe("UsersPage — users table", () => {
  it("renders the users table for admins", async () => {
    routeGet("admin", [userRow({ username: "bob", id: "u2" })]);
    await renderUsers();
    expect(await screen.findByText("bob")).toBeTruthy();
  });

  it("creates a user, posts the right body, and reveals the api key once", async () => {
    routeGet("admin", []);
    postMock.mockResolvedValue({
      ...userRow({ username: "carol", id: "u3" }),
      api_key: "raw-user-api-key",
    });
    await renderUsers();
    await screen.findByRole("heading", { name: /^Users$/ });

    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "carol" },
    });
    fireEvent.click(screen.getByRole("button", { name: /create user/i }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/users", {
        username: "carol",
        role: "user",
      }),
    );
    expect(await screen.findByTestId("secret-value")).toHaveProperty(
      "textContent",
      "raw-user-api-key",
    );
    expect(screen.getByText(/won't be shown again/i)).toBeTruthy();
  });

  it("sets a user's role via PUT /users/{id}/role", async () => {
    routeGet("admin", [userRow({ username: "dave", id: "u4", role: "user" })]);
    putMock.mockResolvedValue(undefined);
    await renderUsers();
    await screen.findByText("dave");

    fireEvent.change(screen.getByLabelText("Role for dave"), {
      target: { value: "admin" },
    });
    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/users/u4/role", {
        role: "admin",
      }),
    );
  });
});
