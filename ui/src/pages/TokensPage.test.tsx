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

async function renderTokens() {
  const { TokensPage } = await import("@/pages/TokensPage");
  return render(
    <QueryClientProvider client={testClient()}>
      <TokensPage />
    </QueryClientProvider>,
  );
}

function meBody(role: string): AuthMe {
  return {
    user: {
      id: "u1",
      username: "alice",
      email: "",
      role,
      active: true,
      created_at: "2026-01-01T00:00:00Z",
    },
    scopes: role === "admin" ? ["search", "index", "admin"] : ["search"],
    auth_method: "bearer",
  };
}

/** Route GET by path: /auth/me, /users/u1/tokens, /api-keys. */
function routeGet(role: string, tokens: unknown[], keys: unknown[] = []) {
  getMock.mockImplementation((p: string) => {
    if (p === "/auth/me") return Promise.resolve(meBody(role));
    if (p === "/users/u1/tokens") return Promise.resolve(tokens);
    if (p === "/api-keys") return Promise.resolve(keys);
    return Promise.reject(new Error(`unexpected GET ${p}`));
  });
}

afterEach(() => {
  cleanup();
  getMock.mockReset();
  postMock.mockReset();
  delMock.mockReset();
});

describe("TokensPage — PATs", () => {
  it("reveals a created PAT secret once", async () => {
    routeGet("user", []);
    postMock.mockResolvedValue({
      id: "t1",
      user_id: "u1",
      name: "laptop",
      scopes: ["search"],
      created_at: "2026-06-01T00:00:00Z",
      expires_at: null,
      last_used_at: null,
      revoked: false,
      token: "super-secret-raw-token",
    });
    await renderTokens();
    await screen.findByRole("heading", { name: "My API keys" });

    fireEvent.change(screen.getByLabelText("Token name"), {
      target: { value: "laptop" },
    });
    fireEvent.click(screen.getByRole("button", { name: /create api key/i }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/users/u1/tokens", {
        name: "laptop",
        scopes: ["search"],
      }),
    );
    expect(await screen.findByTestId("secret-value")).toHaveProperty(
      "textContent",
      "super-secret-raw-token",
    );
    expect(screen.getByText(/won't be shown again/i)).toBeTruthy();
  });

  it("revokes a PAT via DELETE", async () => {
    routeGet("user", [
      {
        id: "t9",
        user_id: "u1",
        name: "old",
        scopes: ["search"],
        created_at: "2026-06-01T00:00:00Z",
        expires_at: null,
        last_used_at: null,
        revoked: false,
      },
    ]);
    delMock.mockResolvedValue(undefined);
    await renderTokens();
    await screen.findByText("old");

    fireEvent.click(screen.getByRole("button", { name: /^revoke$/i }));
    await waitFor(() =>
      expect(delMock).toHaveBeenCalledWith("/users/u1/tokens/t9"),
    );
  });
});

describe("TokensPage — System API keys (admin gating)", () => {
  it("renders 'My API keys' but hides 'System API keys' for non-admins", async () => {
    routeGet("user", []);
    await renderTokens();
    // Every logged-in user sees their own API keys.
    await screen.findByRole("heading", { name: "My API keys" });
    // The admin-only system keys section is hidden.
    expect(
      screen.queryByRole("heading", { name: "System API keys" }),
    ).toBeNull();
  });

  it("shows 'System API keys' for admins and creates one", async () => {
    routeGet("admin", [], []);
    postMock.mockResolvedValue({
      id: "k1",
      name: "ci",
      scopes: ["search"],
      all_access: false,
      created_by: "u1",
      created_at: "2026-06-01T00:00:00Z",
      expires_at: null,
      last_used_at: null,
      revoked: false,
      key: "raw-api-key-value",
    });
    await renderTokens();
    await screen.findByRole("heading", { name: "System API keys" });

    fireEvent.change(screen.getByLabelText("Key name"), {
      target: { value: "ci" },
    });
    // Two "Create API key" buttons exist (My + System); the System form has the
    // "Key name" field — click its sibling submit button.
    const keyName = screen.getByLabelText("Key name");
    const systemForm = keyName.closest("form")!;
    const createBtn = systemForm.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement;
    fireEvent.click(createBtn);

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/api-keys", {
        name: "ci",
        scopes: ["search"],
        all_access: false,
      }),
    );
    expect(await screen.findByText("raw-api-key-value")).toBeTruthy();
  });
});
