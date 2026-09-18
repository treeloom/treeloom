import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const getMock = vi.fn();
const postMock = vi.fn();
vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>(
    "@/api/client",
  );
  return {
    ...actual,
    api: {
      get: (p: string) => getMock(p),
      post: (p: string, b?: unknown) => postMock(p, b),
    },
  };
});

afterEach(() => {
  cleanup();
  getMock.mockReset();
  postMock.mockReset();
});

async function renderControls() {
  const { AuthControls } = await import("@/auth/AuthControls");
  const { AuthProvider } = await import("@/auth/AuthProvider");
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <AuthProvider queryClientOverride={qc}>
        <AuthControls />
      </AuthProvider>
    </QueryClientProvider>,
  );
}

describe("displayNameFromMe", () => {
  it("prefers username, then name, then email", async () => {
    const { displayNameFromMe } = await import("@/auth/AuthControls");
    expect(displayNameFromMe({ username: "u", name: "n", email: "e" })).toBe("u");
    expect(displayNameFromMe({ name: "n", email: "e" })).toBe("n");
    expect(displayNameFromMe({ email: "e" })).toBe("e");
  });

  it("returns null for empty / missing data", async () => {
    const { displayNameFromMe } = await import("@/auth/AuthControls");
    expect(displayNameFromMe(null)).toBeNull();
    expect(displayNameFromMe(undefined)).toBeNull();
    expect(displayNameFromMe({})).toBeNull();
  });
});

describe("AuthControls", () => {
  it("the Logout button calls signOut (POST /auth/logout)", async () => {
    getMock.mockResolvedValue({ username: "admin" });
    postMock.mockResolvedValue(undefined);
    await renderControls();

    fireEvent.click(screen.getByRole("button", { name: /logout/i }));

    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/auth/logout", undefined),
    );
  });

  it("renders 'logged in as <username>' from /auth/me", async () => {
    getMock.mockResolvedValue({ username: "admin" });
    await renderControls();
    expect(await screen.findByText("admin")).toBeTruthy();
  });
});
