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
import { AuthProvider, type AuthApi } from "@/auth/AuthProvider";
import { LoginScreen, loginErrorMessage } from "@/auth/LoginScreen";

type PostMock = ReturnType<typeof vi.fn>;

function renderLogin(post: PostMock, props: { expired?: boolean } = {}) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const client: AuthApi = { post: post as unknown as AuthApi["post"] };
  return render(
    <QueryClientProvider client={qc}>
      <AuthProvider client={client} queryClientOverride={qc}>
        <LoginScreen {...props} />
      </AuthProvider>
    </QueryClientProvider>,
  );
}

function enter(username: string, password: string) {
  fireEvent.change(screen.getByLabelText("Username"), {
    target: { value: username },
  });
  fireEvent.change(screen.getByLabelText("Password"), {
    target: { value: password },
  });
}

afterEach(cleanup);

describe("LoginScreen", () => {
  it("renders username and password fields", () => {
    renderLogin(vi.fn(async () => ({})));
    expect(screen.getByLabelText("Username")).toBeTruthy();
    expect(screen.getByLabelText("Password")).toBeTruthy();
  });

  it("calls signIn (POST /auth/login) with the entered credentials on submit", async () => {
    const post = vi.fn(async () => ({}));
    renderLogin(post);

    enter("alice", "hunter2");
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith("/auth/login", {
        username: "alice",
        password: "hunter2",
      }),
    );
  });

  it("does not submit when username or password is empty", () => {
    const post = vi.fn(async () => ({}));
    renderLogin(post);

    const button = screen.getByRole("button", { name: /sign in/i });
    // No input -> disabled, click is a no-op.
    fireEvent.click(button);
    expect(post).not.toHaveBeenCalled();

    // Username only, blank password -> still disabled.
    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "alice" },
    });
    fireEvent.click(button);
    expect(post).not.toHaveBeenCalled();
  });

  it("maps a 401 to 'Invalid username or password'", async () => {
    const post = vi.fn(async () => {
      throw new ApiError(401, "Unauthorized");
    });
    renderLogin(post);

    enter("alice", "wrong");
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    expect(
      await screen.findByText(/invalid username or password/i),
    ).toBeTruthy();
  });

  it("maps a 429 to a rate-limit message", async () => {
    const post = vi.fn(async () => {
      throw new ApiError(429, "Too Many Requests");
    });
    renderLogin(post);

    enter("alice", "hunter2");
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText(/too many attempts/i)).toBeTruthy();
  });

  it("falls back to the ApiError message for other errors", async () => {
    const post = vi.fn(async () => {
      throw new ApiError(500, "server exploded");
    });
    renderLogin(post);

    enter("alice", "hunter2");
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText(/server exploded/i)).toBeTruthy();
  });

  it("shows the session-expired banner only when expired", () => {
    renderLogin(vi.fn(async () => ({})));
    expect(screen.queryByRole("alert")).toBeNull();

    cleanup();
    renderLogin(vi.fn(async () => ({})), { expired: true });
    expect(screen.getByRole("alert").textContent).toMatch(/session expired/i);
  });
});

describe("loginErrorMessage", () => {
  it("maps statuses and falls back", () => {
    expect(loginErrorMessage(new ApiError(401, "x"))).toMatch(
      /invalid username or password/i,
    );
    expect(loginErrorMessage(new ApiError(429, "x"))).toMatch(
      /too many attempts/i,
    );
    expect(loginErrorMessage(new ApiError(500, "boom"))).toBe("boom");
    expect(loginErrorMessage(new Error("weird"))).toMatch(/sign in failed/i);
  });
});
