import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import {
  AuthProvider,
  useAuth,
  type AuthApi,
  type AuthContextValue,
} from "@/auth/AuthProvider";

type PostMock = ReturnType<typeof vi.fn>;

function clientStub(post: PostMock = vi.fn(async () => ({}))): AuthApi {
  // The generic `post<T>` doesn't unify with a concrete mock return type; the
  // cast is test-only and the mock satisfies the runtime contract.
  return { post: post as unknown as AuthApi["post"] };
}

/** Render the provider and expose its context value to the test. */
function renderAuth(props: {
  client?: AuthApi;
  subscribe?: (listener: () => void) => () => void;
  queryClient?: QueryClient;
}) {
  const qc =
    props.queryClient ??
    new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const captured: { value: AuthContextValue | null } = { value: null };
  function Probe() {
    captured.value = useAuth();
    return null;
  }
  render(
    <QueryClientProvider client={qc}>
      <AuthProvider
        client={props.client}
        subscribe={props.subscribe}
        queryClientOverride={qc}
      >
        <Probe />
      </AuthProvider>
    </QueryClientProvider>,
  );
  return { captured, queryClient: qc };
}

afterEach(cleanup);

describe("AuthProvider", () => {
  it("starts NOT auth-required (AUTH off / no session works)", () => {
    const { captured } = renderAuth({ client: clientStub() });
    expect(captured.value?.isAuthRequired).toBe(false);
  });

  it("signIn posts /auth/login with username + password and clears the flag", async () => {
    const post = vi.fn(async () => ({}));
    const { captured } = renderAuth({ client: clientStub(post) });

    await act(async () => {
      await captured.value?.signIn("alice", "hunter2");
    });

    expect(post).toHaveBeenCalledWith("/auth/login", {
      username: "alice",
      password: "hunter2",
    });
    expect(captured.value?.isAuthRequired).toBe(false);
  });

  it("signIn propagates the error (e.g. 401) so the form can show it", async () => {
    const post = vi.fn(async () => {
      throw new Error("bad creds");
    });
    const { captured } = renderAuth({ client: clientStub(post) });

    await expect(
      act(async () => {
        await captured.value?.signIn("alice", "wrong");
      }),
    ).rejects.toThrow("bad creds");
  });

  it("signOut posts /auth/logout, clears the cache, and forces login", async () => {
    const post = vi.fn(async () => ({}));
    const qc = new QueryClient();
    const clearSpy = vi.spyOn(qc, "clear");
    const { captured } = renderAuth({ client: clientStub(post), queryClient: qc });

    await act(async () => {
      await captured.value?.signOut();
    });

    expect(post).toHaveBeenCalledWith("/auth/logout");
    expect(clearSpy).toHaveBeenCalledTimes(1);
    expect(captured.value?.isAuthRequired).toBe(true);
  });

  it("signOut still forces login even if /auth/logout fails", async () => {
    const post = vi.fn(async () => {
      throw new Error("network");
    });
    const { captured } = renderAuth({ client: clientStub(post) });

    await act(async () => {
      await captured.value?.signOut();
    });

    expect(captured.value?.isAuthRequired).toBe(true);
  });

  it("reacts to a 401 from the bus by forcing login (gating unchanged)", () => {
    let fire: (() => void) | null = null;
    const subscribe = (listener: () => void) => {
      fire = listener;
      return () => {
        fire = null;
      };
    };

    const { captured } = renderAuth({ client: clientStub(), subscribe });
    expect(captured.value?.isAuthRequired).toBe(false);

    act(() => fire?.());

    expect(captured.value?.isAuthRequired).toBe(true);
  });
});
