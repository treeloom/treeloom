import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";

import { api } from "@/api/client";
import { onAuthRequired } from "@/auth/authEvents";

/** Value exposed by {@link useAuth}. */
export interface AuthContextValue {
  /**
   * True once a real 401 has forced re-authentication (or after an explicit
   * sign-out). When true the app renders the login screen. "Authenticated" is
   * NOT tracked here — it derives from `GET /auth/me` (the `["auth","me"]`
   * query). On AUTH_ENABLED-off installs this stays false and the app renders
   * until/unless the server actually 401s.
   */
  isAuthRequired: boolean;
  /**
   * Username/password sign-in (cookie/session). Posts `POST /auth/login`; on
   * success invalidates `["auth","me"]` and clears the re-auth flag. Throws on
   * failure (401/429/…) so the form can surface the error.
   */
  signIn: (username: string, password: string) => Promise<void>;
  /**
   * Sign out: `POST /auth/logout`, clear the React Query cache, and force the
   * login screen. Logout never throws — even if the server call fails the local
   * state is cleared.
   */
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** Minimal API surface the provider needs — injectable for tests. */
export interface AuthApi {
  post: <T>(path: string, body?: unknown) => Promise<T>;
}

export interface AuthProviderProps {
  children: ReactNode;
  /** Injectable API client seam for tests; defaults to the app client. */
  client?: AuthApi;
  /** Injectable 401-subscription seam for tests; defaults to the real bus. */
  subscribe?: typeof onAuthRequired;
  /**
   * Injectable React Query client. Defaults to the one from context
   * (`useQueryClient`). Tests may override to assert cache effects.
   */
  queryClientOverride?: QueryClient;
}

const AUTH_ME_KEY = ["auth", "me"] as const;

/**
 * Tracks an `isAuthRequired` flag and exposes cookie/session sign-in & sign-out.
 *
 * Authentication state is NOT stored here: it derives from `GET /auth/me`
 * (rendered via the `["auth","me"]` query). The provider only reacts to a 401
 * from anywhere by flipping into "needs re-authentication" (the login screen),
 * and owns the login/logout side effects.
 *
 * Initial state is permissive: the app renders even with no session (see
 * {@link AuthGate}); the login screen is forced *reactively* on a 401 — never
 * pre-emptively — so an open / AUTH_ENABLED-off indexer is usable as-is.
 */
export function AuthProvider({
  children,
  client = api,
  subscribe = onAuthRequired,
  queryClientOverride,
}: AuthProviderProps) {
  const ctxQueryClient = useQueryClient();
  const queryClient = queryClientOverride ?? ctxQueryClient;
  const [isAuthRequired, setAuthRequired] = useState(false);

  const signIn = useCallback(
    async (username: string, password: string) => {
      // Sets the session cookie on success; throws ApiError on 401/429/etc.
      await client.post("/auth/login", { username, password });
      // Re-resolve "who am I" against the fresh session and drop the gate.
      await queryClient.invalidateQueries({ queryKey: AUTH_ME_KEY });
      setAuthRequired(false);
    },
    [client, queryClient],
  );

  const signOut = useCallback(async () => {
    try {
      await client.post("/auth/logout");
    } catch {
      // Best-effort: a failed logout still clears local state below.
    }
    queryClient.clear();
    setAuthRequired(true);
  }, [client, queryClient]);

  // React to a 401 surfaced from any query/mutation: drop cached auth state and
  // force the login screen with the "session expired" message.
  useEffect(() => {
    return subscribe(() => {
      queryClient.removeQueries({ queryKey: AUTH_ME_KEY });
      setAuthRequired(true);
    });
  }, [queryClient, subscribe]);

  const value = useMemo<AuthContextValue>(
    () => ({ isAuthRequired, signIn, signOut }),
    [isAuthRequired, signIn, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Access the auth context. Throws if used outside an {@link AuthProvider}. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an <AuthProvider>");
  }
  return ctx;
}
