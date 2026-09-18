import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";

import { notifyAuthRequired } from "@/auth/authEvents";
import { shouldForceLogin } from "@/auth/shouldForceLogin";

/**
 * On ANY query/mutation error, if it is a 401 ({@link shouldForceLogin}) signal
 * the AuthProvider to clear the token and drop to the login screen. Wired here
 * (the single QueryCache/MutationCache) so the gating reacts to a 401 from
 * anywhere in the app — not just one component's fetch. Non-401 errors are
 * ignored here and handled by their callers.
 */
function forceLoginOn401(error: unknown): void {
  if (shouldForceLogin(error)) {
    notifyAuthRequired();
  }
}

/** App-wide TanStack Query client. Defaults are conservative; tune per-issue. */
export const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: forceLoginOn401 }),
  mutationCache: new MutationCache({ onError: forceLoginOn401 }),
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
