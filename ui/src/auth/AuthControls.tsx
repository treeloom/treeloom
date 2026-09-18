import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/auth/AuthProvider";

/** Shape of `GET /auth/me` (only a display name is consumed). */
interface AuthMe {
  username?: string;
  name?: string;
  email?: string;
}

/** Best-effort display name from the `/auth/me` body, or null. */
export function displayNameFromMe(me: AuthMe | undefined | null): string | null {
  if (!me) return null;
  return me.username ?? me.name ?? me.email ?? null;
}

/**
 * Header auth controls: an optional "logged in as <user>" label (from
 * `GET /auth/me`) and a Logout button.
 *
 * `/auth/me` is best-effort — on AUTH_ENABLED-off installs or for an anonymous
 * caller it may 404/401; we treat any failure as "anonymous" and render no
 * label (the query's own 401, like any other, still routes through the
 * QueryCache → login gate, which is the desired behaviour for a real 401).
 */
export function AuthControls() {
  const { signOut } = useAuth();

  const me = useQuery({
    // Shared app-wide ["auth","me"] cache: the header is mounted on every page,
    // so this warms the cache that pages (Users/Tokens/Sources admin gating)
    // read. staleTime 0 so a re-login (which invalidates this key) refetches the
    // new user rather than serving the previous one.
    queryKey: ["auth", "me"],
    queryFn: () => api.get<AuthMe>("/auth/me"),
    retry: false,
    staleTime: 0,
  });

  const name = me.isSuccess ? displayNameFromMe(me.data) : null;

  return (
    <div className="flex items-center gap-3">
      {name && (
        <span className="text-[13px] text-muted-foreground">
          {name}
        </span>
      )}
      <Button variant="outline" size="sm" onClick={() => void signOut()}>
        Logout
      </Button>
    </div>
  );
}
