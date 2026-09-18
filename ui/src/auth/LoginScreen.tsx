import { useState, type FormEvent } from "react";

import { ApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/auth/AuthProvider";

export interface LoginScreenProps {
  /**
   * When true, show the "session expired / re-authenticate" banner. Set by the
   * gate when a 401 forced the user back here (vs. a first-time sign-in).
   */
  expired?: boolean;
}

/** Map a sign-in failure to a user-facing message. */
export function loginErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Invalid username or password";
    if (error.status === 429) {
      return "Too many attempts — try again in a moment";
    }
    return error.message;
  }
  return "Sign in failed. Please try again.";
}

/**
 * Username/password login screen (cookie/session auth).
 *
 * On submit it calls {@link useAuth}'s `signIn(username, password)`, which posts
 * `POST /auth/login` and (on success) sets the `treeloom_session` cookie. Errors
 * are surfaced inline (401 → bad credentials; 429 → rate-limited; else the
 * `ApiError` message). The form is disabled while the request is in flight.
 */
export function LoginScreen({ expired = false }: LoginScreenProps) {
  const { signIn } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = username.trim().length > 0 && password.length > 0;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await signIn(username.trim(), password);
    } catch (err) {
      setError(loginErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 text-foreground">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center">
          <img
            src="/brand/treeloom-logo-stacked.svg"
            alt="Treeloom — Parse Trees · Weave Graphs"
            className="h-24 w-auto select-none"
          />
          <p className="mt-4 text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
            Operator Console
          </p>
        </div>

        {expired && (
          <div
            role="alert"
            className="mb-4 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning"
          >
            Your session expired. Please re-authenticate.
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="space-y-4 rounded-lg border border-border bg-card p-6"
          aria-label="Sign in"
        >
          <div className="space-y-1.5">
            <label
              htmlFor="login-username"
              className="block text-sm font-medium text-foreground"
            >
              Username
            </label>
            <Input
              id="login-username"
              type="text"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={submitting}
              aria-label="Username"
            />
          </div>

          <div className="space-y-1.5">
            <label
              htmlFor="login-password"
              className="block text-sm font-medium text-foreground"
            >
              Password
            </label>
            <Input
              id="login-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
              aria-label="Password"
            />
          </div>

          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={!canSubmit || submitting}
          >
            {submitting ? "Signing in…" : "Sign in"}
          </Button>
        </form>
      </div>
    </div>
  );
}
