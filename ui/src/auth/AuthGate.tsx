import type { ReactNode } from "react";

import { LoginScreen } from "@/auth/LoginScreen";
import { useAuth } from "@/auth/AuthProvider";

export interface AuthGateProps {
  children: ReactNode;
}

/**
 * Renders the login screen when re-authentication is required, otherwise the
 * app. AUTH_ENABLED-off behaviour: with no token and no prior 401,
 * `isAuthRequired` is false → the app renders normally. Only a real 401 (which
 * sets `isAuthRequired`) — or an explicit sign-out — forces the login screen.
 */
export function AuthGate({ children }: AuthGateProps) {
  const { isAuthRequired } = useAuth();

  if (isAuthRequired) {
    return <LoginScreen expired />;
  }

  return <>{children}</>;
}
