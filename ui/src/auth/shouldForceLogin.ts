import { ApiError } from "@/api/client";

/**
 * Pure predicate: does this error indicate the app should drop to the login
 * screen? True only for an {@link ApiError} that is a 401 (`isUnauthorized`).
 *
 * Everything else — non-401 ApiErrors (403/404/500…), network/`TypeError`s, and
 * non-Error throwables — returns false, so a transient server/network failure
 * never forces a re-login. Kept pure (no React, no storage) so it can be unit
 * tested directly and reused by the QueryCache/MutationCache `onError` hooks.
 */
export function shouldForceLogin(error: unknown): boolean {
  return error instanceof ApiError && error.isUnauthorized;
}
