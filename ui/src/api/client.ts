/** Local-dev fallback when `window.__TREELOOM_API_BASE__` is unset. */
export const DEFAULT_API_BASE = "http://localhost:8001";

/**
 * Typed error thrown on any non-2xx response. Carries the HTTP `status` so
 * callers can branch — notably `status === 401` for re-auth.
 */
export class ApiError extends Error {
  readonly status: number;
  /** Parsed/raw response body, when available — handy for surfacing detail. */
  readonly body: unknown;

  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }

  /** Convenience flag for the re-auth hook point. */
  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

/**
 * Resolve the API base. Reads `window.__TREELOOM_API_BASE__` (set by
 * `public/config.js`) and falls back to {@link DEFAULT_API_BASE}. The `win`
 * param is injectable so it can be stubbed in tests; defaults to the real
 * `window` (or `undefined` in non-browser envs).
 */
export function resolveApiBase(
  win: Window | undefined = typeof window !== "undefined" ? window : undefined,
): string {
  const base = win?.__TREELOOM_API_BASE__;
  return base && base.length > 0 ? base : DEFAULT_API_BASE;
}

/** Dependencies the request core reads — all injectable for tests. */
export interface ClientDeps {
  /** Resolves the API base URL. */
  getBase: () => string;
  /** The fetch implementation to use. */
  fetchImpl: typeof fetch;
}

function defaultDeps(): ClientDeps {
  return {
    getBase: () => resolveApiBase(),
    fetchImpl: (...args: Parameters<typeof fetch>) => fetch(...args),
  };
}

function joinUrl(base: string, path: string): string {
  const b = base.replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

async function request<T>(
  method: string,
  path: string,
  body: unknown | undefined,
  deps: ClientDeps,
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  // Cookie/session auth: send the `treeloom_session` cookie cross-origin.
  // No `Authorization` header — the SPA no longer stores a bearer token.
  const res = await deps.fetchImpl(joinUrl(deps.getBase(), path), {
    method,
    headers,
    credentials: "include",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  const payload = await parseBody(res);

  if (!res.ok) {
    throw new ApiError(res.status, errorMessage(res, payload), payload);
  }

  return payload as T;
}

/** Parse the response body as JSON, falling back to text / null. */
async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text().catch(() => "");
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorMessage(res: Response, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown; message?: unknown }).detail;
    const message = (body as { message?: unknown }).message;
    if (typeof detail === "string") return detail;
    if (typeof message === "string") return message;
  }
  if (typeof body === "string" && body.length > 0) return body;
  return `HTTP ${res.status} ${res.statusText}`.trim();
}

/**
 * Build an API client bound to the given dependencies. Production code uses the
 * default-export {@link api}; tests construct one with stubbed deps.
 */
export function createClient(deps: ClientDeps = defaultDeps()) {
  return {
    get: <T>(path: string): Promise<T> => request<T>("GET", path, undefined, deps),
    post: <T>(path: string, body?: unknown): Promise<T> =>
      request<T>("POST", path, body, deps),
    put: <T>(path: string, body?: unknown): Promise<T> =>
      request<T>("PUT", path, body, deps),
    patch: <T>(path: string, body?: unknown): Promise<T> =>
      request<T>("PATCH", path, body, deps),
    del: <T>(path: string): Promise<T> => request<T>("DELETE", path, undefined, deps),
  };
}

/** Default app-wide API client (reads window + global fetch; cookie auth). */
export const api = createClient();
