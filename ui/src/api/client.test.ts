import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  DEFAULT_API_BASE,
  createClient,
  resolveApiBase,
  type ClientDeps,
} from "@/api/client";

/** Build a fetch stub returning the given status/body once, capturing the call. */
function stubFetch(opts: {
  status: number;
  body?: unknown;
  statusText?: string;
}): { fetchImpl: typeof fetch; calls: { url: string; init?: RequestInit }[] } {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetchImpl = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), init });
      const text =
        opts.body === undefined ? "" : JSON.stringify(opts.body);
      return new Response(text, {
        status: opts.status,
        statusText: opts.statusText ?? "",
        headers: { "Content-Type": "application/json" },
      });
    },
  ) as unknown as typeof fetch;
  return { fetchImpl, calls };
}

function deps(overrides: Partial<ClientDeps>): ClientDeps {
  return {
    getBase: () => "http://api.test",
    fetchImpl: stubFetch({ status: 200, body: {} }).fetchImpl,
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("resolveApiBase", () => {
  it("reads window.__TREELOOM_API_BASE__ when set", () => {
    const win = { __TREELOOM_API_BASE__: "http://runtime:9999" } as Window;
    expect(resolveApiBase(win)).toBe("http://runtime:9999");
  });

  it("falls back to the default when undefined", () => {
    expect(resolveApiBase({} as Window)).toBe(DEFAULT_API_BASE);
    expect(resolveApiBase(undefined)).toBe(DEFAULT_API_BASE);
  });

  it("falls back to the default when the value is empty", () => {
    const win = { __TREELOOM_API_BASE__: "" } as Window;
    expect(resolveApiBase(win)).toBe(DEFAULT_API_BASE);
  });
});

describe("createClient request behavior", () => {
  it("prefixes the base and parses a 2xx JSON body", async () => {
    const { fetchImpl, calls } = stubFetch({
      status: 200,
      body: { status: "ok" },
    });
    const client = createClient(deps({ fetchImpl }));

    const result = await client.get<{ status: string }>("/health");

    expect(result).toEqual({ status: "ok" });
    expect(calls[0].url).toBe("http://api.test/health");
    expect(calls[0].init?.method).toBe("GET");
  });

  it("sends credentials: 'include' for cookie/session auth", async () => {
    const { fetchImpl, calls } = stubFetch({ status: 200, body: {} });
    const client = createClient(deps({ fetchImpl }));

    await client.get("/sources");

    expect(calls[0].init?.credentials).toBe("include");
  });

  it("never attaches an Authorization header (cookie auth, no bearer)", async () => {
    const { fetchImpl, calls } = stubFetch({ status: 200, body: {} });
    const client = createClient(deps({ fetchImpl }));

    await client.get("/sources");
    await client.post("/auth/login", { username: "a", password: "b" });

    for (const call of calls) {
      const headers = (call.init?.headers ?? {}) as Record<string, string>;
      expect(headers["Authorization"]).toBeUndefined();
    }
  });

  it("sets a JSON content-type and serializes the body on POST", async () => {
    const { fetchImpl, calls } = stubFetch({ status: 200, body: { id: 1 } });
    const client = createClient(deps({ fetchImpl }));

    await client.post("/index-repo", { path: "/x" });

    const headers = calls[0].init?.headers as Record<string, string>;
    expect(calls[0].init?.method).toBe("POST");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(calls[0].init?.body).toBe(JSON.stringify({ path: "/x" }));
  });

  it("throws ApiError with the right status on a non-2xx response", async () => {
    const { fetchImpl } = stubFetch({
      status: 500,
      body: { detail: "boom" },
    });
    const client = createClient(deps({ fetchImpl }));

    await expect(client.get("/health")).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "boom",
    });
    await expect(client.get("/health")).rejects.toBeInstanceOf(ApiError);
  });

  it("distinguishes a 401 via status and isUnauthorized", async () => {
    const { fetchImpl } = stubFetch({
      status: 401,
      body: { detail: "unauthorized" },
    });
    const client = createClient(deps({ fetchImpl }));

    let caught: unknown;
    try {
      await client.get("/sources");
    } catch (err) {
      caught = err;
    }

    expect(caught).toBeInstanceOf(ApiError);
    const apiErr = caught as ApiError;
    expect(apiErr.status).toBe(401);
    expect(apiErr.isUnauthorized).toBe(true);
  });

  it("uses the resolved base via getBase()", async () => {
    const { fetchImpl, calls } = stubFetch({ status: 200, body: {} });
    const client = createClient(
      deps({ fetchImpl, getBase: () => "http://other.test/" }),
    );

    await client.del("/sources/abc");

    // trailing slash on base + leading slash on path collapse to one.
    expect(calls[0].url).toBe("http://other.test/sources/abc");
    expect(calls[0].init?.method).toBe("DELETE");
  });
});
