import { describe, expect, it } from "vitest";

import { ApiError } from "@/api/client";
import { shouldForceLogin } from "@/auth/shouldForceLogin";

describe("shouldForceLogin", () => {
  it("returns true for a 401 ApiError", () => {
    expect(shouldForceLogin(new ApiError(401, "unauthorized"))).toBe(true);
  });

  it("returns false for a non-401 ApiError (403/404/500)", () => {
    expect(shouldForceLogin(new ApiError(403, "forbidden"))).toBe(false);
    expect(shouldForceLogin(new ApiError(404, "not found"))).toBe(false);
    expect(shouldForceLogin(new ApiError(500, "boom"))).toBe(false);
  });

  it("returns false for a plain Error / network failure", () => {
    expect(shouldForceLogin(new Error("network down"))).toBe(false);
    expect(shouldForceLogin(new TypeError("Failed to fetch"))).toBe(false);
  });

  it("returns false for non-Error throwables", () => {
    expect(shouldForceLogin(null)).toBe(false);
    expect(shouldForceLogin(undefined)).toBe(false);
    expect(shouldForceLogin("401")).toBe(false);
    expect(shouldForceLogin({ status: 401 })).toBe(false);
  });
});
