import { describe, expect, it } from "vitest";

import { isAdmin, userId, type AuthMe } from "@/api/auth";

function me(over: Partial<AuthMe> & { role?: string }): AuthMe {
  const { role, ...rest } = over;
  return {
    user: {
      id: "u1",
      username: "alice",
      email: "",
      role: role ?? "user",
      active: true,
      created_at: "2026-01-01T00:00:00Z",
    },
    scopes: [],
    auth_method: "bearer",
    ...rest,
  };
}

describe("isAdmin", () => {
  it("true for role admin", () => {
    expect(isAdmin(me({ role: "admin" }))).toBe(true);
  });
  it("true when admin scope present", () => {
    expect(isAdmin(me({ role: "user", scopes: ["admin"] }))).toBe(true);
  });
  it("false for a plain user", () => {
    expect(isAdmin(me({ role: "user", scopes: ["search"] }))).toBe(false);
  });
  it("false for null/undefined", () => {
    expect(isAdmin(null)).toBe(false);
    expect(isAdmin(undefined)).toBe(false);
  });
});

describe("userId", () => {
  it("returns the user id", () => {
    expect(userId(me({}))).toBe("u1");
  });
  it("null when missing", () => {
    expect(userId(null)).toBeNull();
  });
});
