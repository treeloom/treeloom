import { describe, expect, it } from "vitest";

import { toggleScope, validateTokenName } from "@/api/tokens";

describe("validateTokenName", () => {
  it("rejects blank", () => {
    expect(validateTokenName("   ")).toMatch(/name/i);
  });
  it("accepts a name", () => {
    expect(validateTokenName("ci")).toBeNull();
  });
});

describe("toggleScope", () => {
  it("adds a scope, preserving canonical order", () => {
    expect(toggleScope(["index"], "search")).toEqual(["search", "index"]);
  });
  it("removes a present scope", () => {
    expect(toggleScope(["search", "index"], "search")).toEqual(["index"]);
  });
  it("keeps canonical order even when added last", () => {
    expect(toggleScope(["admin"], "search")).toEqual(["search", "admin"]);
  });
});
