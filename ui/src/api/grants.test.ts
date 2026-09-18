import { describe, expect, it } from "vitest";

import {
  hasGrant,
  principalLabel,
  validateGrant,
  type GroupSummary,
  type SourceGrant,
  type UserSummary,
} from "@/api/grants";

const users: UserSummary[] = [
  { id: "u1", username: "alice", role: "admin" },
  { id: "u2", username: "bob", role: "user" },
];
const groups: GroupSummary[] = [{ id: "g1", name: "engineering" }];

describe("principalLabel", () => {
  it("resolves a user id to its username", () => {
    expect(principalLabel("user", "u2", users, groups)).toBe("bob");
  });
  it("resolves a group id to its name", () => {
    expect(principalLabel("group", "g1", users, groups)).toBe("engineering");
  });
  it("falls back to the raw id for an unknown principal", () => {
    expect(principalLabel("user", "ghost", users, groups)).toBe("ghost");
    expect(principalLabel("group", "ghost", users, groups)).toBe("ghost");
  });
});

describe("validateGrant", () => {
  it("rejects a bad principal type", () => {
    expect(validateGrant("team", "u1", "allow")).toMatch(/principal type/i);
  });
  it("requires a chosen user", () => {
    expect(validateGrant("user", "", "allow")).toMatch(/user/i);
  });
  it("requires a chosen group", () => {
    expect(validateGrant("group", "  ", "deny")).toMatch(/group/i);
  });
  it("rejects a bad effect", () => {
    expect(validateGrant("user", "u1", "maybe")).toMatch(/allow or deny/i);
  });
  it("accepts a well-formed allow/deny request", () => {
    expect(validateGrant("user", "u1", "allow")).toBeNull();
    expect(validateGrant("group", "g1", "deny")).toBeNull();
  });
});

describe("hasGrant", () => {
  const grants: SourceGrant[] = [
    { principal_type: "user", principal_id: "u1", effect: "allow" },
  ];
  it("detects an existing principal regardless of effect", () => {
    expect(hasGrant(grants, "user", "u1")).toBe(true);
  });
  it("is false for an ungranted principal", () => {
    expect(hasGrant(grants, "user", "u2")).toBe(false);
    expect(hasGrant(grants, "group", "u1")).toBe(false);
  });
});
