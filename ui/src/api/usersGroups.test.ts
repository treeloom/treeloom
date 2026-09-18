import { describe, expect, it, vi } from "vitest";

vi.mock("@/api/client", () => ({ api: { get: vi.fn() } }));

import { api } from "@/api/client";
import {
  buildCreateUserBody,
  fetchGroupMembers,
  normalizeRole,
  validateGroupName,
  validatePassword,
  validateUsername,
} from "@/api/usersGroups";

describe("fetchGroupMembers", () => {
  it("unwraps {group_id, members} to a bare id array", async () => {
    vi.mocked(api.get).mockResolvedValue({ group_id: "g1", members: ["u1", "u2"] });
    expect(await fetchGroupMembers("g1")).toEqual(["u1", "u2"]);
  });

  it("returns [] when members is missing (regression: .includes on object)", async () => {
    vi.mocked(api.get).mockResolvedValue({ group_id: "g1" });
    expect(await fetchGroupMembers("g1")).toEqual([]);
  });
});

describe("validateUsername", () => {
  it("rejects blank", () => {
    expect(validateUsername("   ")).toMatch(/username/i);
  });
  it("accepts a name", () => {
    expect(validateUsername("jdoe")).toBeNull();
  });
});

describe("validatePassword", () => {
  it("rejects short passwords", () => {
    expect(validatePassword("short")).toMatch(/8/);
  });
  it("accepts an 8+ char password", () => {
    expect(validatePassword("longenough")).toBeNull();
  });
});

describe("validateGroupName", () => {
  it("rejects blank", () => {
    expect(validateGroupName("")).toMatch(/group/i);
  });
  it("accepts a name", () => {
    expect(validateGroupName("team")).toBeNull();
  });
});

describe("normalizeRole", () => {
  it("keeps admin", () => {
    expect(normalizeRole("admin")).toBe("admin");
  });
  it("defaults unknown to user", () => {
    expect(normalizeRole("superuser")).toBe("user");
  });
});

describe("buildCreateUserBody", () => {
  it("trims username and includes role", () => {
    expect(buildCreateUserBody("  jdoe  ", "", "user")).toEqual({
      username: "jdoe",
      role: "user",
    });
  });
  it("includes a non-blank email", () => {
    expect(buildCreateUserBody("jdoe", " jdoe@example.com ", "admin")).toEqual({
      username: "jdoe",
      email: "jdoe@example.com",
      role: "admin",
    });
  });
  it("omits a blank email", () => {
    const body = buildCreateUserBody("jdoe", "   ", "user");
    expect("email" in body).toBe(false);
  });
});
