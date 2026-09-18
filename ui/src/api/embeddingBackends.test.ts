import { describe, expect, it } from "vitest";

import { validateBackend } from "@/api/embeddingBackends";

describe("validateBackend", () => {
  it("rejects empty url", () => {
    expect(validateBackend("", "gpu")).toMatch(/url/i);
  });
  it("requires http(s) scheme", () => {
    expect(validateBackend("localhost:8082", "gpu")).toMatch(/http/i);
  });
  it("rejects a bad class", () => {
    expect(validateBackend("http://x", "tpu")).toMatch(/gpu or cpu/i);
  });
  it("accepts a valid gpu/cpu backend", () => {
    expect(validateBackend("http://localhost:8082", "gpu")).toBeNull();
    expect(validateBackend("https://x/embed", "cpu")).toBeNull();
  });
});
