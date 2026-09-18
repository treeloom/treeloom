import { describe, expect, it, vi } from "vitest";

import { notifyAuthRequired, onAuthRequired } from "@/auth/authEvents";

describe("authEvents bus", () => {
  it("delivers notify to all current subscribers", () => {
    const a = vi.fn();
    const b = vi.fn();
    const offA = onAuthRequired(a);
    const offB = onAuthRequired(b);

    notifyAuthRequired();

    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
    offA();
    offB();
  });

  it("stops delivering after unsubscribe", () => {
    const a = vi.fn();
    const off = onAuthRequired(a);
    off();

    notifyAuthRequired();

    expect(a).not.toHaveBeenCalled();
  });
});
