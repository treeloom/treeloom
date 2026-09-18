import { describe, expect, it } from "vitest";

import {
  filterSources,
  sourceLabel,
  stalenessLabel,
  type SourceRecord,
  type SourceStaleness,
} from "@/api/sources";

function src(over: Partial<SourceRecord>): SourceRecord {
  return {
    id: "sha1abc",
    path: "/home/u/repo",
    url: "",
    branch: "main",
    indexed_at: 1_750_000_000,
    file_count: 10,
    chunk_count: 20,
    commit_sha: "abc123",
    graph_indexed: true,
    ...over,
  };
}

describe("sourceLabel", () => {
  it("prefers path", () => {
    expect(sourceLabel(src({ path: "/a/b", url: "http://x" }))).toBe("/a/b");
  });

  it("falls back to url for url-only sources", () => {
    expect(sourceLabel(src({ path: "", url: "https://git/x" }))).toBe(
      "https://git/x",
    );
  });

  it("falls back to id when both path and url are empty", () => {
    expect(sourceLabel(src({ path: "", url: "", id: "sha_z" }))).toBe("sha_z");
  });
});

describe("filterSources", () => {
  const sources = [
    src({ id: "1", path: "/home/u/treeloom", url: "", branch: "main" }),
    src({ id: "2", path: "", url: "https://github.com/acme/widgets", branch: "dev" }),
    src({ id: "3", path: "/srv/code/featbit", url: "", branch: "release" }),
  ];

  it("returns all rows for an empty query", () => {
    expect(filterSources(sources, "")).toEqual(sources);
    expect(filterSources(sources, "   ")).toEqual(sources);
  });

  it("matches on path, case-insensitively", () => {
    const out = filterSources(sources, "TREELOOM");
    expect(out.map((s) => s.id)).toEqual(["1"]);
  });

  it("matches on url (url-only source)", () => {
    const out = filterSources(sources, "github.com/acme");
    expect(out.map((s) => s.id)).toEqual(["2"]);
  });

  it("matches on branch", () => {
    const out = filterSources(sources, "release");
    expect(out.map((s) => s.id)).toEqual(["3"]);
  });

  it("returns an empty array when nothing matches", () => {
    expect(filterSources(sources, "nope-zzz")).toEqual([]);
  });
});

describe("stalenessLabel", () => {
  function st(over: Partial<SourceStaleness>): SourceStaleness {
    return { indexed_sha: "a", current_sha: "b", is_stale: false, ...over };
  }

  it("returns 'n/a' when is_stale is null (non-git source)", () => {
    expect(stalenessLabel(st({ is_stale: null }))).toBe("n/a");
  });

  it("returns 'stale' when is_stale is true", () => {
    expect(stalenessLabel(st({ is_stale: true }))).toBe("stale");
  });

  it("returns 'fresh' when is_stale is false", () => {
    expect(stalenessLabel(st({ is_stale: false }))).toBe("fresh");
  });
});
