import { describe, expect, it } from "vitest";

import {
  buildBody,
  buildRepoEntryBody,
  defaultJobLabel,
  emptyForm,
  endpointFor,
  parseRepoLines,
  validateForm,
  type RepoEntry,
  type SubmitFormState,
} from "@/api/submit";

function form(over: Partial<SubmitFormState>): SubmitFormState {
  return { ...emptyForm(), ...over };
}

describe("endpointFor", () => {
  it("maps each kind to its endpoint", () => {
    expect(endpointFor("repo")).toBe("/index-repo");
    expect(endpointFor("directory")).toBe("/index-directory");
    expect(endpointFor("file")).toBe("/index-file");
  });
});

describe("validateForm", () => {
  it("requires a repo path in path mode", () => {
    expect(validateForm(form({ kind: "repo", repoMode: "path", path: "" }))).toMatch(
      /repository path/i,
    );
    expect(
      validateForm(form({ kind: "repo", repoMode: "path", path: "/r" })),
    ).toBeNull();
  });

  it("requires a url in url mode", () => {
    expect(
      validateForm(form({ kind: "repo", repoMode: "url", url: "" })),
    ).toMatch(/url/i);
    expect(
      validateForm(form({ kind: "repo", repoMode: "url", url: "http://x" })),
    ).toBeNull();
  });

  it("requires a directory path", () => {
    expect(validateForm(form({ kind: "directory", path: "" }))).toMatch(
      /directory/i,
    );
    expect(validateForm(form({ kind: "directory", path: "/d" }))).toBeNull();
  });

  it("requires a file path", () => {
    expect(validateForm(form({ kind: "file", path: "" }))).toMatch(/file/i);
    expect(validateForm(form({ kind: "file", path: "/f.py" }))).toBeNull();
  });
});

describe("buildBody", () => {
  it("repo path mode → { path, force, skip_graph }", () => {
    const body = buildBody(
      form({ kind: "repo", repoMode: "path", path: "  /r  ", force: true }),
    );
    expect(body).toEqual({ path: "/r", force: true, skip_graph: false });
  });

  it("repo url mode omits blank branch, includes filled branch", () => {
    expect(
      buildBody(form({ kind: "repo", repoMode: "url", url: "http://x" })),
    ).toEqual({ url: "http://x", force: false, skip_graph: false });
    expect(
      buildBody(
        form({
          kind: "repo",
          repoMode: "url",
          url: "http://x",
          branch: "dev",
        }),
      ),
    ).toEqual({ url: "http://x", branch: "dev", force: false, skip_graph: false });
  });

  it("directory → { directory } and omits blank pattern", () => {
    expect(buildBody(form({ kind: "directory", path: "/d" }))).toEqual({
      directory: "/d",
      force: false,
      skip_graph: false,
    });
    expect(
      buildBody(form({ kind: "directory", path: "/d", pattern: "**/*.py" })),
    ).toEqual({
      directory: "/d",
      pattern: "**/*.py",
      force: false,
      skip_graph: false,
    });
  });

  it("file → { file_path }", () => {
    expect(
      buildBody(form({ kind: "file", path: "/f.py", skipGraph: true })),
    ).toEqual({ file_path: "/f.py", force: false, skip_graph: true });
  });
});

describe("parseRepoLines", () => {
  it("treats scheme/git@ lines as URLs and others as paths", () => {
    expect(parseRepoLines("https://github.com/org/repo")).toEqual([
      { url: "https://github.com/org/repo" },
    ]);
    expect(parseRepoLines("git@github.com:org/repo.git")).toEqual([
      { url: "git@github.com:org/repo.git" },
    ]);
    expect(parseRepoLines("git://host/repo")).toEqual([
      { url: "git://host/repo" },
    ]);
    expect(parseRepoLines("ssh://git@host/repo")).toEqual([
      { url: "ssh://git@host/repo" },
    ]);
    expect(parseRepoLines("/home/user/source/myrepo")).toEqual([
      { path: "/home/user/source/myrepo" },
    ]);
  });

  it("takes an optional branch as the 2nd token on URL lines", () => {
    expect(parseRepoLines("https://github.com/org/repo main")).toEqual([
      { url: "https://github.com/org/repo", branch: "main" },
    ]);
    // extra tokens beyond branch are ignored
    expect(parseRepoLines("https://github.com/org/repo dev extra")).toEqual([
      { url: "https://github.com/org/repo", branch: "dev" },
    ]);
  });

  it("keeps spaces in path lines (does not split paths)", () => {
    expect(parseRepoLines("/home/user/my repo dir")).toEqual([
      { path: "/home/user/my repo dir" },
    ]);
  });

  it("drops blank lines and # comments, and trims", () => {
    const text = [
      "",
      "  # a comment",
      "  /a  ",
      "",
      "https://github.com/org/repo ",
      "# trailing comment",
    ].join("\n");
    expect(parseRepoLines(text)).toEqual([
      { path: "/a" },
      { url: "https://github.com/org/repo" },
    ]);
  });

  it("deduplicates identical entries (path and url+branch)", () => {
    expect(parseRepoLines("/a\n/a\n/b")).toEqual([{ path: "/a" }, { path: "/b" }]);
    expect(
      parseRepoLines(
        "https://x/r main\nhttps://x/r main\nhttps://x/r dev",
      ),
    ).toEqual([
      { url: "https://x/r", branch: "main" },
      { url: "https://x/r", branch: "dev" },
    ]);
  });

  it("returns [] for empty / comments-only input", () => {
    expect(parseRepoLines("")).toEqual([]);
    expect(parseRepoLines("\n# only\n  \n")).toEqual([]);
  });
});

describe("buildRepoEntryBody", () => {
  const opts = { force: true, skipGraph: false, groupId: "grp_1" };

  it("path entry → { path, force, skip_graph, group_id }", () => {
    expect(buildRepoEntryBody({ path: "/r" }, opts)).toEqual({
      path: "/r",
      force: true,
      skip_graph: false,
      group_id: "grp_1",
    });
  });

  it("url entry omits blank branch, includes a filled branch", () => {
    expect(buildRepoEntryBody({ url: "http://x" }, opts)).toEqual({
      url: "http://x",
      force: true,
      skip_graph: false,
      group_id: "grp_1",
    });
    expect(
      buildRepoEntryBody({ url: "http://x", branch: "dev" }, opts),
    ).toEqual({
      url: "http://x",
      branch: "dev",
      force: true,
      skip_graph: false,
      group_id: "grp_1",
    });
  });

  it("never emits both path and url (xor)", () => {
    const pathBody = buildRepoEntryBody({ path: "/r" }, opts);
    expect("url" in pathBody).toBe(false);
    const urlBody = buildRepoEntryBody({ url: "http://x" }, opts);
    expect("path" in urlBody).toBe(false);
  });
});

describe("defaultJobLabel", () => {
  it("uses the single repo's basename for one entry", () => {
    expect(defaultJobLabel([{ path: "/home/u/myrepo" }])).toBe("myrepo");
    expect(
      defaultJobLabel([{ url: "https://github.com/org/cool-repo.git" }]),
    ).toBe("cool-repo");
  });

  it("uses 'N repositories' for many entries", () => {
    const entries: RepoEntry[] = [{ path: "/a" }, { path: "/b" }, { path: "/c" }];
    expect(defaultJobLabel(entries)).toBe("3 repositories");
  });
});
