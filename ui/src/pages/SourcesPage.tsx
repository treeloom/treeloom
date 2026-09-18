import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import { fetchAuthMe, isAdmin, type AuthMe } from "@/api/auth";
import { relativeTime } from "@/api/jobGroups";
import {
  fetchSources,
  fetchStaleness,
  filterSources,
  sourceLabel,
  stalenessLabel,
  type SourceRecord,
  type SourceStaleness,
} from "@/api/sources";
import { SourceAccessDialog } from "@/components/sources/SourceAccessDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

/** Per-row staleness UI state. Staleness is fetched lazily, never on load. */
type StalenessState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; data: SourceStaleness }
  | { kind: "error"; message: string };

/**
 * Resolve whether the caller is an admin, to gate the per-row "Access" action.
 *
 * This reads the app-wide ["auth","me"] query CACHE (populated by the Users /
 * Tokens tabs) without ever triggering a fetch of its own — so the Sources
 * catalog never issues an extra request on load. When the cache is cold the
 * "Access" column is simply hidden; the dialog independently enforces admin
 * (a 401/403 renders "Admin access required"), so nothing leaks either way.
 */
function useIsAdmin(): boolean {
  const qc = useQueryClient();
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: fetchAuthMe,
    retry: false,
    staleTime: 60_000,
    enabled: false, // never fetch from this page — cache-only.
  });
  const cached = me.data ?? qc.getQueryData<AuthMe>(["auth", "me"]);
  return isAdmin(cached);
}

export function SourcesPage() {
  const query = useQuery({
    queryKey: ["sources"],
    queryFn: fetchSources,
    // A catalog, not live activity — don't poll. Manual refresh below.
    staleTime: 60_000,
  });

  const admin = useIsAdmin();

  const [filter, setFilter] = useState("");
  // The source whose access dialog is open, or null.
  const [accessSource, setAccessSource] = useState<SourceRecord | null>(null);
  // Opt-in staleness results, keyed by source id. A row only appears here
  // after the user clicks "Check" — initial load fetches nothing.
  const [staleness, setStaleness] = useState<Record<string, StalenessState>>(
    {},
  );

  const sources = query.data ?? [];
  const visible = useMemo(
    () => filterSources(sources, filter),
    [sources, filter],
  );

  async function checkStaleness(id: string) {
    setStaleness((prev) => ({ ...prev, [id]: { kind: "loading" } }));
    try {
      const data = await fetchStaleness(id);
      setStaleness((prev) => ({ ...prev, [id]: { kind: "done", data } }));
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Staleness check failed.";
      setStaleness((prev) => ({ ...prev, [id]: { kind: "error", message } }));
    }
  }

  async function checkVisible() {
    // Fire the live git checks only for the currently-filtered rows.
    await Promise.all(visible.map((s) => checkStaleness(s.id)));
  }

  return (
    <section>
      <Header
        onRefresh={() => query.refetch()}
        refreshing={query.isFetching}
      />

      {query.isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading sources…</p>
      ) : query.isError ? (
        <p
          role="alert"
          className="mt-6 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {query.error instanceof ApiError
            ? query.error.message
            : "Failed to load sources."}
        </p>
      ) : sources.length === 0 ? (
        <p className="mt-8 text-center text-sm text-muted-foreground">
          No indexed sources.
        </p>
      ) : (
        <>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Input
              type="search"
              placeholder="Filter by path, url, or branch…"
              aria-label="Filter sources"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="max-w-sm"
            />
            <span className="text-xs tabular-nums text-muted-foreground">
              {visible.length} / {sources.length}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void checkVisible()}
              disabled={visible.length === 0}
            >
              Check visible staleness
            </Button>
          </div>

          {visible.length === 0 ? (
            <p className="mt-8 text-center text-sm text-muted-foreground">
              No sources match “{filter}”.
            </p>
          ) : (
            <SourcesTable
              sources={visible}
              staleness={staleness}
              onCheck={(id) => void checkStaleness(id)}
              admin={admin}
              onAccess={(s) => setAccessSource(s)}
            />
          )}
        </>
      )}

      {accessSource ? (
        <SourceAccessDialog
          source={accessSource}
          onClose={() => setAccessSource(null)}
        />
      ) : null}
    </section>
  );
}

function Header({
  onRefresh,
  refreshing,
}: {
  onRefresh: () => void;
  refreshing: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Sources</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Indexed sources catalog.
        </p>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onRefresh}
        disabled={refreshing}
      >
        {refreshing ? "Refreshing…" : "Refresh"}
      </Button>
    </div>
  );
}

function SourcesTable({
  sources,
  staleness,
  onCheck,
  admin,
  onAccess,
}: {
  sources: SourceRecord[];
  staleness: Record<string, StalenessState>;
  onCheck: (id: string) => void;
  admin: boolean;
  onAccess: (source: SourceRecord) => void;
}) {
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-4 py-2 font-medium">Source</th>
            <th className="px-4 py-2 font-medium">Indexed</th>
            <th className="px-4 py-2 text-right font-medium">Files</th>
            <th className="px-4 py-2 text-right font-medium">Chunks</th>
            <th className="px-4 py-2 font-medium">Graph</th>
            <th className="px-4 py-2 font-medium">Staleness</th>
            {admin ? (
              <th className="px-4 py-2 font-medium">Access</th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {sources.map((s) => (
            <SourceRow
              key={s.id}
              source={s}
              state={staleness[s.id] ?? { kind: "idle" }}
              onCheck={() => onCheck(s.id)}
              admin={admin}
              onAccess={() => onAccess(s)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SourceRow({
  source,
  state,
  onCheck,
  admin,
  onAccess,
}: {
  source: SourceRecord;
  state: StalenessState;
  onCheck: () => void;
  admin: boolean;
  onAccess: () => void;
}) {
  const label = sourceLabel(source);
  const indexedAtMs = source.indexed_at * 1000;
  return (
    <tr className="border-t border-border align-top">
      <td className="px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-medium" title={label}>
            {label}
          </span>
          {source.branch ? (
            <Badge variant="outline" className="shrink-0">
              {source.branch}
            </Badge>
          ) : null}
        </div>
      </td>
      <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
        <span title={new Date(indexedAtMs).toLocaleString()}>
          {relativeTime(source.indexed_at)}
        </span>
      </td>
      <td className="px-4 py-3 text-right tabular-nums">
        {source.file_count.toLocaleString()}
      </td>
      <td className="px-4 py-3 text-right tabular-nums">
        {source.chunk_count.toLocaleString()}
      </td>
      <td className="px-4 py-3">
        <Badge variant={source.graph_indexed ? "success" : "muted"}>
          {source.graph_indexed ? "yes" : "no"}
        </Badge>
      </td>
      <td className="px-4 py-3">
        <StalenessCell state={state} onCheck={onCheck} />
      </td>
      {admin ? (
        <td className="px-4 py-3">
          <Button variant="outline" size="sm" onClick={onAccess}>
            Access
          </Button>
        </td>
      ) : null}
    </tr>
  );
}

function StalenessCell({
  state,
  onCheck,
}: {
  state: StalenessState;
  onCheck: () => void;
}) {
  if (state.kind === "loading") {
    return <span className="text-xs text-muted-foreground">Checking…</span>;
  }
  if (state.kind === "error") {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-danger" role="alert">
          {state.message}
        </span>
        <Button variant="ghost" size="sm" onClick={onCheck}>
          Retry
        </Button>
      </div>
    );
  }
  if (state.kind === "done") {
    const label = stalenessLabel(state.data);
    const variant =
      label === "stale" ? "warning" : label === "fresh" ? "success" : "muted";
    return <Badge variant={variant}>{label}</Badge>;
  }
  // idle — staleness has not been requested for this row.
  return (
    <Button variant="ghost" size="sm" onClick={onCheck}>
      Check
    </Button>
  );
}
