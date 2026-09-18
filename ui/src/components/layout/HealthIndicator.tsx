import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { cn } from "@/lib/utils";

/**
 * Top-bar indexer health indicator.
 *
 * Polls the indexer `GET /health` via the typed API client. Green on success,
 * red on error, grey while loading. No live indexer is reachable in CI/build,
 * so the dot shows loading/down there — that is expected.
 */
type IndexerStatus = "ok" | "degraded" | "down" | "unknown";

const DOT_COLOR: Record<IndexerStatus, string> = {
  ok: "bg-success",
  degraded: "bg-warning",
  down: "bg-danger",
  unknown: "bg-muted-foreground",
};

/** Shape of the indexer `/health` response (only `status` is consumed). */
interface HealthResponse {
  status?: string;
}

/** Maps the query state to a discrete dot status. Pure for testability. */
export function healthStatus(state: {
  isLoading: boolean;
  isError: boolean;
  data?: HealthResponse;
}): IndexerStatus {
  if (state.isLoading) return "unknown";
  if (state.isError || !state.data) return "down";
  if (state.data.status && state.data.status !== "ok") return "degraded";
  return "ok";
}

export function HealthIndicator() {
  const query = useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<HealthResponse>("/health"),
    refetchInterval: 30_000,
  });

  const status = healthStatus(query);

  return (
    <div className="flex items-center gap-2" aria-live="polite">
      <span
        className={cn("h-2 w-2 rounded-full", DOT_COLOR[status])}
        aria-hidden="true"
      />
      <span className="text-[13px] text-muted-foreground">
        indexer: {status}
      </span>
    </div>
  );
}
