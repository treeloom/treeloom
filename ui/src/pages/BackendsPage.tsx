import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import { relativeTime } from "@/api/jobGroups";
import {
  addBackend,
  deleteBackend,
  fetchBackends,
  validateBackend,
  type BackendListResponse,
  type BackendRow,
} from "@/api/embeddingBackends";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";

const QK = ["embedding-backends"] as const;

export function BackendsPage() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: QK,
    queryFn: fetchBackends,
    retry: false,
    staleTime: 30_000,
  });

  const [url, setUrl] = useState("");
  const [klass, setKlass] = useState("gpu");
  const [formError, setFormError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => addBackend({ url: url.trim(), klass }),
    onSuccess: () => {
      setUrl("");
      void qc.invalidateQueries({ queryKey: QK });
    },
  });

  const remove = useMutation({
    mutationFn: (id: string) => deleteBackend(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: QK }),
  });

  function onAdd(e: React.FormEvent) {
    e.preventDefault();
    const err = validateBackend(url, klass);
    setFormError(err);
    if (err) return;
    create.reset();
    create.mutate();
  }

  // Admin-gated endpoint: a 401/403 means "not authorized", not a real error.
  const forbidden =
    query.error instanceof ApiError &&
    (query.error.status === 401 || query.error.status === 403);

  return (
    <section>
      <div>
        <h1 className="text-xl font-semibold tracking-tight">
          Embedding Backends
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          TEI embedding backends in the Postgres registry (admin only).
        </p>
      </div>

      {forbidden ? (
        <AdminRequired />
      ) : query.isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading backends…</p>
      ) : query.isError ? (
        <p
          role="alert"
          className="mt-6 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {query.error instanceof ApiError
            ? query.error.message
            : "Failed to load backends."}
        </p>
      ) : (
        <>
          <AddForm
            url={url}
            klass={klass}
            onUrl={setUrl}
            onKlass={setKlass}
            onSubmit={onAdd}
            pending={create.isPending}
            formError={formError}
            apiError={
              create.error instanceof ApiError ? create.error.message : null
            }
          />
          <BackendsTable
            data={query.data}
            onDelete={(id) => remove.mutate(id)}
            deletingId={remove.isPending ? (remove.variables ?? null) : null}
            deleteError={
              remove.error instanceof ApiError ? remove.error.message : null
            }
          />
        </>
      )}
    </section>
  );
}

function AdminRequired() {
  return (
    <p
      role="alert"
      className="mt-6 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning"
    >
      Admin access required to manage embedding backends.
    </p>
  );
}

function AddForm({
  url,
  klass,
  onUrl,
  onKlass,
  onSubmit,
  pending,
  formError,
  apiError,
}: {
  url: string;
  klass: string;
  onUrl: (v: string) => void;
  onKlass: (v: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  pending: boolean;
  formError: string | null;
  apiError: string | null;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="mt-6 flex flex-wrap items-end gap-3 rounded-lg border border-border bg-card p-4"
    >
      <div className="flex-1 space-y-1.5">
        <label htmlFor="backend-url" className="text-sm font-medium">
          Backend URL
        </label>
        <Input
          id="backend-url"
          placeholder="http://localhost:8082"
          value={url}
          onChange={(e) => onUrl(e.target.value)}
        />
      </div>
      <div className="space-y-1.5">
        <label htmlFor="backend-klass" className="text-sm font-medium">
          Class
        </label>
        <Select
          id="backend-klass"
          value={klass}
          onChange={(e) => onKlass(e.target.value)}
          className="w-28"
        >
          <option value="gpu">gpu</option>
          <option value="cpu">cpu</option>
        </Select>
      </div>
      <Button type="submit" disabled={pending}>
        {pending ? "Adding…" : "Add backend"}
      </Button>
      {(formError || apiError) && (
        <p
          role="alert"
          className="w-full text-sm text-danger"
        >
          {formError ?? apiError}
        </p>
      )}
    </form>
  );
}

function BackendsTable({
  data,
  onDelete,
  deletingId,
  deleteError,
}: {
  data: BackendListResponse | undefined;
  onDelete: (id: string) => void;
  deletingId: string | null;
  deleteError: string | null;
}) {
  const rows = data?.registry ?? [];
  const local = data?.local ?? {};

  if (rows.length === 0) {
    return (
      <p className="mt-8 text-center text-sm text-muted-foreground">
        No embedding backends registered.
      </p>
    );
  }

  return (
    <div className="mt-6">
      {deleteError && (
        <p role="alert" className="mb-3 text-sm text-danger">
          {deleteError}
        </p>
      )}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-2 font-medium">URL</th>
              <th className="px-4 py-2 font-medium">Class</th>
              <th className="px-4 py-2 font-medium">Enabled</th>
              <th className="px-4 py-2 text-right font-medium">In-flight</th>
              <th className="px-4 py-2 text-right font-medium">Failures</th>
              <th className="px-4 py-2 font-medium">Added</th>
              <th className="px-4 py-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((b: BackendRow) => {
              const counters = local[b.url];
              return (
                <tr key={b.id} className="border-t border-border">
                  <td className="px-4 py-3 font-mono text-xs">{b.url}</td>
                  <td className="px-4 py-3">
                    <Badge variant="outline">{b.klass}</Badge>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={b.enabled ? "success" : "muted"}>
                      {b.enabled ? "yes" : "no"}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {counters ? counters.in_flight_tokens.toLocaleString() : "—"}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {counters ? counters.failures.toLocaleString() : "—"}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
                    {relativeTime(Date.parse(b.created_at) / 1000)}
                  </td>
                  <td className="px-4 py-3">
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={deletingId === b.id}
                      onClick={() => onDelete(b.id)}
                    >
                      {deletingId === b.id ? "Deleting…" : "Delete"}
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
