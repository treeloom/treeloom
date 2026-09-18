import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import { fetchAuthMe, isAdmin, userId } from "@/api/auth";
import { relativeTime } from "@/api/jobGroups";
import {
  createApiKey,
  createToken,
  fetchApiKeys,
  fetchTokens,
  revokeApiKey,
  revokeToken,
  toggleScope,
  validateTokenName,
  TOKEN_SCOPES,
  type ApiKeyRow,
  type PatRow,
} from "@/api/tokens";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { SecretReveal } from "@/components/ui/secret-reveal";

export function TokensPage() {
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: fetchAuthMe,
    retry: false,
    staleTime: 60_000,
  });

  const uid = userId(me.data);
  const admin = isAdmin(me.data);

  return (
    <section>
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Access Tokens</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Your personal API keys and (admin) system API keys for programmatic /
          CLI access.
        </p>
      </div>

      {me.isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading…</p>
      ) : uid ? (
        <MyTokens userId={uid} />
      ) : (
        <p
          role="alert"
          className="mt-6 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning"
        >
          Sign in to manage your API keys.
        </p>
      )}

      {admin && <ApiKeys />}
    </section>
  );
}

// --------------------------------------------------------------------------
// My Tokens (PATs)
// --------------------------------------------------------------------------

function MyTokens({ userId: uid }: { userId: string }) {
  const qc = useQueryClient();
  const qk = ["tokens", uid] as const;
  const query = useQuery({
    queryKey: qk,
    queryFn: () => fetchTokens(uid),
    retry: false,
  });

  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<string[]>(["search"]);
  const [formError, setFormError] = useState<string | null>(null);
  const [secret, setSecret] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => createToken(uid, { name: name.trim(), scopes }),
    onSuccess: (data) => {
      setSecret(data.token);
      setName("");
      void qc.invalidateQueries({ queryKey: qk });
    },
  });

  const revoke = useMutation({
    mutationFn: (id: string) => revokeToken(uid, id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk }),
  });

  function onCreate(e: React.FormEvent) {
    e.preventDefault();
    const err = validateTokenName(name);
    setFormError(err);
    if (err) return;
    create.reset();
    create.mutate();
  }

  return (
    <div className="mt-8">
      <h2 className="text-base font-semibold">My API keys</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Personal access tokens scoped to your account — use them for
        programmatic / CLI access.
      </p>

      <form
        onSubmit={onCreate}
        className="mt-4 space-y-3 rounded-lg border border-border bg-card p-4"
      >
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 space-y-1.5">
            <label htmlFor="pat-name" className="text-sm font-medium">
              Token name
            </label>
            <Input
              id="pat-name"
              placeholder="my-laptop"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <Button type="submit" disabled={create.isPending}>
            {create.isPending ? "Creating…" : "Create API key"}
          </Button>
        </div>
        <ScopePicker
          idPrefix="pat"
          scopes={scopes}
          onToggle={(s) => setScopes((prev) => toggleScope(prev, s))}
        />
        {(formError ||
          (create.error instanceof ApiError && create.error.message)) && (
          <p role="alert" className="text-sm text-danger">
            {formError ??
              (create.error instanceof ApiError ? create.error.message : "")}
          </p>
        )}
      </form>

      {secret && (
        <SecretReveal
          secret={secret}
          label="API key"
          onDismiss={() => setSecret(null)}
        />
      )}

      {query.isLoading ? (
        <p className="mt-4 text-sm text-muted-foreground">Loading tokens…</p>
      ) : query.isError ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {query.error instanceof ApiError
            ? query.error.message
            : "Failed to load tokens."}
        </p>
      ) : (
        <PatTable
          rows={query.data ?? []}
          onRevoke={(id) => revoke.mutate(id)}
          revokingId={revoke.isPending ? (revoke.variables ?? null) : null}
        />
      )}
    </div>
  );
}

function PatTable({
  rows,
  onRevoke,
  revokingId,
}: {
  rows: PatRow[];
  onRevoke: (id: string) => void;
  revokingId: string | null;
}) {
  if (rows.length === 0) {
    return (
      <p className="mt-6 text-center text-sm text-muted-foreground">
        No API keys.
      </p>
    );
  }
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-4 py-2 font-medium">Name</th>
            <th className="px-4 py-2 font-medium">Scopes</th>
            <th className="px-4 py-2 font-medium">Created</th>
            <th className="px-4 py-2 font-medium">Last used</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.id} className="border-t border-border">
              <td className="px-4 py-3 font-medium">{t.name}</td>
              <td className="px-4 py-3">
                <ScopeBadges scopes={t.scopes} />
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
                {relativeTime(Date.parse(t.created_at) / 1000)}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
                {t.last_used_at
                  ? relativeTime(Date.parse(t.last_used_at) / 1000)
                  : "never"}
              </td>
              <td className="px-4 py-3">
                <Badge variant={t.revoked ? "muted" : "success"}>
                  {t.revoked ? "revoked" : "active"}
                </Badge>
              </td>
              <td className="px-4 py-3">
                {!t.revoked && (
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={revokingId === t.id}
                    onClick={() => onRevoke(t.id)}
                  >
                    {revokingId === t.id ? "Revoking…" : "Revoke"}
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --------------------------------------------------------------------------
// API Keys (admin)
// --------------------------------------------------------------------------

function ApiKeys() {
  const qc = useQueryClient();
  const qk = ["api-keys"] as const;
  const query = useQuery({
    queryKey: qk,
    queryFn: fetchApiKeys,
    retry: false,
  });

  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<string[]>(["search"]);
  const [allAccess, setAllAccess] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [secret, setSecret] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      createApiKey({ name: name.trim(), scopes, all_access: allAccess }),
    onSuccess: (data) => {
      setSecret(data.key);
      setName("");
      void qc.invalidateQueries({ queryKey: qk });
    },
  });

  const revoke = useMutation({
    mutationFn: (id: string) => revokeApiKey(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk }),
  });

  function onCreate(e: React.FormEvent) {
    e.preventDefault();
    const err = validateTokenName(name);
    setFormError(err);
    if (err) return;
    create.reset();
    create.mutate();
  }

  return (
    <div className="mt-10">
      <h2 className="text-base font-semibold">System API keys</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Service credentials (admin only) — not tied to a single user.
      </p>

      <form
        onSubmit={onCreate}
        className="mt-4 space-y-3 rounded-lg border border-border bg-card p-4"
      >
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 space-y-1.5">
            <label htmlFor="key-name" className="text-sm font-medium">
              Key name
            </label>
            <Input
              id="key-name"
              placeholder="ci-pipeline"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <Button type="submit" disabled={create.isPending}>
            {create.isPending ? "Creating…" : "Create API key"}
          </Button>
        </div>
        <ScopePicker
          idPrefix="key"
          scopes={scopes}
          onToggle={(s) => setScopes((prev) => toggleScope(prev, s))}
        />
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={allAccess}
            onChange={(e) => setAllAccess(e.target.checked)}
          />
          <span>all_access (read every source)</span>
        </label>
        {(formError ||
          (create.error instanceof ApiError && create.error.message)) && (
          <p role="alert" className="text-sm text-danger">
            {formError ??
              (create.error instanceof ApiError ? create.error.message : "")}
          </p>
        )}
      </form>

      {secret && (
        <SecretReveal
          secret={secret}
          label="API key"
          onDismiss={() => setSecret(null)}
        />
      )}

      {query.isLoading ? (
        <p className="mt-4 text-sm text-muted-foreground">Loading keys…</p>
      ) : query.isError ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {query.error instanceof ApiError
            ? query.error.message
            : "Failed to load API keys."}
        </p>
      ) : (
        <ApiKeyTable
          rows={query.data ?? []}
          onRevoke={(id) => revoke.mutate(id)}
          revokingId={revoke.isPending ? (revoke.variables ?? null) : null}
        />
      )}
    </div>
  );
}

function ApiKeyTable({
  rows,
  onRevoke,
  revokingId,
}: {
  rows: ApiKeyRow[];
  onRevoke: (id: string) => void;
  revokingId: string | null;
}) {
  if (rows.length === 0) {
    return (
      <p className="mt-6 text-center text-sm text-muted-foreground">
        No API keys.
      </p>
    );
  }
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-4 py-2 font-medium">Name</th>
            <th className="px-4 py-2 font-medium">Scopes</th>
            <th className="px-4 py-2 font-medium">all_access</th>
            <th className="px-4 py-2 font-medium">Created</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((k) => (
            <tr key={k.id} className="border-t border-border">
              <td className="px-4 py-3 font-medium">{k.name}</td>
              <td className="px-4 py-3">
                <ScopeBadges scopes={k.scopes} />
              </td>
              <td className="px-4 py-3">
                <Badge variant={k.all_access ? "success" : "muted"}>
                  {k.all_access ? "yes" : "no"}
                </Badge>
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
                {relativeTime(Date.parse(k.created_at) / 1000)}
              </td>
              <td className="px-4 py-3">
                <Badge variant={k.revoked ? "muted" : "success"}>
                  {k.revoked ? "revoked" : "active"}
                </Badge>
              </td>
              <td className="px-4 py-3">
                {!k.revoked && (
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={revokingId === k.id}
                    onClick={() => onRevoke(k.id)}
                  >
                    {revokingId === k.id ? "Revoking…" : "Revoke"}
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --------------------------------------------------------------------------
// Shared bits
// --------------------------------------------------------------------------

function ScopePicker({
  idPrefix,
  scopes,
  onToggle,
}: {
  idPrefix: string;
  scopes: string[];
  onToggle: (scope: string) => void;
}) {
  return (
    <fieldset>
      <legend className="text-sm font-medium">Scopes</legend>
      <div className="mt-2 flex flex-wrap gap-4">
        {TOKEN_SCOPES.map((s) => {
          const id = `${idPrefix}-scope-${s}`;
          return (
            <label key={s} htmlFor={id} className="flex items-center gap-2 text-sm">
              <Checkbox
                id={id}
                checked={scopes.includes(s)}
                onChange={() => onToggle(s)}
              />
              <span>{s}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

function ScopeBadges({ scopes }: { scopes: string[] }) {
  if (scopes.length === 0) {
    return <span className="text-xs text-muted-foreground">none</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {scopes.map((s) => (
        <Badge key={s} variant="outline">
          {s}
        </Badge>
      ))}
    </div>
  );
}
