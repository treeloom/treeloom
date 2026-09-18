import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import {
  addGrant,
  deleteGrant,
  fetchGrants,
  fetchGroups,
  fetchUsers,
  hasGrant,
  principalLabel,
  validateGrant,
  type GrantEffect,
  type PrincipalType,
  type SourceGrant,
} from "@/api/grants";
import { sourceLabel, type SourceRecord } from "@/api/sources";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";

/**
 * Per-source access-grants dialog. Lists the source's grants,
 * resolves principals to user/group names, and lets an admin add or revoke a
 * grant. Rendered as an inline accessible overlay (no Radix dependency).
 */
export function SourceAccessDialog({
  source,
  onClose,
}: {
  source: SourceRecord;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const grantsKey = ["source-grants", source.id] as const;

  const grantsQuery = useQuery({
    queryKey: grantsKey,
    queryFn: () => fetchGrants(source.id),
    retry: false,
  });
  const usersQuery = useQuery({
    queryKey: ["users"],
    queryFn: fetchUsers,
    retry: false,
    staleTime: 60_000,
  });
  const groupsQuery = useQuery({
    queryKey: ["groups"],
    queryFn: fetchGroups,
    retry: false,
    staleTime: 60_000,
  });

  const [principalType, setPrincipalType] = useState<PrincipalType>("user");
  const [principalId, setPrincipalId] = useState("");
  const [effect, setEffect] = useState<GrantEffect>("allow");
  const [formError, setFormError] = useState<string | null>(null);

  const users = usersQuery.data ?? [];
  const groups = groupsQuery.data ?? [];
  const grants = grantsQuery.data ?? [];

  const add = useMutation({
    mutationFn: () =>
      addGrant(source.id, { principal_type: principalType, principal_id: principalId, effect }),
    onSuccess: () => {
      setPrincipalId("");
      void qc.invalidateQueries({ queryKey: grantsKey });
    },
  });

  const revoke = useMutation({
    mutationFn: (g: SourceGrant) =>
      deleteGrant(source.id, g.principal_type, g.principal_id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: grantsKey }),
  });

  // Escape closes the dialog.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  function onAdd(e: React.FormEvent) {
    e.preventDefault();
    const err = validateGrant(principalType, principalId, effect);
    if (!err && hasGrant(grants, principalType, principalId)) {
      setFormError("That principal already has a grant.");
      return;
    }
    setFormError(err);
    if (err) return;
    add.reset();
    add.mutate();
  }

  const forbidden =
    grantsQuery.error instanceof ApiError &&
    (grantsQuery.error.status === 401 || grantsQuery.error.status === 403);

  const directory = principalType === "user" ? users : groups;
  const directoryEmpty = directory.length === 0;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="source-access-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
    >
      {/* Backdrop — click to close. */}
      <button
        type="button"
        aria-label="Close dialog"
        tabIndex={-1}
        onClick={onClose}
        className="absolute inset-0 cursor-default bg-black/60"
      />

      <div className="relative z-10 w-full max-w-lg rounded-lg border border-border bg-card p-5 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2
              id="source-access-title"
              className="text-lg font-semibold tracking-tight"
            >
              Access
            </h2>
            <p className="mt-1 truncate text-sm text-muted-foreground" title={sourceLabel(source)}>
              {sourceLabel(source)}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
            Close
          </Button>
        </div>

        {forbidden ? (
          <p
            role="alert"
            className="mt-5 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning"
          >
            Admin access required to manage grants.
          </p>
        ) : grantsQuery.isLoading ? (
          <p className="mt-5 text-sm text-muted-foreground">Loading grants…</p>
        ) : grantsQuery.isError ? (
          <p
            role="alert"
            className="mt-5 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
          >
            {grantsQuery.error instanceof ApiError
              ? grantsQuery.error.message
              : "Failed to load grants."}
          </p>
        ) : (
          <>
            <GrantsList
              grants={grants}
              users={users}
              groups={groups}
              onRevoke={(g) => revoke.mutate(g)}
              revoking={revoke.isPending ? (revoke.variables ?? null) : null}
              revokeError={
                revoke.error instanceof ApiError ? revoke.error.message : null
              }
            />

            <AddGrantForm
              principalType={principalType}
              principalId={principalId}
              effect={effect}
              directory={directory.map((d) => ({
                id: d.id,
                label: "username" in d ? d.username : d.name,
              }))}
              directoryEmpty={directoryEmpty}
              onPrincipalType={(t) => {
                setPrincipalType(t);
                setPrincipalId("");
                setFormError(null);
              }}
              onPrincipalId={setPrincipalId}
              onEffect={setEffect}
              onSubmit={onAdd}
              pending={add.isPending}
              error={
                formError ??
                (add.error instanceof ApiError ? add.error.message : null)
              }
            />
          </>
        )}
      </div>
    </div>
  );
}

function EffectBadge({ effect }: { effect: GrantEffect }) {
  return (
    <Badge variant={effect === "allow" ? "success" : "danger"}>
      {effect === "allow" ? "Allow" : "Deny"}
    </Badge>
  );
}

function GrantsList({
  grants,
  users,
  groups,
  onRevoke,
  revoking,
  revokeError,
}: {
  grants: SourceGrant[];
  users: import("@/api/grants").UserSummary[];
  groups: import("@/api/grants").GroupSummary[];
  onRevoke: (g: SourceGrant) => void;
  revoking: SourceGrant | null;
  revokeError: string | null;
}) {
  return (
    <div className="mt-5">
      {revokeError && (
        <p role="alert" className="mb-3 text-sm text-danger">
          {revokeError}
        </p>
      )}
      {grants.length === 0 ? (
        <p className="text-sm text-muted-foreground">No grants yet.</p>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {grants.map((g) => {
            const isRevoking =
              revoking?.principal_type === g.principal_type &&
              revoking?.principal_id === g.principal_id;
            return (
              <li
                key={`${g.principal_type}:${g.principal_id}`}
                className="flex items-center justify-between gap-3 px-3 py-2"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <Badge variant="outline">{g.principal_type}</Badge>
                  <span className="truncate text-sm font-medium">
                    {principalLabel(g.principal_type, g.principal_id, users, groups)}
                  </span>
                  <EffectBadge effect={g.effect} />
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={isRevoking}
                  onClick={() => onRevoke(g)}
                >
                  {isRevoking ? "Revoking…" : "Revoke"}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function AddGrantForm({
  principalType,
  principalId,
  effect,
  directory,
  directoryEmpty,
  onPrincipalType,
  onPrincipalId,
  onEffect,
  onSubmit,
  pending,
  error,
}: {
  principalType: PrincipalType;
  principalId: string;
  effect: GrantEffect;
  directory: { id: string; label: string }[];
  directoryEmpty: boolean;
  onPrincipalType: (t: PrincipalType) => void;
  onPrincipalId: (id: string) => void;
  onEffect: (e: GrantEffect) => void;
  onSubmit: (e: React.FormEvent) => void;
  pending: boolean;
  error: string | null;
}) {
  const emptyMsg =
    principalType === "user" ? "No users to grant." : "No groups to grant.";
  return (
    <form onSubmit={onSubmit} className="mt-5 space-y-3 border-t border-border pt-4">
      <p className="text-sm font-medium">Grant access</p>
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <label htmlFor="grant-type" className="text-xs text-muted-foreground">
            Type
          </label>
          <Select
            id="grant-type"
            aria-label="Principal type"
            value={principalType}
            onChange={(e) => onPrincipalType(e.target.value as PrincipalType)}
            className="w-28"
          >
            <option value="user">user</option>
            <option value="group">group</option>
          </Select>
        </div>

        <div className="flex-1 space-y-1.5">
          <label htmlFor="grant-principal" className="text-xs text-muted-foreground">
            {principalType === "user" ? "User" : "Group"}
          </label>
          <Select
            id="grant-principal"
            aria-label="Principal"
            value={principalId}
            disabled={directoryEmpty}
            onChange={(e) => onPrincipalId(e.target.value)}
          >
            <option value="">
              {directoryEmpty
                ? emptyMsg
                : `Choose a ${principalType}…`}
            </option>
            {directory.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </Select>
        </div>

        <div className="space-y-1.5">
          <label htmlFor="grant-effect" className="text-xs text-muted-foreground">
            Effect
          </label>
          <Select
            id="grant-effect"
            aria-label="Effect"
            value={effect}
            onChange={(e) => onEffect(e.target.value as GrantEffect)}
            className="w-28"
          >
            <option value="allow">allow</option>
            <option value="deny">deny</option>
          </Select>
        </div>

        <Button type="submit" disabled={pending || directoryEmpty}>
          {pending ? "Granting…" : "Grant access"}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </form>
  );
}
