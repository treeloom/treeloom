import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/api/client";
import { fetchAuthMe, isAdmin } from "@/api/auth";
import { relativeTime } from "@/api/jobGroups";
import {
  addGroupMember,
  buildCreateUserBody,
  createGroup,
  createUser,
  deactivateUser,
  deleteGroup,
  fetchGroupMembers,
  fetchGroups,
  fetchUsers,
  removeGroupMember,
  rotateUserKey,
  setGroupAllAccess,
  setUserAllAccess,
  setUserPassword,
  setUserRole,
  USER_ROLES,
  validateGroupName,
  validatePassword,
  validateUsername,
  type GroupRow,
  type UserRole,
  type UserRow,
} from "@/api/usersGroups";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { SecretReveal } from "@/components/ui/secret-reveal";

const USERS_QK = ["users"] as const;
const GROUPS_QK = ["groups"] as const;

export function UsersPage() {
  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: fetchAuthMe,
    retry: false,
    staleTime: 60_000,
  });

  return (
    <section>
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Users &amp; Groups</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage user accounts, roles, and access groups.
        </p>
      </div>

      {me.isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading…</p>
      ) : isAdmin(me.data) ? (
        <>
          <UsersSection />
          <GroupsSection />
        </>
      ) : (
        <p
          role="alert"
          className="mt-6 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning"
        >
          Admin access required.
        </p>
      )}
    </section>
  );
}

function errText(err: unknown): string | null {
  return err instanceof ApiError ? err.message : null;
}

// --------------------------------------------------------------------------
// Users
// --------------------------------------------------------------------------

function UsersSection() {
  const qc = useQueryClient();
  const query = useQuery({ queryKey: USERS_QK, queryFn: fetchUsers, retry: false });

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<UserRole>("user");
  const [formError, setFormError] = useState<string | null>(null);
  const [secret, setSecret] = useState<string | null>(null);

  const invalidate = () => void qc.invalidateQueries({ queryKey: USERS_QK });

  const create = useMutation({
    mutationFn: () =>
      createUser(buildCreateUserBody(username, email, role)),
    onSuccess: (data) => {
      setSecret(data.api_key);
      setUsername("");
      setEmail("");
      setRole("user");
      invalidate();
    },
  });

  function onCreate(e: React.FormEvent) {
    e.preventDefault();
    const err = validateUsername(username);
    setFormError(err);
    if (err) return;
    create.reset();
    create.mutate();
  }

  return (
    <div className="mt-8">
      <h2 className="text-base font-semibold">Users</h2>

      <form
        onSubmit={onCreate}
        className="mt-4 flex flex-wrap items-end gap-3 rounded-lg border border-border bg-card p-4"
      >
        <div className="flex-1 space-y-1.5">
          <label htmlFor="new-username" className="text-sm font-medium">
            Username
          </label>
          <Input
            id="new-username"
            placeholder="jdoe"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </div>
        <div className="flex-1 space-y-1.5">
          <label htmlFor="new-email" className="text-sm font-medium">
            Email
          </label>
          <Input
            id="new-email"
            type="email"
            placeholder="jdoe@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <label htmlFor="new-role" className="text-sm font-medium">
            Role
          </label>
          <Select
            id="new-role"
            value={role}
            onChange={(e) => setRole(e.target.value as UserRole)}
          >
            {USER_ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </Select>
        </div>
        <Button type="submit" disabled={create.isPending}>
          {create.isPending ? "Creating…" : "Create user"}
        </Button>
        {(formError || errText(create.error)) && (
          <p role="alert" className="w-full text-sm text-danger">
            {formError ?? errText(create.error)}
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
        <p className="mt-4 text-sm text-muted-foreground">Loading users…</p>
      ) : query.isError ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {errText(query.error) ?? "Failed to load users."}
        </p>
      ) : (
        <UserTable rows={query.data ?? []} onChanged={invalidate} onSecret={setSecret} />
      )}
    </div>
  );
}

function UserTable({
  rows,
  onChanged,
  onSecret,
}: {
  rows: UserRow[];
  onChanged: () => void;
  onSecret: (key: string) => void;
}) {
  if (rows.length === 0) {
    return (
      <p className="mt-6 text-center text-sm text-muted-foreground">
        No users yet.
      </p>
    );
  }
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-4 py-2 font-medium">Username</th>
            <th className="px-4 py-2 font-medium">Email</th>
            <th className="px-4 py-2 font-medium">Role</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium">all_access</th>
            <th className="px-4 py-2 font-medium">Created</th>
            <th className="px-4 py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((u) => (
            <UserRowView
              key={u.id}
              user={u}
              onChanged={onChanged}
              onSecret={onSecret}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UserRowView({
  user,
  onChanged,
  onSecret,
}: {
  user: UserRow;
  onChanged: () => void;
  onSecret: (key: string) => void;
}) {
  const [pwOpen, setPwOpen] = useState(false);

  const setRole = useMutation({
    mutationFn: (role: UserRole) => setUserRole(user.id, role),
    onSuccess: onChanged,
  });
  const toggleAll = useMutation({
    mutationFn: (allAccess: boolean) => setUserAllAccess(user.id, allAccess),
    onSuccess: onChanged,
  });
  const deactivate = useMutation({
    mutationFn: () => deactivateUser(user.id),
    onSuccess: onChanged,
  });
  const rotate = useMutation({
    mutationFn: () => rotateUserKey(user.id),
    onSuccess: (data) => onSecret(data.api_key),
  });

  const busy =
    setRole.isPending ||
    toggleAll.isPending ||
    deactivate.isPending ||
    rotate.isPending;

  return (
    <>
      <tr className="border-t border-border">
        <td className="px-4 py-3 font-medium">{user.username}</td>
        <td className="px-4 py-3 text-muted-foreground">{user.email || "—"}</td>
        <td className="px-4 py-3">
          <Select
            aria-label={`Role for ${user.username}`}
            value={user.role}
            disabled={busy}
            className="h-8 w-24"
            onChange={(e) => setRole.mutate(e.target.value as UserRole)}
          >
            {USER_ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </Select>
        </td>
        <td className="px-4 py-3">
          <Badge variant={user.active ? "success" : "muted"}>
            {user.active ? "active" : "inactive"}
          </Badge>
        </td>
        <td className="px-4 py-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              aria-label={`all_access for ${user.username}`}
              checked={!!user.all_access}
              disabled={busy}
              onChange={(e) => toggleAll.mutate(e.target.checked)}
            />
            <Badge variant={user.all_access ? "success" : "muted"}>
              {user.all_access ? "yes" : "no"}
            </Badge>
          </label>
        </td>
        <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">
          {relativeTime(Date.parse(user.created_at) / 1000)}
        </td>
        <td className="px-4 py-3">
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => setPwOpen((v) => !v)}
            >
              Reset password
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => rotate.mutate()}
            >
              {rotate.isPending ? "Rotating…" : "Rotate key"}
            </Button>
            {user.active && (
              <Button
                variant="destructive"
                size="sm"
                disabled={busy}
                onClick={() => deactivate.mutate()}
              >
                {deactivate.isPending ? "Deactivating…" : "Deactivate"}
              </Button>
            )}
          </div>
          {(errText(setRole.error) ||
            errText(toggleAll.error) ||
            errText(deactivate.error) ||
            errText(rotate.error)) && (
            <p role="alert" className="mt-2 text-xs text-danger">
              {errText(setRole.error) ??
                errText(toggleAll.error) ??
                errText(deactivate.error) ??
                errText(rotate.error)}
            </p>
          )}
        </td>
      </tr>
      {pwOpen && (
        <tr className="border-t border-border bg-muted/20">
          <td colSpan={7} className="px-4 py-3">
            <SetPasswordForm
              user={user}
              onDone={() => setPwOpen(false)}
            />
          </td>
        </tr>
      )}
    </>
  );
}

function SetPasswordForm({
  user,
  onDone,
}: {
  user: UserRow;
  onDone: () => void;
}) {
  const [password, setPassword] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: () => setUserPassword(user.id, password),
    onSuccess: () => {
      setPassword("");
      onDone();
    },
  });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const err = validatePassword(password);
    setFormError(err);
    if (err) return;
    save.reset();
    save.mutate();
  }

  return (
    <form onSubmit={onSubmit} className="flex flex-wrap items-end gap-3">
      <div className="flex-1 space-y-1.5">
        <label
          htmlFor={`pw-${user.id}`}
          className="text-sm font-medium"
        >
          New password for {user.username}
        </label>
        <Input
          id={`pw-${user.id}`}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>
      <Button type="submit" size="sm" disabled={save.isPending}>
        {save.isPending ? "Saving…" : "Set password"}
      </Button>
      <Button type="button" size="sm" variant="ghost" onClick={onDone}>
        Cancel
      </Button>
      {(formError || errText(save.error)) && (
        <p role="alert" className="w-full text-sm text-danger">
          {formError ?? errText(save.error)}
        </p>
      )}
    </form>
  );
}

// --------------------------------------------------------------------------
// Groups
// --------------------------------------------------------------------------

function GroupsSection() {
  const qc = useQueryClient();
  const groups = useQuery({
    queryKey: GROUPS_QK,
    queryFn: fetchGroups,
    retry: false,
  });
  const users = useQuery({ queryKey: USERS_QK, queryFn: fetchUsers, retry: false });

  const [name, setName] = useState("");
  const [allAccess, setAllAccess] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const invalidate = () => void qc.invalidateQueries({ queryKey: GROUPS_QK });

  const create = useMutation({
    mutationFn: () =>
      createGroup({ name: name.trim(), all_access: allAccess }),
    onSuccess: () => {
      setName("");
      setAllAccess(false);
      invalidate();
    },
  });

  function onCreate(e: React.FormEvent) {
    e.preventDefault();
    const err = validateGroupName(name);
    setFormError(err);
    if (err) return;
    create.reset();
    create.mutate();
  }

  return (
    <div className="mt-10">
      <h2 className="text-base font-semibold">Groups</h2>

      <form
        onSubmit={onCreate}
        className="mt-4 flex flex-wrap items-end gap-3 rounded-lg border border-border bg-card p-4"
      >
        <div className="flex-1 space-y-1.5">
          <label htmlFor="new-group" className="text-sm font-medium">
            Group name
          </label>
          <Input
            id="new-group"
            placeholder="platform-team"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <label className="flex items-center gap-2 pb-2 text-sm">
          <Checkbox
            checked={allAccess}
            onChange={(e) => setAllAccess(e.target.checked)}
          />
          <span>all_access (read every source)</span>
        </label>
        <Button type="submit" disabled={create.isPending}>
          {create.isPending ? "Creating…" : "Create group"}
        </Button>
        {(formError || errText(create.error)) && (
          <p role="alert" className="w-full text-sm text-danger">
            {formError ?? errText(create.error)}
          </p>
        )}
      </form>

      {groups.isLoading ? (
        <p className="mt-4 text-sm text-muted-foreground">Loading groups…</p>
      ) : groups.isError ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {errText(groups.error) ?? "Failed to load groups."}
        </p>
      ) : (groups.data ?? []).length === 0 ? (
        <p className="mt-6 text-center text-sm text-muted-foreground">
          No groups yet.
        </p>
      ) : (
        <div className="mt-4 space-y-3">
          {(groups.data ?? []).map((g) => (
            <GroupCard
              key={g.id}
              group={g}
              users={users.data ?? []}
              onChanged={invalidate}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function GroupCard({
  group,
  users,
  onChanged,
}: {
  group: GroupRow;
  users: UserRow[];
  onChanged: () => void;
}) {
  const qc = useQueryClient();
  const [membersOpen, setMembersOpen] = useState(false);

  const toggleAll = useMutation({
    mutationFn: (allAccess: boolean) => setGroupAllAccess(group.id, allAccess),
    onSuccess: onChanged,
  });
  const remove = useMutation({
    mutationFn: () => deleteGroup(group.id),
    onSuccess: onChanged,
  });

  const membersQk = ["group-members", group.id] as const;
  const members = useQuery({
    queryKey: membersQk,
    queryFn: () => fetchGroupMembers(group.id),
    retry: false,
    enabled: membersOpen,
  });

  const invalidateMembers = () =>
    void qc.invalidateQueries({ queryKey: membersQk });

  const memberIds = members.data ?? [];
  const usersById = useMemo(
    () => new Map(users.map((u) => [u.id, u])),
    [users],
  );
  const nonMembers = users.filter((u) => !memberIds.includes(u.id));

  const [pick, setPick] = useState("");

  const add = useMutation({
    mutationFn: (userId: string) => addGroupMember(group.id, userId),
    onSuccess: () => {
      setPick("");
      invalidateMembers();
    },
  });
  const removeMember = useMutation({
    mutationFn: (userId: string) => removeGroupMember(group.id, userId),
    onSuccess: invalidateMembers,
  });

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="font-medium">{group.name}</span>
          <Badge variant="muted">
            {relativeTime(Date.parse(group.created_at) / 1000)}
          </Badge>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              aria-label={`all_access for ${group.name}`}
              checked={group.all_access}
              disabled={toggleAll.isPending}
              onChange={(e) => toggleAll.mutate(e.target.checked)}
            />
            <span>all_access</span>
          </label>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setMembersOpen((v) => !v)}
          >
            {membersOpen ? "Hide members" : "Members"}
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={remove.isPending}
            onClick={() => remove.mutate()}
          >
            {remove.isPending ? "Deleting…" : "Delete"}
          </Button>
        </div>
      </div>

      {membersOpen && (
        <div className="mt-4 border-t border-border pt-4">
          {members.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading members…</p>
          ) : members.isError ? (
            <p role="alert" className="text-sm text-danger">
              {errText(members.error) ?? "Failed to load members."}
            </p>
          ) : (
            <>
              {memberIds.length === 0 ? (
                <p className="text-sm text-muted-foreground">No members yet.</p>
              ) : (
                <ul className="space-y-2">
                  {memberIds.map((id) => (
                    <li
                      key={id}
                      className="flex items-center justify-between gap-3"
                    >
                      <span className="text-sm">
                        {usersById.get(id)?.username ?? id}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={removeMember.isPending}
                        onClick={() => removeMember.mutate(id)}
                      >
                        Remove
                      </Button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="mt-3 flex flex-wrap items-end gap-3">
                <div className="flex-1 space-y-1.5">
                  <label
                    htmlFor={`add-member-${group.id}`}
                    className="text-sm font-medium"
                  >
                    Add member
                  </label>
                  <Select
                    id={`add-member-${group.id}`}
                    value={pick}
                    onChange={(e) => setPick(e.target.value)}
                  >
                    <option value="">Select a user…</option>
                    {nonMembers.map((u) => (
                      <option key={u.id} value={u.id}>
                        {u.username}
                      </option>
                    ))}
                  </Select>
                </div>
                <Button
                  type="button"
                  size="sm"
                  disabled={!pick || add.isPending}
                  onClick={() => pick && add.mutate(pick)}
                >
                  {add.isPending ? "Adding…" : "Add"}
                </Button>
              </div>
              {(errText(add.error) || errText(removeMember.error)) && (
                <p role="alert" className="mt-2 text-xs text-danger">
                  {errText(add.error) ?? errText(removeMember.error)}
                </p>
              )}
            </>
          )}
        </div>
      )}
      {errText(toggleAll.error) || errText(remove.error) ? (
        <p role="alert" className="mt-2 text-xs text-danger">
          {errText(toggleAll.error) ?? errText(remove.error)}
        </p>
      ) : null}
    </div>
  );
}
