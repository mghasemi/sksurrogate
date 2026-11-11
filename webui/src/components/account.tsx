/**
 * Account, API-key, user and sharing management for the control plane.
 *
 * Rendered from the Settings page. Everything here is inert when the API does
 * not enforce authentication (`/api/auth/status` says so): the cards explain
 * that instead of pretending to manage accounts that do not exist.
 *
 * Secrets are shown once. `createApiKey` returns the plaintext key in its
 * response and the list endpoint only ever returns the prefix, so the modal-ish
 * box below is the only place the full key exists in the browser.
 */

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  adoptOrphans,
  changePassword,
  createApiKey,
  createUser,
  listApiKeys,
  listUsers,
  revokeApiKey,
  shareTask,
  taskShares,
  unshareTask,
  updateUser,
} from "../api/client";
import type { AclRole, ApiKeyRecord, AuthUser, ShareEntry } from "../api/client";
import { Badge, Card, ErrorNote, Loading, Table } from "../components/ui";
import { useAuth } from "../lib/auth-context";
import { useTask } from "../lib/task-context";

const ROLES: AclRole[] = ["viewer", "editor", "owner"];

function NotEnforcedNote() {
  return (
    <p className="muted">
      This server is not enforcing authentication, so accounts and sharing are inactive. Set{" "}
      <code>SKSURROGATE_REQUIRE_AUTH=true</code> on the API and create an account (or set{" "}
      <code>SKSURROGATE_ADMIN_USERNAME</code>/<code>SKSURROGATE_ADMIN_PASSWORD</code>) to enable
      them.
    </p>
  );
}

export function AccountCard() {
  const { status, me, refresh } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  if (status && !status.auth_enforced) {
    return (
      <Card title="Account" sub="Password and session management.">
        <NotEnforcedNote />
      </Card>
    );
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await changePassword(current, next);
      setCurrent("");
      setNext("");
      setMessage("Password changed. Your other sessions were signed out.");
      await refresh();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Account"
      sub="Your sign-in identity. Passwords are hashed with scrypt; sessions live in an HttpOnly cookie."
    >
      {me?.user ? (
        <>
          <p>
            Signed in as <strong>{me.user.username}</strong>{" "}
            {me.user.is_admin ? <Badge tone="info">administrator</Badge> : <Badge tone="muted">user</Badge>}
          </p>
          <form onSubmit={submit} className="row fixed" style={{ alignItems: "flex-end" }}>
            <div className="field" style={{ flex: "1 1 220px" }}>
              <label>Current password</label>
              <input
                type="password"
                value={current}
                onChange={(e) => setCurrent(e.target.value)}
                autoComplete="current-password"
                required
              />
            </div>
            <div className="field" style={{ flex: "1 1 220px" }}>
              <label>New password (min 8 characters)</label>
              <input
                type="password"
                value={next}
                onChange={(e) => setNext(e.target.value)}
                autoComplete="new-password"
                minLength={8}
                required
              />
            </div>
            <button className="btn primary" type="submit" disabled={busy || !current || next.length < 8}>
              {busy ? "Saving…" : "Change password"}
            </button>
          </form>
          {message && <p className="success-note">{message}</p>}
          {error !== null && <ErrorNote error={error} />}
        </>
      ) : (
        <p className="muted">
          This request authenticated with the shared API key, which has no account and therefore no
          password to change.
        </p>
      )}
    </Card>
  );
}

export function ApiKeysCard() {
  const { status, me } = useAuth();
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [secret, setSecret] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const keysQ = useQuery({
    queryKey: ["api-keys"],
    queryFn: listApiKeys,
    enabled: !!status?.auth_enforced && !!me?.user,
  });

  if (status && !status.auth_enforced) {
    return (
      <Card title="API keys" sub="Programmatic access for scripts and CI.">
        <NotEnforcedNote />
      </Card>
    );
  }
  if (!me?.user) {
    return (
      <Card title="API keys" sub="Programmatic access for scripts and CI.">
        <p className="muted">API keys belong to an account; the shared key cannot own keys.</p>
      </Card>
    );
  }

  const create = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const created = await createApiKey(name.trim() || "default");
      setSecret(created.key);
      setName("");
      await qc.invalidateQueries({ queryKey: ["api-keys"] });
    } catch (err) {
      setError(err);
    }
  };

  const revoke = async (record: ApiKeyRecord) => {
    setError(null);
    try {
      await revokeApiKey(record.id);
      await qc.invalidateQueries({ queryKey: ["api-keys"] });
    } catch (err) {
      setError(err);
    }
  };

  const keys = keysQ.data ?? [];

  return (
    <Card
      title="API keys"
      sub="Send these as X-API-Key (or Authorization: Bearer) from scripts. Each key acts as you, so it sees exactly your tasks."
    >
      {secret && (
        <div className="warn-note">
          <p style={{ marginTop: 0 }}>
            <strong>Copy this key now — it is shown once and cannot be recovered.</strong>
          </p>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{secret}</pre>
          <button className="btn" type="button" onClick={() => setSecret(null)}>
            I have stored it
          </button>
        </div>
      )}

      <form onSubmit={create} className="row fixed" style={{ alignItems: "flex-end" }}>
        <div className="field" style={{ flex: "1 1 260px" }}>
          <label>Key name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. ci-pipeline" />
        </div>
        <button className="btn primary" type="submit">
          Create key
        </button>
      </form>

      {error !== null && <ErrorNote error={error} />}
      {keysQ.isLoading && <Loading />}
      {keysQ.isError && <ErrorNote error={keysQ.error} />}
      {keys.length === 0 && !keysQ.isLoading && <p className="muted">No keys yet.</p>}
      {keys.length > 0 && (
        <Table<ApiKeyRecord>
          columns={["Name", "Prefix", "Created", "Last used", "Status", ""]}
          rows={keys}
          keyOf={(k) => k.id}
          render={(k) => [
            <td key="n">{k.name}</td>,
            <td key="p" className="mono">{k.prefix}…</td>,
            <td key="c" className="mono">{new Date(k.created_at).toLocaleString()}</td>,
            <td key="u" className="mono">{k.last_used_at ? new Date(k.last_used_at).toLocaleString() : "never"}</td>,
            <td key="s">
              {k.revoked_at ? <Badge tone="err">revoked</Badge> : <Badge tone="ok">active</Badge>}
            </td>,
            <td key="a">
              {!k.revoked_at && (
                <button className="btn" type="button" onClick={() => void revoke(k)}>
                  Revoke
                </button>
              )}
            </td>,
          ]}
        />
      )}
    </Card>
  );
}

export function UsersCard() {
  const { status, isAdmin } = useAuth();
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [asAdmin, setAsAdmin] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const usersQ = useQuery({
    queryKey: ["auth-users"],
    queryFn: listUsers,
    enabled: !!status?.auth_enforced && isAdmin,
  });

  if (status && !status.auth_enforced) {
    return (
      <Card title="Users" sub="Accounts on this control plane.">
        <NotEnforcedNote />
      </Card>
    );
  }
  if (!isAdmin) {
    return (
      <Card title="Users" sub="Accounts on this control plane.">
        <p className="muted">Only administrators can manage accounts.</p>
      </Card>
    );
  }

  const create = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await createUser(username.trim(), password, asAdmin);
      setUsername("");
      setPassword("");
      setAsAdmin(false);
      await qc.invalidateQueries({ queryKey: ["auth-users"] });
    } catch (err) {
      setError(err);
    }
  };

  const patch = async (userId: number, body: { is_active?: boolean; is_admin?: boolean }) => {
    setError(null);
    try {
      await updateUser(userId, body);
      await qc.invalidateQueries({ queryKey: ["auth-users"] });
    } catch (err) {
      setError(err);
    }
  };

  const resetPassword = async (userId: number, value: string) => {
    if (value.length < 8) return;
    setError(null);
    try {
      await updateUser(userId, { password: value });
      await qc.invalidateQueries({ queryKey: ["auth-users"] });
    } catch (err) {
      setError(err);
    }
  };

  const users = usersQ.data ?? [];

  return (
    <Card
      title="Users"
      sub="Create accounts, disable them (which revokes their sessions) and reset passwords. The last active administrator cannot be disabled or demoted."
    >
      <form onSubmit={create} className="row fixed" style={{ alignItems: "flex-end" }}>
        <div className="field" style={{ flex: "1 1 200px" }}>
          <label>Username</label>
          <input value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3} />
        </div>
        <div className="field" style={{ flex: "1 1 200px" }}>
          <label>Password (min 8 characters)</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            autoComplete="new-password"
          />
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, paddingBottom: 8 }}>
          <input type="checkbox" checked={asAdmin} onChange={(e) => setAsAdmin(e.target.checked)} />
          administrator
        </label>
        <button className="btn primary" type="submit" disabled={!username || password.length < 8}>
          Add user
        </button>
      </form>

      {error !== null && <ErrorNote error={error} />}
      {usersQ.isLoading && <Loading />}
      {usersQ.isError && <ErrorNote error={usersQ.error} />}
      {users.length > 0 && (
        <Table<AuthUser>
          columns={["Username", "Admin", "Active", "Created", "Actions", "Reset password"]}
          rows={users}
          keyOf={(u) => u.user_id}
          render={(u) => [
            <td key="u" className="mono">{u.username}</td>,
            <td key="a">{u.is_admin ? <Badge tone="info">admin</Badge> : <Badge tone="muted">user</Badge>}</td>,
            <td key="s">{u.is_active ? <Badge tone="ok">active</Badge> : <Badge tone="err">disabled</Badge>}</td>,
            <td key="c" className="mono">{new Date(u.created_at).toLocaleDateString()}</td>,
            <td key="x">
              <span style={{ display: "inline-flex", gap: 6, flexWrap: "wrap" }}>
                <button className="btn" type="button" onClick={() => void patch(u.user_id, { is_active: !u.is_active })}>
                  {u.is_active ? "Disable" : "Enable"}
                </button>
                <button className="btn" type="button" onClick={() => void patch(u.user_id, { is_admin: !u.is_admin })}>
                  {u.is_admin ? "Demote" : "Promote"}
                </button>
              </span>
            </td>,
            <td key="p">
              <ResetPasswordCell onSubmit={(value) => resetPassword(u.user_id, value)} />
            </td>,
          ]}
        />
      )}
    </Card>
  );
}

function ResetPasswordCell({ onSubmit }: { onSubmit: (value: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <span style={{ display: "inline-flex", gap: 6 }}>
      <input
        type="password"
        value={value}
        placeholder="new password"
        style={{ minWidth: 140 }}
        onChange={(e) => setValue(e.target.value)}
      />
      <button
        className="btn"
        type="button"
        disabled={value.length < 8}
        onClick={() => {
          onSubmit(value);
          setValue("");
        }}
      >
        Set
      </button>
    </span>
  );
}

export function SharingCard() {
  const { status, isAdmin, me } = useAuth();
  const { task } = useTask();
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState<AclRole>("viewer");
  const [error, setError] = useState<unknown>(null);
  const [message, setMessage] = useState<string | null>(null);

  const sharesQ = useQuery({
    queryKey: ["task-shares", task],
    queryFn: () => taskShares(task),
    enabled: !!status?.auth_enforced && !!task,
    retry: false,
  });

  if (status && !status.auth_enforced) {
    return (
      <Card title="Sharing" sub="Give other accounts access to the current task.">
        <NotEnforcedNote />
      </Card>
    );
  }
  if (!task) {
    return (
      <Card title="Sharing" sub="Give other accounts access to the current task.">
        <p className="muted">Pick a task in the top bar to manage who can see it.</p>
      </Card>
    );
  }

  const add = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);
    try {
      await shareTask(task, username.trim(), role);
      setUsername("");
      setMessage(`Shared “${task}” with ${username.trim()} as ${role}.`);
      await qc.invalidateQueries({ queryKey: ["task-shares", task] });
      await qc.invalidateQueries({ queryKey: ["tasks"] });
    } catch (err) {
      setError(err);
    }
  };

  const remove = async (entry: ShareEntry) => {
    setError(null);
    setMessage(null);
    try {
      await unshareTask(task, entry.user_id);
      await qc.invalidateQueries({ queryKey: ["task-shares", task] });
    } catch (err) {
      setError(err);
    }
  };

  const adopt = async () => {
    setError(null);
    setMessage(null);
    try {
      const result = await adoptOrphans("task");
      setMessage(
        result.count
          ? `Adopted ${result.count} unowned task(s) as ${result.username}.`
          : "Nothing to adopt — every task already has an owner.",
      );
      await qc.invalidateQueries({ queryKey: ["task-shares", task] });
    } catch (err) {
      setError(err);
    }
  };

  const payload = sharesQ.data;
  const myRole = payload?.role;
  const canShare = myRole === "owner";

  return (
    <Card
      title={`Sharing — ${task}`}
      sub="Ownership is per task: sharing a task shares its datasets, bundles, registry aliases, jobs and logs. Roles: viewer (read), editor (train/register/promote), owner (also sharing)."
    >
      {sharesQ.isLoading && <Loading />}
      {sharesQ.isError && <ErrorNote error={sharesQ.error} />}

      {payload && (
        <>
          <p>
            Your role: <Badge tone={myRole === "owner" ? "info" : "muted"}>{myRole ?? "none"}</Badge>
            {me?.is_admin && <span className="muted"> (administrator — owner on every task)</span>}
          </p>
          {payload.shares.length === 0 ? (
            <p className="muted">
              No explicit shares yet. An unowned task is visible to every signed-in user until an
              administrator adopts it.
            </p>
          ) : (
            <Table<ShareEntry>
              columns={["User", "Role", "Shared by", "Since", ""]}
              rows={payload.shares}
              keyOf={(share) => share.user_id}
              render={(share) => [
                <td key="u" className="mono">
                  {share.username ?? `user ${share.user_id}`}{" "}
                  {!share.is_active && <Badge tone="err">disabled</Badge>}
                </td>,
                <td key="r"><Badge tone={share.role === "owner" ? "info" : "muted"}>{share.role}</Badge></td>,
                <td key="b" className="mono">{share.granted_by ?? "—"}</td>,
                <td key="s" className="mono">{new Date(share.created_at).toLocaleDateString()}</td>,
                <td key="a">
                  {canShare && (
                    <button className="btn" type="button" onClick={() => void remove(share)}>
                      Remove
                    </button>
                  )}
                </td>,
              ]}
            />
          )}

          {canShare ? (
            <form onSubmit={add} className="row fixed" style={{ alignItems: "flex-end", marginTop: 10 }}>
              <div className="field" style={{ flex: "1 1 220px" }}>
                <label>Username</label>
                <input value={username} onChange={(e) => setUsername(e.target.value)} required />
              </div>
              <div className="field" style={{ flex: "0 0 150px" }}>
                <label>Role</label>
                <select value={role} onChange={(e) => setRole(e.target.value as AclRole)}>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>
              <button className="btn primary" type="submit" disabled={!username.trim()}>
                Share
              </button>
            </form>
          ) : (
            <p className="muted">Only an owner can change the share list.</p>
          )}
        </>
      )}

      {isAdmin && (
        <p style={{ marginTop: 12 }}>
          <button className="btn" type="button" onClick={() => void adopt()}>
            Adopt unowned tasks as me
          </button>{" "}
          <span className="muted">
            Gives you ownership of material that predates accounts (idempotent).
          </span>
        </p>
      )}

      {message && <p className="success-note">{message}</p>}
      {error !== null && <ErrorNote error={error} />}
    </Card>
  );
}
