import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { health, getApiKey, getApiBase, getEffectiveApiBase, registryAudit, setApiKey, setApiBase } from "../api/client";
import type { AuditEvent } from "../api/client";
import { Badge, Card, ErrorNote, Loading, Table, fmtNum } from "../components/ui";
import { AccountCard, ApiKeysCard, SharingCard, UsersCard } from "../components/account";
import { useAuth } from "../lib/auth-context";
import { useTask } from "../lib/task-context";

function ApiBaseCard() {
  const qc = useQueryClient();
  const [url, setUrl] = useState(getApiBase());
  const [key, setKey] = useState(getApiKey());
  const [saved, setSaved] = useState(false);

  // The server states its own posture at /api/auth/status. Probing a protected
  // endpoint instead cannot tell "open server" from "keyed server + accepted
  // key" — both answer 200 — which is why the note below reads the explicit
  // flag rather than guessing.
  const { status: authStatusInfo } = useAuth();

  const onSave = () => {
    setApiBase(url.trim());
    setApiKey(key);
    // The client reads the base URL and key per request, so new calls hit the
    // new endpoint immediately — drop cached responses to force a fresh fetch.
    void qc.invalidateQueries();
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  return (
    <Card title="API connection" sub="Base URL of the SKSurrogate control-plane API. Leave empty to use this page's own origin, which the Vite dev server proxies to the backend — that works from localhost, a LAN IP, or a forwarded URL. Stored in this browser only; VITE_API_BASE_URL is used when no override is saved.">
      <div className="row fixed">
        <div className="field" style={{ flex: "1 1 320px", minWidth: 240 }}>
          <label>Base URL</label>
          <input className="mono" value={url} onChange={(e) => setUrl(e.target.value)} spellCheck={false} placeholder="(empty = same origin)" />
        </div>
        <button className="btn primary" style={{ alignSelf: "flex-end", marginBottom: 2 }} onClick={onSave}>
          Save
        </button>
      </div>
      <p className="muted" style={{ marginTop: 2 }}>
        Requests currently go to <code className="mono">{getEffectiveApiBase()}</code>
        {getApiBase() ? " (absolute override)" : " (same origin, via the dev-server proxy)"}.
      </p>
      <div className="row fixed">
        <div className="field" style={{ flex: "1 1 320px", minWidth: 240 }}>
          <label>API key</label>
          <input
            className="mono"
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            spellCheck={false}
            placeholder="(none — only needed when the server sets SKSURROGATE_API_KEY)"
          />
        </div>
      </div>
      <p className="muted" style={{ marginTop: 6 }}>
        Sent as an <code>X-API-Key</code> header on every request (and a query parameter for job streams) when the server requires one. Leave empty for local, keyless use.
      </p>
      {authStatusInfo?.auth_enforced && (
        <p className="muted">
          This server requires an identity: the browser signs in with a username and password (see the Account card), and scripts use an API key. A key in the field above is optional and only needed by non-browser clients.
        </p>
      )}
      {authStatusInfo && !authStatusInfo.auth_enforced && (
        <p className="muted">
          Server is not enforcing authentication. Fine for localhost; if this port may be reachable beyond a trusted network, set{" "}
          <code>SKSURROGATE_REQUIRE_AUTH=true</code> and create an account on the server side.
        </p>
      )}
      {authStatusInfo?.auth_enforced && getApiKey() && (
        <p className="muted">
          A key is stored in this browser; it is sent alongside the session cookie. Remove it unless a non-browser tool needs it.
        </p>
      )}
      {saved && <p className="success-note">Saved — all views now talk to the new endpoint.</p>}
    </Card>
  );
}

function ServerCard() {
  const q = useQuery({ queryKey: ["health"], queryFn: health, retry: false });
  return (
    <Card title="Server" sub="The control-plane API this browser is connected to.">
      {q.isLoading && <Loading />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && (
        <p>
          <Badge tone="ok">{q.data.status}</Badge> SKSurrogate{" "}
          <span className="mono">{q.data.version ?? "unknown version"}</span>
        </p>
      )}
    </Card>
  );
}

function AuditLogCard({ task }: { task: string }) {
  const [filter, setFilter] = useState("all");

  const auditQ = useQuery({
    queryKey: ["registry-audit", task],
    queryFn: () => registryAudit(task),
    enabled: !!task,
  });

  const events = useMemo(() => (auditQ.data?.audit ?? []).slice().reverse(), [auditQ.data]);
  const types = useMemo(
    () => ["all", ...Array.from(new Set(events.map((a) => a.event_type)))],
    [events],
  );
  const rows = filter === "all" ? events : events.filter((a) => a.event_type === filter);

  return (
    <Card title={`Audit log (${fmtNum(rows.length)})`} sub="Every registry event for this task, newest first.">
      {!task && <p className="muted">Pick a task name in the top bar to see its audit trail.</p>}
      {auditQ.isLoading && <Loading />}
      {auditQ.isError && <ErrorNote error={auditQ.error} />}

      {task && (
        <>
          <div style={{ marginBottom: 10 }}>
            <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ maxWidth: 260 }}>
              {types.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>

          {rows.length === 0 && !auditQ.isLoading && <p className="muted">No audit events yet.</p>}
          {rows.length > 0 && (
            <Table<AuditEvent>
              columns={["Timestamp", "Event", "Action", "Model version", "Details"]}
              rows={rows}
              keyOf={(a, i) => `${a.timestamp}-${i}`}
              render={(a) => [
                <td key="t" className="mono">{new Date(a.timestamp).toLocaleString()}</td>,
                <td key="e">
                  <Badge tone={a.event_type === "promote" ? "info" : a.event_type === "rollback" ? "warn" : "muted"}>
                    {a.event_type}
                  </Badge>
                </td>,
                <td key="a">{a.action ?? "—"}</td>,
                <td key="v" className="mono">{a.model_version ?? "—"}</td>,
                <td key="d" className="mono" style={{ fontSize: 12 }}>{JSON.stringify(a.details ?? null)}</td>,
              ]}
            />
          )}
        </>
      )}
    </Card>
  );
}

export default function SettingsPage() {
  const { task } = useTask();

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Settings / Audit</h1>
        <span className="desc">
          Account, API keys and sharing for this deployment, the API connection for this browser,
          and the registry audit trail for task “{task || "…"}”.
        </span>
      </div>

      <AccountCard />
      <ApiKeysCard />
      <SharingCard />
      <UsersCard />
      <ApiBaseCard />
      <ServerCard />
      <AuditLogCard task={task} />
    </div>
  );
}
