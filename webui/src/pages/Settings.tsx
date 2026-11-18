import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  health,
  getApiKey,
  getApiBase,
  getComputeSettings,
  getEffectiveApiBase,
  registryAudit,
  resetMyData,
  setApiKey,
  setApiBase,
  updateComputeSettings,
} from "../api/client";
import type { AuditEvent, ComputeSettingsValues, ResetResult } from "../api/client";
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

function ComputeCard() {
  const qc = useQueryClient();
  const { me, status } = useAuth();
  const settingsQ = useQuery({ queryKey: ["compute-settings"], queryFn: getComputeSettings });
  const [draft, setDraft] = useState<ComputeSettingsValues | null>(null);
  const [saved, setSaved] = useState(false);

  // Seed the form once the server answers; afterwards the draft is the user's.
  useEffect(() => {
    if (settingsQ.data && draft === null) setDraft(settingsQ.data.values);
  }, [settingsQ.data, draft]);

  const saveMut = useMutation({
    mutationFn: (values: ComputeSettingsValues) => updateComputeSettings(values),
    onSuccess: (snapshot) => {
      qc.setQueryData(["compute-settings"], snapshot);
      setDraft(snapshot.values);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2000);
    },
  });

  const cores = settingsQ.data?.cores ?? 0;
  const defaults = settingsQ.data?.defaults;
  const values = draft;
  const used = values ? values.job_workers * values.blas_threads + values.job_threads : 0;
  const overBudget = values !== null && used > cores;
  const daskUsed = values ? values.dask_workers * values.dask_threads : 0;
  const daskOver = values !== null && daskUsed > cores;
  const canEdit = !status?.auth_enforced || Boolean(me?.is_admin);

  const update = (key: keyof ComputeSettingsValues, raw: string) => {
    if (!draft) return;
    const parsed = Number(raw);
    setDraft({ ...draft, [key]: Number.isFinite(parsed) ? Math.max(1, Math.trunc(parsed)) : draft[key] });
  };

  return (
    <Card
      title="Compute budget"
      sub="Server-side limit for background jobs. Every value is capped by the cores this API process may use, and a running job keeps the limits it started with."
    >
      {settingsQ.isLoading && <Loading />}
      {settingsQ.isError && <ErrorNote error={settingsQ.error} />}
      {values && (
        <>
          <p className="muted" style={{ marginTop: 0 }}>
            This server has <b>{cores}</b> core{cores === 1 ? "" : "s"} available. Budget used:{" "}
            <b>{used}</b> / {cores}
            {overBudget ? " — over budget" : ""}.
          </p>

          <div className="row">
            <div className="field">
              <label>Concurrent jobs</label>
              <input
                type="number"
                min={1}
                max={cores}
                value={values.job_workers}
                disabled={!canEdit}
                onChange={(e) => update("job_workers", e.target.value)}
              />
              <span className="hint">Jobs evaluated at the same time; extra submissions queue.</span>
            </div>
            <div className="field">
              <label>Fold workers (total)</label>
              <input
                type="number"
                min={1}
                max={cores}
                value={values.job_threads}
                disabled={!canEdit}
                onChange={(e) => update("job_threads", e.target.value)}
              />
              <span className="hint">Shared joblib pool: parallel CV-fold fits across all jobs.</span>
            </div>
            <div className="field">
              <label>BLAS threads per job</label>
              <input
                type="number"
                min={1}
                max={cores}
                value={values.blas_threads}
                disabled={!canEdit}
                onChange={(e) => update("blas_threads", e.target.value)}
              />
              <span className="hint">OpenMP/MKL threads each job slot may use.</span>
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Dask workers</label>
              <input
                type="number"
                min={1}
                max={cores}
                value={values.dask_workers}
                disabled={!canEdit}
                onChange={(e) => update("dask_workers", e.target.value)}
              />
              <span className="hint">Processes in the local cluster used by the Dask backend.</span>
            </div>
            <div className="field">
              <label>Dask threads per worker</label>
              <input
                type="number"
                min={1}
                max={cores}
                value={values.dask_threads}
                disabled={!canEdit}
                onChange={(e) => update("dask_threads", e.target.value)}
              />
              <span className="hint">Threads inside each Dask worker.</span>
            </div>
          </div>

          <p className={overBudget || daskOver ? "warn-note" : "muted"}>
            Job budget: {values.job_workers} × {values.blas_threads} BLAS + {values.job_threads} fold
            workers = {used} of {cores} cores. Dask: {values.dask_workers} × {values.dask_threads} ={" "}
            {daskUsed} of {cores}.
            {overBudget || daskOver
              ? " Reduce the marked values until both totals fit."
              : ""}
          </p>

          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          {!canEdit && (
            <p className="muted">Only an administrator can change the compute budget.</p>
          )}

          <div className="row fixed">
            <button
              className="btn primary"
              disabled={!canEdit || overBudget || daskOver || saveMut.isPending}
              onClick={() => values && saveMut.mutate(values)}
            >
              {saveMut.isPending ? "Saving…" : "Save compute budget"}
            </button>
            {defaults && (
              <button className="btn" disabled={!canEdit || saveMut.isPending} onClick={() => setDraft(defaults)}>
                Use defaults
              </button>
            )}
            {saved && <span className="muted">Saved — new jobs pick this up immediately.</span>}
          </div>
        </>
      )}
    </Card>
  );
}

function DangerZoneCard() {
  const qc = useQueryClient();
  const { status, me } = useAuth();
  const { setTask } = useTask();
  const [confirm, setConfirm] = useState("");
  const [includeShared, setIncludeShared] = useState(false);
  const [result, setResult] = useState<ResetResult | null>(null);

  const openMode = !(status?.auth_enforced ?? false);
  const expected = openMode ? "reset everything" : `reset ${me?.username ?? "<username>"}`;
  const staticKey = !openMode && me?.user_id === null;

  const mut = useMutation({
    mutationFn: () => resetMyData(confirm.trim(), includeShared),
    onSuccess: (payload) => {
      setResult(payload);
      setConfirm("");
      setTask("");
      qc.clear();
      void qc.invalidateQueries();
    },
  });

  return (
    <Card
      title="Danger zone — reset my data"
      sub="Permanently deletes the datasets, tasks, model bundles, registry versions, predictions, monitoring records, checkpoints, jobs, logs and research projects this account owns."
    >
      <div className="warn-note">
        This cannot be undone. Accounts, sessions and API keys are kept.{" "}
        {openMode
          ? "This server has no authentication, so the single anonymous user owns everything: the reset wipes the entire local workspace."
          : "Resources shared with other users, and material with no owner, are left untouched unless you explicitly include them."}
      </div>

      <div className="row fixed">
        <div className="field fixed" style={{ flex: "0 1 340px", minWidth: 260 }}>
          <label>
            Type <code>{expected}</code> to confirm
          </label>
          <input
            className="mono"
            value={confirm}
            onChange={(e) => {
              setConfirm(e.target.value);
              setResult(null);
            }}
            placeholder={expected}
            spellCheck={false}
            autoComplete="off"
          />
        </div>
        <label className="checkbox-row" style={{ alignSelf: "flex-end", marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={includeShared}
            onChange={(e) => setIncludeShared(e.target.checked)}
          />
          Also delete resources I own that are shared with other users
        </label>
      </div>

      {staticKey && (
        <p className="muted">
          You are authenticated with the shared API key, which has no account identity. Sign in
          with a user account to reset personal data.
        </p>
      )}
      {mut.isError && <ErrorNote error={mut.error} />}

      {result && (
        <div className="success-note">
          Removed {fmtNum(result.tasks.length)} task(s), {fmtNum(result.projects.length)}{" "}
          project(s), {fmtNum(result.jobs)} job(s), {fmtNum(result.log_files)} log file(s) and{" "}
          {fmtNum(result.files)} stored file(s) ({fmtNum(result.bytes)} bytes).
          {result.skipped_shared.tasks.length + result.skipped_shared.projects.length > 0 && (
            <>
              {" "}
              Kept {fmtNum(result.skipped_shared.tasks.length)} shared task(s) and{" "}
              {fmtNum(result.skipped_shared.projects.length)} shared project(s).
            </>
          )}
          <div style={{ marginTop: 8 }}>
            <button className="btn" onClick={() => window.location.assign("/")}>
              Reload application
            </button>
          </div>
        </div>
      )}

      <button
        className="btn danger"
        disabled={mut.isPending || staticKey || confirm.trim() !== expected}
        onClick={() => mut.mutate()}
      >
        {mut.isPending ? "Resetting…" : "Reset all my data"}
      </button>
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
      <ComputeCard />
      <DangerZoneCard />
      <ServerCard />
      <AuditLogCard task={task} />
    </div>
  );
}
