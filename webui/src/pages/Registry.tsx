import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  REGISTRY_STATES,
  deleteArtifacts,
  getRegistryArtifacts,
  listBundles,
  promoteBundle,
  registerBundle,
  registryAudit,
  registryHistory,
  registrySummary,
  retryUnlessNotFound,
  rollbackBundle,
} from "../api/client";
import type { AuditEvent, RegistryArtifact, RegistryHistoryEntry } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, Loading, StatusBadge, Table, fmtBytes, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

/** How many registered versions currently sit in a given lifecycle state. */
function countInState(versions: Record<string, string>, state: string): number {
  return Object.values(versions).filter((s) => s === state).length;
}

export default function RegistryPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const bundlesQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });
  // One round-trip for the whole lifecycle: aliases + per-version states.
  // A 404 just means "no registry state yet" — not an error worth surfacing.
  const summaryQ = useQuery({
    queryKey: ["registry-summary", task],
    queryFn: () => registrySummary(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });
  const historyQ = useQuery({
    queryKey: ["registry-history", task],
    queryFn: () => registryHistory(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });
  const auditQ = useQuery({
    queryKey: ["registry-audit", task],
    queryFn: () => registryAudit(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });

  const aliases = summaryQ.data?.aliases ?? {};
  const versionStates = summaryQ.data?.versions ?? {};

  // Versions the user can act on: bundles folder + registry versions + history/audit.
  const versions = useMemo(() => {
    const set = new Set<string>(bundlesQ.data?.bundles ?? []);
    for (const v of Object.keys(versionStates)) set.add(v);
    for (const h of historyQ.data?.history ?? []) if (h.model_version) set.add(h.model_version);
    for (const a of auditQ.data?.audit ?? []) if (a.model_version) set.add(a.model_version);
    return [...set].reverse();
  }, [bundlesQ.data, versionStates, historyQ.data, auditQ.data]);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["registry-summary", task] });
    qc.invalidateQueries({ queryKey: ["registry-history", task] });
    qc.invalidateQueries({ queryKey: ["registry-audit", task] });
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Registry &amp; Deployment</h1>
        <span className="desc">Lifecycle, approvals and rollback for task “{task || "…"}” — approvers are free-text names in v1 (no auth)</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {task && (
        <Card title="Lifecycle" sub="Current holder of each alias. Promote moves a version forward; rollback re-points an alias at a prior version.">
          {summaryQ.isLoading && <Loading />}
          {!summaryQ.isLoading && summaryQ.isError && !summaryQ.data && (
            <p className="muted">No registry state yet for this task — register a bundle below to get started.</p>
          )}
          {summaryQ.data && (
            <>
              {aliases.latest && (
                <div style={{ marginBottom: 10 }}>
                  <span className="muted" style={{ marginRight: 8 }}>latest →</span>
                  <code>{aliases.latest}</code>{" "}
                  <StatusBadge status={versionStates[aliases.latest] ?? "candidate"} />
                </div>
              )}
              <div className="state-flow">
                {REGISTRY_STATES.map((s, i) => (
                  <span key={s} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    {i > 0 && <span className="state-arrow">→</span>}
                    <div style={{ textAlign: "center" }}>
                      <div className={`state-node${aliases[s] ? " current" : ""}`}>{s}</div>
                      <div className="muted mono" style={{ fontSize: 11, marginTop: 3 }}>
                        {aliases[s] ?? (countInState(versionStates, s) > 0 ? `${countInState(versionStates, s)} version(s)` : "—")}
                      </div>
                    </div>
                  </span>
                ))}
              </div>

              {[...new Set([aliases.latest, ...Object.values(aliases)].filter((v): v is string => !!v))].map((v) => (
                <LineageRail key={v} task={task} modelVersion={v} />
              ))}
            </>
          )}
        </Card>
      )}

      <RegisterCard task={task} versions={versions} onDone={() => { invalidate(); qc.invalidateQueries({ queryKey: ["bundles-list", task] }); }} />
      <PromoteCard task={task} versions={versions} onDone={invalidate} />
      <RollbackCard task={task} versions={versions} onDone={invalidate} />

      <Card title="History" sub="Promote / rollback events, newest last">
        {historyQ.isLoading && <Loading />}
        {historyQ.isError && <ErrorNote error={historyQ.error} />}
        {(historyQ.data?.history ?? []).length === 0 && !historyQ.isLoading && (
          <p className="muted">No lifecycle events yet — register a bundle above to get started.</p>
        )}
        {(historyQ.data?.history ?? []).length > 0 && (
          <Table<RegistryHistoryEntry>
            columns={["Timestamp", "Action", "From", "To", "Model version"]}
            rows={[...(historyQ.data?.history ?? [])].reverse()}
            keyOf={(h, i) => `${h.timestamp}-${i}`}
            render={(h) => [
              <td key="t" className="mono">{h.timestamp}</td>,
              <td key="a"><Badge tone={h.action === "promote" ? "info" : "warn"}>{h.action}</Badge></td>,
              <td key="f">{h.from ?? "—"}</td>,
              <td key="to">{h.to}</td>,
              <td key="v" className="mono">{h.model_version ?? h.state ?? "—"}</td>,
            ]}
          />
        )}
      </Card>

      <ArtifactCleanupCard task={task} versions={versions} onDone={invalidate} />

      <AuditCard task={task} audit={auditQ.data?.audit ?? []} loading={auditQ.isLoading} error={auditQ.error} />
    </div>
  );
}

/**
 * Retention view over the registry's stored dataset/prediction artifacts.
 *
 * The dataset fingerprint list is derived from the artifacts the registry
 * actually holds (not from the mltrace metadata), so the selection always
 * matches what a cleanup call can delete. At least one selector is required —
 * the server rejects an unscoped delete with 422 — so the button stays
 * disabled until a scope is chosen.
 */
function ArtifactCleanupCard({ task, versions, onDone }: { task: string; versions: string[]; onDone: () => void }) {
  const qc = useQueryClient();
  const [version, setVersion] = useState("");
  const [fingerprint, setFingerprint] = useState("");

  const artifactsQ = useQuery({
    queryKey: ["registry-artifacts", task],
    queryFn: () => getRegistryArtifacts(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });

  const datasets = artifactsQ.data?.datasets ?? [];
  const predictions = artifactsQ.data?.predictions ?? [];

  const fingerprints = useMemo(
    () => [...new Set(datasets.map((d) => d.dataset_fingerprint).filter((f): f is string => !!f))],
    [datasets],
  );

  // What the pending selection would remove, so the confirmation is concrete.
  const doomed = useMemo(() => {
    const matchingDatasets = fingerprint ? datasets.filter((d) => d.dataset_fingerprint === fingerprint) : [];
    const matchingPredictions = predictions.filter(
      (p) =>
        (!version || p.model_version === version) &&
        (!fingerprint || p.dataset_fingerprint === fingerprint),
    );
    const bytes = [...matchingDatasets, ...matchingPredictions].reduce((sum, a) => sum + a.size_bytes, 0);
    return { datasets: matchingDatasets, predictions: matchingPredictions, bytes };
  }, [datasets, predictions, version, fingerprint]);

  const mut = useMutation({
    mutationFn: () => deleteArtifacts(task, { model_version: version || null, dataset_fingerprint: fingerprint || null }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-artifacts", task] });
      onDone();
    },
  });

  const scope = [version && `model ${version}`, fingerprint && `dataset ${fingerprint}`].filter(Boolean).join(" + ");

  return (
    <Card
      title="Artifact cleanup"
      sub="Drops the registry's record of registered dataset versions and deletes stored prediction CSVs. The source CSVs under datasets/ and all bundles are untouched."
      actions={
        <button className="btn" onClick={() => artifactsQ.refetch()} disabled={artifactsQ.isFetching}>
          {artifactsQ.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      }
    >
      {!task && <p className="muted">Pick a task to inspect its artifacts.</p>}
      {artifactsQ.isLoading && task && <Loading />}
      {artifactsQ.isError && <ErrorNote error={artifactsQ.error} />}

      {task && artifactsQ.data && (
        <>
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
              <label>Model version (predictions)</label>
              <select value={version} onChange={(e) => setVersion(e.target.value)}>
                <option value="">— any model —</option>
                {[...new Set([...versions, ...predictions.map((p) => p.model_version).filter((v): v is string => !!v)])].map(
                  (v) => (
                    <option key={v}>{v}</option>
                  ),
                )}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 420px", minWidth: 280 }}>
              <label>Dataset fingerprint</label>
              <select value={fingerprint} onChange={(e) => setFingerprint(e.target.value)}>
                <option value="">— any dataset —</option>
                {fingerprints.map((f) => (
                  <option key={f}>{f}</option>
                ))}
              </select>
            </div>
          </div>

          <p className="muted" style={{ marginTop: 4 }}>
            {fmtNum(datasets.length)} dataset(s) · {fmtNum(predictions.length)} prediction(s) ·{" "}
            {fmtBytes(artifactsQ.data.total_bytes)} stored
          </p>

          {scope ? (
            <div className="warn-note">
              Removing <strong>{fmtNum(doomed.datasets.length)}</strong> dataset copy/copies and{" "}
              <strong>{fmtNum(doomed.predictions.length)}</strong> prediction file(s) matching {scope} (
              {fmtBytes(doomed.bytes)}).
            </div>
          ) : (
            <p className="muted">Choose a model version and/or dataset fingerprint to scope the deletion.</p>
          )}

          {mut.isError && <ErrorNote error={mut.error} />}
          {mut.isSuccess && (
            <div className="success-note">
              Deleted {fmtNum(mut.data.deleted.datasets)} dataset(s) and {fmtNum(mut.data.deleted.predictions)}{" "}
              prediction(s).
            </div>
          )}

          <button
            className="btn danger"
            disabled={!scope || mut.isPending}
            onClick={() => {
              if (window.confirm(`Permanently delete artifacts matching ${scope} for task “${task}”?`)) mut.mutate();
            }}
          >
            {mut.isPending ? "Deleting…" : "Delete artifacts"}
          </button>

          {(datasets.length > 0 || predictions.length > 0) && (
            <>
              <h3 style={{ marginTop: 16 }}>Registered artifacts</h3>
              <Table<RegistryArtifact>
                columns={["Kind", "Model version", "Dataset fingerprint", "Path", "Size", "Registered"]}
                rows={[...datasets, ...predictions]}
                keyOf={(a, i) => `${a.kind}-${a.path}-${i}`}
                render={(a) => [
                  <td key="k">
                    <Badge tone={a.kind === "dataset" ? "info" : "muted"}>{a.kind}</Badge>
                  </td>,
                  <td key="m" className="mono">{a.model_version ?? "—"}</td>,
                  <td key="f" className="mono" style={{ fontSize: 11 }}>{a.dataset_fingerprint ?? "—"}</td>,
                  <td key="p" className="mono" style={{ fontSize: 11 }}>{a.path}</td>,
                  <td key="s">{fmtBytes(a.size_bytes)}</td>,
                  <td key="t" className="mono">{a.registered_at ?? "—"}</td>,
                ]}
              />
            </>
          )}
        </>
      )}
    </Card>
  );
}

function RegisterCard({ task, versions, onDone }: { task: string; versions: string[]; onDone: () => void }) {
  const [version, setVersion] = useState("");
  const mut = useMutation({ mutationFn: () => registerBundle(task, version), onSuccess: onDone });

  return (
    <Card title="Register a bundle" sub="Copies a bundle from the task's bundles folder into the registry as a candidate.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Model version (from Model Bundles)</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>
      {mut.isError && <ErrorNote error={mut.error} />}
      {mut.isSuccess && (
        <div className="success-note">
          Registered <code>{(mut.data as { model_version: string }).model_version}</code> as candidate.
        </div>
      )}
      <button className="btn primary" disabled={!task || !version || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Registering…" : "Register"}
      </button>
    </Card>
  );
}

function PromoteCard({ task, versions, onDone }: { task: string; versions: string[]; onDone: () => void }) {
  const [version, setVersion] = useState("");
  const [state, setState] = useState<string>("validated");
  const [approversText, setApproversText] = useState("mehdi");
  const [required, setRequired] = useState(1);

  const mut = useMutation({
    mutationFn: () => {
      const approvers = approversText.split(",").map((s) => s.trim()).filter(Boolean);
      if (!approvers.length) throw new Error("Provide at least one approver name (comma-separated).");
      return promoteBundle(task, { model_version: version, state, approvers, required_approvals: required });
    },
    onSuccess: onDone,
  });

  return (
    <Card title="Promote" sub="Requires N unique named approvers; the target alias is updated atomically.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Model version</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 130 }}>
          <label>Target state</label>
          <select value={state} onChange={(e) => setState(e.target.value)}>
            {REGISTRY_STATES.filter((s) => s !== "candidate").map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
          <label>Approvers (comma-separated)</label>
          <input value={approversText} onChange={(e) => setApproversText(e.target.value)} spellCheck={false} />
        </div>
        <div className="field fixed" style={{ flex: "0 1 120px", minWidth: 90 }}>
          <label>Required approvals</label>
          <input type="number" min={1} value={required} onChange={(e) => setRequired(Number(e.target.value))} />
        </div>
      </div>
      {mut.isError && <ErrorNote error={mut.error} />}
      {mut.isSuccess && (
        <div className="success-note">
          Promoted to <StatusBadge status={(mut.data as Record<string, unknown>).state as string} /> —{" "}
          {JSON.stringify(mut.data)}
        </div>
      )}
      <button className="btn primary" disabled={!task || !version || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Promoting…" : "Promote"}
      </button>
    </Card>
  );
}

function RollbackCard({
  task,
  versions,
  onDone,
}: {
  task: string;
  versions: string[];
  onDone: () => void;
}) {
  const [state, setState] = useState("production");
  const [version, setVersion] = useState("");

  const mut = useMutation({ mutationFn: () => rollbackBundle(task, state, version || undefined), onSuccess: onDone });

  return (
    <Card title="Rollback" sub="Re-points a lifecycle alias at a prior model version.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 130 }}>
          <label>State (alias)</label>
          <select value={state} onChange={(e) => setState(e.target.value)}>
            {REGISTRY_STATES.filter((s) => s !== "archived").map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Target version (empty = previous holder of this alias)</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— previous —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>
      {mut.isError && <ErrorNote error={mut.error} />}
      {mut.isSuccess && (
        <div className="success-note">
          Rolled back: alias now at <code>{(mut.data as { model_version: string | null }).model_version ?? "—"}</code>.
        </div>
      )}
      <button
        className="btn danger"
        disabled={!task || mut.isPending}
        onClick={() => {
          if (window.confirm(`Rollback alias “${state}” for task “${task}”?`)) mut.mutate();
        }}
      >
        {mut.isPending ? "Rolling back…" : "Rollback"}
      </button>
    </Card>
  );
}

function AuditCard({
  task,
  audit,
  loading,
  error,
}: {
  task: string;
  audit: AuditEvent[];
  loading: boolean;
  error: unknown;
}) {
  const [filter, setFilter] = useState("all");

  const types = useMemo(() => ["all", ...new Set(audit.map((a) => a.event_type))], [audit]);
  const rows = filter === "all" ? audit : audit.filter((a) => a.event_type === filter);

  return (
    <Card title={`Audit log (${fmtNum(rows.length)})`} sub="Every registry event for this task, newest last.">
      {loading && <Loading />}
      {error ? <ErrorNote error={error} /> : null}
      {!task && <p className="muted">Pick a task to see its audit trail.</p>}
      {task && (
        <>
          <div style={{ marginBottom: 10 }}>
            <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ maxWidth: 240 }}>
              {types.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>
          {(rows ?? []).length === 0 && !loading && <p className="muted">No audit events yet.</p>}
          {(rows ?? []).length > 0 && (
            <Table<AuditEvent>
              columns={["Timestamp", "Event", "Model version", "Details"]}
              rows={[...rows].reverse()}
              keyOf={(a, i) => `${a.timestamp}-${i}`}
              render={(a) => [
                <td key="t" className="mono">{a.timestamp}</td>,
                <td key="e"><Badge tone={a.event_type === "promote" ? "info" : a.event_type === "rollback" ? "warn" : "muted"}>{a.event_type}</Badge></td>,
                <td key="v" className="mono">{a.model_version ?? "—"}</td>,
                <td key="d" className="mono" style={{ fontSize: 12 }}>{JSON.stringify(a.details ?? a)}</td>,
              ]}
            />
          )}
        </>
      )}
    </Card>
  );
}
