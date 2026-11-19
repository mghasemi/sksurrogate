import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import {
  REGISTRY_STATES,
  deleteArtifacts,
  deleteRegistryAlias,
  getRegistryArtifacts,
  listBundles,
  loadRegisteredBundle,
  promoteBundle,
  registerBundle,
  registryAudit,
  registryHistory,
  registrySummary,
  retryUnlessNotFound,
  rollbackBundle,
  setRegistryAlias,
} from "../api/client";
import type { AuditEvent, RegistryAliasEntry, RegistryArtifact, RegistryHistoryEntry } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, Loading, StatusBadge, Table, fmtBytes, fmtNum } from "../components/ui";
import { useBundleLabel } from "../lib/hooks";
import { useTask } from "../lib/task-context";

/** How many registered versions currently sit in a given lifecycle state. */
function countInState(versions: Record<string, string>, state: string): number {
  return Object.values(versions).filter((s) => s === state).length;
}

/** Grammar for user-chosen alias names — must stay in sync with ``ModelRegistry.validate_alias_name``. */
const ALIAS_NAME_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/;
/** Lifecycle pointer names the registry rewrites itself — never usable as named aliases. */
const RESERVED_ALIAS_NAMES: ReadonlySet<string> = new Set([...REGISTRY_STATES, "latest"]);

export default function RegistryPage() {
  const { task } = useTask();
  const qc = useQueryClient();
  const navigate = useNavigate();

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

  /** Sorted named aliases pointing at each model version (empty when none has been set). */
  const aliasByVersion = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const [name, entry] of Object.entries(summaryQ.data?.named_aliases ?? {})) {
      const names = map.get(entry.model_version) ?? [];
      names.push(name);
      map.set(entry.model_version, names);
    }
    for (const names of map.values()) names.sort();
    return map;
  }, [summaryQ.data]);

  /** Dropdown label: "{alias(es)}-{version}" when a named alias is set, otherwise the version. */
  const labelOf = useBundleLabel(task);
  const loadRegisteredMut = useMutation({
    mutationFn: (alias: string) => loadRegisteredBundle(task, alias),
    onSuccess: (bundle) => {
      navigate(`/inference?model_version=${encodeURIComponent(bundle.model_version)}`);
    },
  });

  // Versions the user can act on: bundles folder + registry versions + history/audit.
  const versions = useMemo(() => {
    const set = new Set<string>(bundlesQ.data?.bundles ?? []);
    for (const v of Object.keys(versionStates)) set.add(v);
    for (const h of historyQ.data?.history ?? []) if (h.model_version) set.add(h.model_version);
    for (const a of auditQ.data?.audit ?? []) if (a.model_version) set.add(a.model_version);
    return [...set].reverse();
  }, [bundlesQ.data, versionStates, historyQ.data, auditQ.data]);

  // Versions that exist in the registry — only these can be aliased / promoted /
  // rolled back. Undefined while the summary is still loading so options are not
  // gated (and misleadingly suffixed) before we know anything.
  const registeredVersions = useMemo<Set<string> | undefined>(() => {
    if (!summaryQ.data && !summaryQ.isError) return undefined;
    return new Set(Object.keys(versionStates));
  }, [summaryQ.data, summaryQ.isError, versionStates]);

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
                <div className="row fixed" style={{ marginBottom: 10 }}>
                  <span>
                    <span className="muted" style={{ marginRight: 8 }}>latest →</span>
                    <code>{aliases.latest}</code>{" "}
                    <StatusBadge status={versionStates[aliases.latest] ?? "candidate"} />
                  </span>
                  <button
                    type="button"
                    className="btn"
                    disabled={loadRegisteredMut.isPending}
                    onClick={() => loadRegisteredMut.mutate("latest")}
                  >
                    Use in Inference
                  </button>
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
                      {aliases[s] && (
                        <button
                          type="button"
                          className="btn"
                          style={{ marginTop: 6 }}
                          disabled={loadRegisteredMut.isPending}
                          onClick={() => loadRegisteredMut.mutate(s)}
                        >
                          Use in Inference
                        </button>
                      )}
                    </div>
                  </span>
                ))}
              </div>

              {[...new Set([aliases.latest, ...Object.values(aliases)].filter((v): v is string => !!v))].map((v) => (
                <LineageRail key={v} task={task} modelVersion={v} />
              ))}
              {loadRegisteredMut.isError && <ErrorNote error={loadRegisteredMut.error} />}
            </>
          )}
        </Card>
      )}

      <AliasCard
        task={task}
        versions={versions}
        registered={registeredVersions}
        namedAliases={summaryQ.data?.named_aliases ?? {}}
        labelOf={labelOf}
        onDone={invalidate}
      />
      <RegisterCard task={task} versions={versions} labelOf={labelOf} onDone={() => { invalidate(); qc.invalidateQueries({ queryKey: ["bundles-list", task] }); }} />
      <PromoteCard task={task} versions={versions} registered={registeredVersions} labelOf={labelOf} onDone={invalidate} />
      <RollbackCard task={task} versions={versions} registered={registeredVersions} labelOf={labelOf} onDone={invalidate} />

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

      <ArtifactCleanupCard task={task} versions={versions} labelOf={labelOf} onDone={invalidate} />

      <AuditCard task={task} audit={auditQ.data?.audit ?? []} loading={auditQ.isLoading} error={auditQ.error} aliasByVersion={aliasByVersion} />
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
function ArtifactCleanupCard({ task, versions, labelOf, onDone }: { task: string; versions: string[]; labelOf: (version: string) => string; onDone: () => void }) {
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
                    <option key={v} value={v}>{labelOf(v)}</option>
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

function AliasCard({
  task,
  versions,
  registered,
  namedAliases,
  labelOf,
  onDone,
}: {
  task: string;
  versions: string[];
  registered?: Set<string>;
  namedAliases: Record<string, RegistryAliasEntry>;
  labelOf: (version: string) => string;
  onDone: () => void;
}) {
  const [name, setName] = useState("");
  const [version, setVersion] = useState("");

  // Client-side mirror of ModelRegistry.validate_alias_name so invalid names get
  // feedback before the round-trip instead of a raw API error.
  const cleanedName = name.trim();
  let aliasProblem: string | null = null;
  if (cleanedName) {
    if (!ALIAS_NAME_RE.test(cleanedName)) {
      aliasProblem = "Use only lowercase letters/digits with '.', '_' or '-' after the first character — e.g. 'rf-v1'. No capitals, spaces, or other symbols.";
    } else if (RESERVED_ALIAS_NAMES.has(cleanedName)) {
      aliasProblem = `'${cleanedName}' is reserved for lifecycle states; choose a different name.`;
    }
  }

  const setMut = useMutation({
    mutationFn: () => {
      if (!cleanedName) throw new Error("Give the alias a name (lowercase letters, digits, '.', '_' or '-').");
      if (aliasProblem) throw new Error(aliasProblem);
      if (!version) throw new Error("Pick the model version the alias should point at.");
      if (registered && !registered.has(version)) {
        throw new Error(`Version ${version} is not registered in this task's registry — register it first with “Register a bundle” below.`);
      }
      return setRegistryAlias(task, cleanedName, version);
    },
    onSuccess: () => {
      setName("");
      onDone();
    },
  });
  const deleteMut = useMutation({
    mutationFn: (alias: string) => deleteRegistryAlias(task, alias),
    onSuccess: onDone,
  });

  const entries = Object.entries(namedAliases);

  return (
    <Card
      title="Model aliases"
      sub="Stable, human-chosen names for immutable model versions (e.g. “baseline”, “credit-card-v2”). Aliases are separate from the lifecycle states above and are recorded in the audit log."
    >
      {entries.length === 0 ? (
        <p className="muted">No aliases yet — name a version below.</p>
      ) : (
        <Table<[string, RegistryAliasEntry]>
          columns={["Alias", "Model version", "Created", "By", ""]}
          rows={entries}
          keyOf={([alias]) => alias}
          render={([alias, entry]) => [
            <td key="a" className="mono">{alias}</td>,
            <td key="v" className="mono">
              <code>{entry.model_version}</code>
            </td>,
            <td key="c" className="mono">{new Date(entry.created_at).toLocaleString()}</td>,
            <td key="b">{entry.created_by ?? "—"}</td>,
            <td key="x">
              <button
                className="btn"
                disabled={deleteMut.isPending}
                onClick={() => deleteMut.mutate(alias)}
              >
                Delete
              </button>
            </td>,
          ]}
        />
      )}

      <div className="row" style={{ marginTop: 12 }}>
        <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 170 }}>
          <label>Alias name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="rf-v1 or baseline-2026-11"
            spellCheck={false}
            maxLength={64}
          />
          {aliasProblem ? (
            <span className="hint" style={{ color: "var(--err)" }}>{aliasProblem}</span>
          ) : (
            <span className="hint">lowercase a–z / 0–9, then '.', '_' or '-'; max 64</span>
          )}
        </div>
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Model version</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v} value={v} disabled={registered !== undefined && !registered.has(v)}>
                {labelOf(v)}{registered !== undefined && !registered.has(v) ? " — register first" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>
      {setMut.isError && <ErrorNote error={setMut.error} />}
      {deleteMut.isError && <ErrorNote error={deleteMut.error} />}
      <button className="btn primary" disabled={!task || aliasProblem !== null || setMut.isPending} onClick={() => setMut.mutate()}>
        {setMut.isPending ? "Saving…" : "Save alias"}
      </button>
    </Card>
  );
}

function RegisterCard({ task, versions, labelOf, onDone }: { task: string; versions: string[]; labelOf: (version: string) => string; onDone: () => void }) {
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
              <option key={v} value={v}>{labelOf(v)}</option>
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

function PromoteCard({ task, versions, registered, labelOf, onDone }: { task: string; versions: string[]; registered?: Set<string>; labelOf: (version: string) => string; onDone: () => void }) {
  const [version, setVersion] = useState("");
  const [state, setState] = useState<string>("validated");
  const [approversText, setApproversText] = useState("mehdi");
  const [required, setRequired] = useState(1);

  const mut = useMutation({
    mutationFn: () => {
      const approvers = approversText.split(",").map((s) => s.trim()).filter(Boolean);
      if (!approvers.length) throw new Error("Provide at least one approver name (comma-separated).");
      if (!version) throw new Error("Pick the model version to promote.");
      if (registered && !registered.has(version)) {
        throw new Error(`Version ${version} is not registered in this task's registry — register it first with “Register a bundle” above.`);
      }
      return promoteBundle(task, { model_version: version, state, approvers, required_approvals: required });
    },
    onSuccess: onDone,
  });

  return (
    <Card title="Promote" sub="Runs bundle quality checks and requires N unique named approvers before updating the target alias.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Model version</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v} value={v} disabled={registered !== undefined && !registered.has(v)}>
                {labelOf(v)}{registered !== undefined && !registered.has(v) ? " — register first" : ""}
              </option>
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
  registered,
  labelOf,
  onDone,
}: {
  task: string;
  versions: string[];
  registered?: Set<string>;
  labelOf: (version: string) => string;
  onDone: () => void;
}) {
  const [state, setState] = useState("production");
  const [version, setVersion] = useState("");

  const mut = useMutation({
    mutationFn: () => {
      if (version && registered && !registered.has(version)) {
        throw new Error(`Version ${version} is not registered in this task's registry — register it first with “Register a bundle” above.`);
      }
      return rollbackBundle(task, state, version || undefined);
    },
    onSuccess: onDone,
  });

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
              <option key={v} value={v} disabled={registered !== undefined && !registered.has(v)}>
                {labelOf(v)}{registered !== undefined && !registered.has(v) ? " — register first" : ""}
              </option>
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
  aliasByVersion,
}: {
  task: string;
  audit: AuditEvent[];
  loading: boolean;
  error: unknown;
  aliasByVersion: Map<string, string[]>;
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
              columns={["Timestamp", "Event", "Model version", "Alias", "Details"]}
              rows={[...rows].reverse()}
              keyOf={(a, i) => `${a.timestamp}-${i}`}
              render={(a) => [
                <td key="t" className="mono">{a.timestamp}</td>,
                <td key="e"><Badge tone={a.event_type === "promote" ? "info" : a.event_type === "rollback" ? "warn" : "muted"}>{a.event_type}</Badge></td>,
                <td key="v" className="mono">{a.model_version ?? "—"}</td>,
                <td key="a">{(() => {
                  const names = a.model_version ? aliasByVersion.get(a.model_version) : undefined;
                  return names && names.length > 0 ? names.join(", ") : <span className="muted">—</span>;
                })()}</td>,
                <td key="d" className="mono" style={{ fontSize: 12 }}>{JSON.stringify(a.details ?? a)}</td>,
              ]}
            />
          )}
        </>
      )}
    </Card>
  );
}
