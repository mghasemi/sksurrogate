import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { UploadCloud } from "lucide-react";

import {
  addProjectArtifact,
  clearDatasetTemporal,
  downloadSynthetic,
  generateSynthetic,
  getDatasetCV,
  getDatasetMetadata,
  getDatasetTemporal,
  getProject,
  getTargetStats,
  inspectDataset,
  listSynthetic,
  previewDataset,
  registerDataset,
  setDatasetCV,
  setDatasetProblemType,
  clearDatasetProblemType,
  setDatasetTarget,
  setDatasetTemporal,
  setIgnoredColumns,
  validateDataset,
  ApiError,
} from "../api/client";
import type {
  CVParamDef,
  CVSpec,
  DatasetInspectResponse,
  DatasetMetadata,
  DatasetPreview,
  RegisterDatasetResponse,
  SensitiveScanReport,
  TemporalSpecState,
  TemporalValidationReport,
} from "../api/client";
import { Card, EmptyState, ErrorNote, Loading, Table, Badge, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

const DATASET_TYPES = [
  "float64",
  "int64",
  "datetime64",
  "other",
  "text",
  "binary",
  "categorical",
  "label",
  "obsolete",
];

/** Sentinel type: the column is stored but excluded from every modeling path. */
const IGNORE_TYPE = "ignore";

/** Human-readable label for a stored CV spec, e.g. "StratifiedKFold(n_splits=5)". */
function describeCv(spec: CVSpec): string {
  if (typeof spec === "number") return `${spec}-fold cross validation`;
  const type = String((spec as Record<string, unknown>).type ?? "");
  if (!type || type === "default") return "ShuffleSplit(n_splits=3, test_size=0.25)";
  const params = Object.entries(spec).filter(([k]) => k !== "type");
  if (!params.length) return type;
  return `${type}(${params.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})`;
}

export default function DatasetsPage() {
  const { task } = useTask();
  const qc = useQueryClient();
  const [searchParams] = useSearchParams();
  const projectId = searchParams.get("project_id");
  const projectQ = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => {
      if (!projectId) throw new Error("Project id is required.");
      return getProject(projectId);
    },
    enabled: Boolean(projectId),
  });

  const [target, setTarget] = useState("");
  const [partition, setPartition] = useState("train");
  const [file, setFile] = useState<File | null>(null);
  const [typeOverrides, setTypeOverrides] = useState<Record<string, string>>({});
  const [ignoredUpload, setIgnoredUpload] = useState<string[]>([]);
  const [binarizeLabel, setBinarizeLabel] = useState(false);
  const [validationPartition, setValidationPartition] = useState("train");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Bumped after a successful registration so every task-keyed query gets a NEW
  // cache entry and therefore a fresh fetch. `invalidateQueries` alone was not
  // enough: these queries are created when the task is first selected (before
  // any dataset exists), answer 404, and — after React StrictMode's
  // mount/unmount/mount cycle — end up pending with no observer that
  // invalidation could re-trigger. Measured: no request followed the
  // invalidation, so the metadata card stayed blank and the CV/target cards
  // kept showing their pre-registration state.
  const [dataRevision, setDataRevision] = useState(0);

  const metaQ = useQuery({
    queryKey: ["dataset-meta", task, dataRevision],
    queryFn: async () => {
      try {
        return await getDatasetMetadata(task);
      } catch (err) {
        // 404 means "this task has no registered dataset yet" — a state, not a
        // failure. Rendering it as an error made a fresh task look broken.
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task,
    retry: false,
  });

  const [previewPartition, setPreviewPartition] = useState<string | null>(null);
  const previewQ = useQuery<DatasetPreview>({
    queryKey: ["dataset-preview", task, previewPartition, dataRevision],
    queryFn: () => previewDataset(task, previewPartition!),
    enabled: !!task && !!previewPartition,
  });

  const inspectMut = useMutation({
    mutationFn: (upload: File) => inspectDataset(task, upload),
    onSuccess: (data: DatasetInspectResponse) => {
      setTypeOverrides(data.deduced_types);
      setIgnoredUpload([]);
      setTarget(data.target_candidates[0] ?? data.columns[0] ?? "");
      setBinarizeLabel(false);
    },
  });

  const validateMut = useMutation({
    mutationFn: (body: { file?: File; partition?: string }) => validateDataset(task, body),
  });
  const validateStoredMut = useMutation({
    mutationFn: (storedPartition: string) => validateDataset(task, { partition: storedPartition }),
  });
  const linkProjectPartitionMut = useMutation({
    mutationFn: ({ taskName, partitionName }: { taskName: string; partitionName: string }) => {
      if (!projectId) throw new Error("Project context is missing.");
      return addProjectArtifact(projectId, {
        kind: "dataset_partition",
        task_name: taskName,
        artifact_id: partitionName,
      });
    },
    onSuccess: (project) => {
      qc.setQueryData(["project", project.project_id], project);
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project-summary", project.project_id] });
    },
  });

  const registerMut = useMutation({
    mutationFn: (vars: { target: string; partition: string; file: File }) =>
      registerDataset(task, vars.target, vars.partition, vars.file, {
        type_overrides: Object.fromEntries(
          Object.entries(typeOverrides).filter(([column]) => !ignoredUpload.includes(column)),
        ),
        ignored_columns: ignoredUpload,
        binarize_label: binarizeLabel,
      }),
    onSuccess: (registered) => {
      // Every task-keyed query in this page must be invalidated, not just the
      // metadata: the CV and preview cards mount as soon as a task is selected,
      // so before the first registration they answer 404 ("No dataset
      // registered for task ...") — and React Query keeps that cached error
      // until the query is invalidated. Missing the CV key left a registered
      // dataset showing a permanent 404 in the cross-validation card.
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["target-stats", task] });
      qc.invalidateQueries({ queryKey: ["dataset-cv", task] });
      qc.invalidateQueries({ queryKey: ["dataset-preview", task] });
      setDataRevision((revision) => revision + 1);
      if (projectId) {
        linkProjectPartitionMut.mutate({
          taskName: registered.task_name,
          partitionName: registered.partition,
        });
      }
      setFile(null);
      setTypeOverrides({});
      setIgnoredUpload([]);
      setBinarizeLabel(false);
      inspectMut.reset();
      validateMut.reset();
      if (fileRef.current) fileRef.current.value = "";
    },
  });

  const inspected = inspectMut.data ?? null;
  const setUpload = (nextFile: File | null) => {
    setFile(nextFile);
    inspectMut.reset();
    validateMut.reset();
    registerMut.reset();
    setTypeOverrides({});
    setIgnoredUpload([]);
    setBinarizeLabel(false);
  };

  // A 404 here is a *state*, not a failure: the task exists but has no dataset
  // registered yet. Treating it as an error left the card either silent (a
  // StrictMode double-mount cancels the first fetch, so the query can sit in
  // "pending, not fetching") or stuck on a stale error after registering.
  const cvQ = useQuery({
    queryKey: ["dataset-cv", task, dataRevision],
    queryFn: async () => {
      try {
        return await getDatasetCV(task);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task,
    retry: false,
  });
  const [cvChoice, setCvChoice] = useState<string>("");
  // Editable constructor parameters of the selected splitter (raw input strings).
  const [cvParamValues, setCvParamValues] = useState<Record<string, string>>({});
  const setCvMut = useMutation({
    mutationFn: (vars: { spec: CVSpec; params?: Record<string, unknown> }) =>
      setDatasetCV(task, vars.spec, vars.params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dataset-cv", task, dataRevision] }),
  });

  // Edit the registered prediction target (metadata-only re-point).
  const [editingTarget, setEditingTarget] = useState(false);
  const [targetDraft, setTargetDraft] = useState("");
  const setTargetMut = useMutation({
    mutationFn: (newTarget: string) => setDatasetTarget(task, newTarget),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["target-stats", task] });
      setDataRevision((revision) => revision + 1);
      setEditingTarget(false);
    },
  });

  // Declared problem type: the one place classification-vs-regression is decided.
  // Everything downstream (Quick start preset, baselines, monitoring) reads it.
  const [editingProblemType, setEditingProblemType] = useState(false);
  const setProblemTypeMut = useMutation({
    mutationFn: (value: "regression" | "classification") => setDatasetProblemType(task, value),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      setEditingProblemType(false);
    },
  });
  const clearProblemTypeMut = useMutation({
    mutationFn: () => clearDatasetProblemType(task),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dataset-meta", task] }),
  });

  // Temporal declaration: also decides which splitters the CV card may offer
  // (decision D2 — SetCV hard-rejects order-violating splitters once declared).
  const temporalQ = useQuery({
    queryKey: ["dataset-temporal", task, dataRevision],
    queryFn: async () => {
      try {
        return await getDatasetTemporal(task);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task,
    retry: false,
  });
  const temporalDeclared = !!temporalQ.data?.spec;
  // Temporal tasks may only use order-respecting splitters (decision D2), and
  // that set includes temporal-only ones (RollingOriginSplit) that are not in
  // the standard option list.
  const cvOptions = temporalDeclared
    ? temporalQ.data?.temporal_splitters ?? []
    : cvQ.data?.options ?? [];
  // Preselect the splitter matching the stored spec's type when available.
  const storedType =
    cvQ.data && typeof cvQ.data.current === "object" ? String(cvQ.data.current.type ?? "") : "";
  const selectValue = cvChoice || (cvOptions.includes(storedType) ? storedType : "");

  // Constructor parameters exposed by the selected splitter.
  const paramDefs: CVParamDef[] = cvQ.data?.param_defs?.[selectValue] ?? [];

  /** Prefill parameter fields from the stored spec (if it matches), else defaults. */
  const initialValuesFor = (splitter: string): Record<string, string> => {
    const defs = cvQ.data?.param_defs?.[splitter] ?? [];
    const current = cvQ.data?.current;
    const stored = typeof current === "object" && String(current.type) === splitter ? current : null;
    const values: Record<string, string> = {};
    for (const def of defs) {
      const fromStored = stored ? (stored as Record<string, unknown>)[def.name] : undefined;
      const value = fromStored !== undefined ? fromStored : def.default;
      // null/undefined default (e.g. test_size "auto") → empty field.
      values[def.name] = value === undefined || value === null ? "" : String(value);
    }
    return values;
  };

  // Seed the fields when data loads with a preselected splitter.
  useEffect(() => {
    if (cvQ.data && !cvChoice && selectValue) setCvParamValues(initialValuesFor(selectValue));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cvQ.data, cvChoice, selectValue]);

  const invalidParam = paramDefs.find((def) => {
    if (def.kind === "bool") return false;
    const raw = cvParamValues[def.name];
    if (raw === undefined || raw.trim() === "") return false; // blank → splitter default
    const num = Number(raw);
    if (!Number.isFinite(num)) return true;
    if (def.kind === "int" && !Number.isInteger(num)) return true;
    if (def.name === "n_splits" && num < 2) return true;
    return false;
  });

  const saveCv = () => {
    if (!selectValue || invalidParam) return;
    const params: Record<string, unknown> = {};
    for (const def of paramDefs) {
      const raw = cvParamValues[def.name];
      if (raw === undefined || raw.trim() === "") continue; // omit → splitter default
      params[def.name] = def.kind === "bool" ? raw === "true" : Number(raw);
    }
    setCvMut.mutate({ spec: { type: selectValue }, params });
  };
  const partitions = metaQ.data ? Object.keys(metaQ.data.partitions ?? {}) : [];
  const previewRows = previewQ.data?.rows ?? [];
  const previewCols = previewRows.length ? Object.keys(previewRows[0]) : [];

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Datasets</h1>
        <span className="desc">Register CSV partitions, inspect schema and fingerprint for task “{task || "…"}”</span>
      </div>

      {projectId && (
        <Card title="Research project">
          {projectQ.isLoading && <Loading label="Loading project…" />}
          {projectQ.isError && <ErrorNote error={projectQ.error} />}
          {projectQ.data && (
            <p className="muted">
              A successfully registered partition will be linked to “{projectQ.data.name}”.
            </p>
          )}
          {linkProjectPartitionMut.isError && <ErrorNote error={linkProjectPartitionMut.error} />}
        </Card>
      )}

      {!task && (
        <div className="error-note">Pick a task name in the top bar to work with datasets.</div>
      )}

      <Card title="Register a partition" sub="Upload, review the inferred schema, then commit the CSV as a task partition.">
        <div className="row top">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 180 }}>
            <label>Target column (what to predict)</label>
            {inspected ? (
              <select value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">— select —</option>
                {inspected.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}{inspected.target_candidates.includes(column) ? " (label candidate)" : ""}
                  </option>
                ))}
              </select>
            ) : (
              <input value={target} readOnly placeholder="Choose after inspecting the CSV" spellCheck={false} />
            )}
            <span className="hint muted" style={{ fontSize: 11.5, marginTop: 3 }}>
              {inspected && target
                ? `Downstream statistical analysis (feature analysis, correlations, EDA) and experiments all predict this column.`
                : "Select the column to predict after inspecting the CSV."}
            </span>
          </div>
          <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 120 }}>
            <label>Partition</label>
            <input value={partition} onChange={(e) => setPartition(e.target.value)} placeholder="train / validation / test" spellCheck={false} />
          </div>
        </div>

        <div
          className={`file-drop${dragOver ? " drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) setUpload(f);
          }}
          onClick={() => fileRef.current?.click()}
        >
          <UploadCloud size={22} style={{ marginBottom: 6 }} />
          {file ? (
            <span>
              <strong>{file.name}</strong> ({fmtNum(file.size)} bytes) — click to replace
            </span>
          ) : (
            <span>Drop a CSV here or click to browse</span>
          )}
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.tsv,text/csv"
            style={{ display: "none" }}
            onChange={(e) => setUpload(e.target.files?.[0] ?? null)}
          />
        </div>

        {inspectMut.isError && <ErrorNote error={inspectMut.error} />}
        {file && (
          <button
            className="btn"
            disabled={!task || inspectMut.isPending}
            onClick={() => inspectMut.mutate(file)}
          >
            {inspectMut.isPending ? "Inspecting…" : "Inspect CSV"}
          </button>
        )}

        {inspected && (
          <>
            <div className="hint" style={{ marginTop: 12 }}>
              {fmtNum(inspected.rows)} rows · {fmtNum(inspected.columns.length)} columns. Review the detected type for each column before committing.
            </div>
            <Table<string>
              columns={["Column", "Detected / selected type", "Sensitive flags"]}
              rows={inspected.columns}
              keyOf={(column) => column}
              render={(column) => {
                const scan = inspected.sensitive_scan;
                const isPii = scan?.pii_columns.includes(column) ?? false;
                const isSensitive = scan?.sensitive_columns.includes(column) ?? false;
                const isIgnored = ignoredUpload.includes(column);
                return [
                  <td key="name" className="mono">{column}</td>,
                  <td key="type">
                    <select
                      aria-label={`Type for ${column}`}
                      value={
                        isIgnored
                          ? IGNORE_TYPE
                          : typeOverrides[column] ?? inspected.deduced_types[column] ?? "other"
                      }
                      onChange={(event) => {
                        const value = event.target.value;
                        if (value === IGNORE_TYPE) {
                          setIgnoredUpload((current) =>
                            current.includes(column) ? current : [...current, column],
                          );
                          setTypeOverrides((current) => {
                            const next = { ...current };
                            delete next[column];
                            return next;
                          });
                          return;
                        }
                        setIgnoredUpload((current) => current.filter((name) => name !== column));
                        setTypeOverrides((current) => ({ ...current, [column]: value }));
                      }}
                    >
                      {DATASET_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
                      <option value={IGNORE_TYPE} disabled={column === target}>ignore (exclude from work)</option>
                    </select>
                  </td>,
                  <td key="flags">
                    {isIgnored && <Badge tone="warn">ignored</Badge>}
                    {isPii && <Badge tone="warn">PII</Badge>}
                    {isSensitive && <Badge tone="warn">sensitive</Badge>}
                    {!isPii && !isSensitive && !isIgnored && <span className="muted">—</span>}
                  </td>,
                ];
              }}
            />
            <div className="hint" style={{ marginTop: 8 }}>
              Columns marked <code>ignore</code> are stored with the partition but excluded from
              experiments, sensitivity/correlation analysis, dataset profiles, synthetic data and
              monitoring drift. The target column cannot be ignored.
            </div>

            <div className="row" style={{ alignItems: "center", marginTop: 12 }}>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={binarizeLabel}
                  onChange={(e) => setBinarizeLabel(e.target.checked)}
                />
                Binarize / ordinal-encode target labels
              </label>
            </div>

            {validateMut.isError && <ErrorNote error={validateMut.error} />}
            {validateMut.data?.valid && (
              <div className="success-note">CSV passed validation against the currently registered schema.</div>
            )}
            {!metaQ.data?.dataset_schema && (
              <div className="hint">
                Upload validation becomes available after the task has a registered schema.
              </div>
            )}
            <div className="row">
              <button
                className="btn"
                disabled={!task || !metaQ.data?.dataset_schema || validateMut.isPending}
                onClick={() => file && validateMut.mutate({ file })}
              >
                {validateMut.isPending ? "Validating…" : "Validate upload"}
              </button>
              <button
                className="btn primary"
                disabled={!task || !target || !partition || !file || registerMut.isPending}
                onClick={() => file && registerMut.mutate({ target, partition, file })}
              >
                {registerMut.isPending ? "Committing…" : "Commit dataset"}
              </button>
            </div>
          </>
        )}

        {registerMut.isError && <ErrorNote error={registerMut.error} />}
        {registerMut.isSuccess && (
          <>
            <div className="success-note">
              Registered {fmtNum((registerMut.data as RegisterDatasetResponse).rows)} rows → fingerprint{" "}
              <code>{(registerMut.data as RegisterDatasetResponse).dataset_fingerprint ?? "n/a"}</code>
            </div>
            <SensitiveScanNote scan={(registerMut.data as RegisterDatasetResponse).sensitive_scan} />
          </>
        )}

      </Card>

      <Card
        title="Cross-validation partitioning"
        sub="Choose how the data is split into train/test folds; saved to the task's tracking pipeline and used by downstream training."
      >
        {cvQ.isLoading && task && <Loading />}
        {cvQ.isError && <ErrorNote error={cvQ.error} />}
        {cvQ.data === null && (
          <EmptyState
            title="No dataset registered for this task yet"
            hint="Register a partition above; the splitter options appear here as soon as the first partition is stored."
          />
        )}
        {cvQ.data && (
          <>
            {temporalDeclared && (
              <div className="muted" style={{ marginBottom: 12 }}>
                Temporal task: only order-respecting splitters are offered — the server rejects
                shuffled splitters once a temporal structure is declared.
              </div>
            )}
            <div className="row">
              <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
                <label>Splitter</label>
                <select
                  value={selectValue}
                  onChange={(e) => {
                    setCvChoice(e.target.value);
                    if (e.target.value) setCvParamValues(initialValuesFor(e.target.value));
                  }}
                >
                  {cvOptions.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {paramDefs.length > 0 && (
              <div className="row">
                {paramDefs.map((def) =>
                  def.kind === "bool" ? (
                    <label key={def.name} className="checkbox-row" style={{ marginBottom: 12 }}>
                      <input
                        type="checkbox"
                        checked={cvParamValues[def.name] === "true"}
                        onChange={(e) =>
                          setCvParamValues((p) => ({ ...p, [def.name]: String(e.target.checked) }))
                        }
                      />
                      {def.name}
                    </label>
                  ) : (
                    <div
                      key={def.name}
                      className="field fixed"
                      style={{ flex: "0 1 160px", minWidth: 120, marginBottom: 12 }}
                    >
                      <label>{def.name}</label>
                      <input
                        type="number"
                        step={def.kind === "int" ? 1 : "any"}
                        min={def.name === "n_splits" ? 2 : undefined}
                        value={cvParamValues[def.name] ?? ""}
                        onChange={(e) => setCvParamValues((p) => ({ ...p, [def.name]: e.target.value }))}
                      />
                    </div>
                  ),
                )}
              </div>
            )}

            {invalidParam && (
              <div className="error-note">
                “{invalidParam.name}” must be a valid number{invalidParam.kind === "int" ? " (whole)" : ""}
                {invalidParam.name === "n_splits" ? ", at least 2" : ""}. Leave it blank to use the
                splitter default.
              </div>
            )}

            <dl className="kv">
              <dt>Current</dt>
              <dd className="mono">{cvQ.data.stored ? describeCv(cvQ.data.current) : `${describeCv(cvQ.data.current)} (default)`}</dd>
              {typeof cvQ.data.current === "object" && (
                <>
                  <dt>Stored spec</dt>
                  <dd className="mono">{JSON.stringify(cvQ.data.current)}</dd>
                </>
              )}
            </dl>

            {setCvMut.isError && <ErrorNote error={setCvMut.error} />}
            {setCvMut.isSuccess && (
              <div className="success-note">Saved cross-validation method: <code>{describeCv(setCvMut.data.cv)}</code></div>
            )}

            <button
              className="btn primary"
              disabled={!task || !selectValue || !!invalidParam || setCvMut.isPending}
              onClick={saveCv}
            >
              {setCvMut.isPending ? "Saving…" : "Save cross-validation method"}
            </button>
          </>
        )}
      </Card>

      <div id="temporal-structure">
        <TemporalCard task={task} revision={dataRevision} fallbackColumns={previewCols} />
      </div>

      <Card
        title="Task definition"
        sub="What you predict (target), how it is scored (learning problem), and whether rows are independent (time structure)."
      >
        {metaQ.isLoading && task && <Loading />}
        {metaQ.isError && <ErrorNote error={metaQ.error} />}
        {metaQ.data === null && (
          <EmptyState
            title="No dataset registered for this task yet"
            hint="Upload a CSV above and commit it as a partition; the schema, fingerprint and partitions appear here."
          />
        )}
        {metaQ.data && (
          <>
            <dl className="kv">
              <dt>Target</dt>
              <dd>
                {editingTarget ? (
                  <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <select
                      aria-label="New target column"
                      value={targetDraft}
                      onChange={(e) => setTargetDraft(e.target.value)}
                    >
                      <option value="">— select column to predict —</option>
                      {(metaQ.data!.dataset_columns ?? []).map((column) => (
                        <option key={column} value={column}>{column}</option>
                      ))}
                    </select>
                    <button
                      className="btn primary"
                      disabled={!targetDraft || targetDraft === metaQ.data!.target_name || setTargetMut.isPending}
                      onClick={() => targetDraft && setTargetMut.mutate(targetDraft)}
                    >
                      {setTargetMut.isPending ? "Saving…" : "Save target"}
                    </button>
                    <button className="btn" disabled={setTargetMut.isPending} onClick={() => setEditingTarget(false)}>
                      Cancel
                    </button>
                  </div>
                ) : (
                  <span className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <span>{metaQ.data.target_name ?? "—"}</span>
                    <button className="btn" onClick={() => {
                      setTargetDraft(metaQ.data!.target_name ?? "");
                      setEditingTarget(true);
                    }}>
                      Change
                    </button>
                  </span>
                )}
              </dd>
              {setTargetMut.isError && (
                <dd className="error-note" style={{ gridColumn: "2" }}>
                  {setTargetMut.error instanceof Error ? setTargetMut.error.message : String(setTargetMut.error)}
                </dd>
              )}
              {setTargetMut.isSuccess && !editingTarget && (
                <dd className="success-note" style={{ gridColumn: "2" }}>
                  Target changed to <code>{setTargetMut.data.target}</code> (was{" "}
                  {setTargetMut.data.previous_target ?? "—"}). Downstream analyses and features are re-pointed.
                </dd>
              )}
              <dt>Learning problem</dt>
              <dd>
                <span className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                  {editingProblemType ? (
                    <>
                      <select
                        aria-label="Problem type"
                        defaultValue={
                          metaQ.data.problem_type ?? metaQ.data.problem_type_inferred ?? "regression"
                        }
                        onChange={(e) =>
                          setProblemTypeMut.mutate(e.target.value as "regression" | "classification")
                        }
                        disabled={setProblemTypeMut.isPending}
                      >
                        <option value="regression">regression — predict a number</option>
                        <option value="classification">classification — predict a class</option>
                        {metaQ.data.temporal?.declared && (
                          <option value="" disabled>
                            (a temporal task stays regression)
                          </option>
                        )}
                      </select>
                      <button className="btn" onClick={() => setEditingProblemType(false)}>
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <Badge tone={metaQ.data.problem_type === "classification" ? "warn" : "ok"}>
                        {metaQ.data.problem_type ?? metaQ.data.problem_type_inferred ?? "unknown"}
                      </Badge>
                      <span className="muted">
                        {metaQ.data.problem_type_source === "user" ? "set by you" : "inferred from the target"}
                      </span>
                      {metaQ.data.temporal?.declared && (
                        <Badge tone="info">temporal</Badge>
                      )}
                      <button className="btn" onClick={() => setEditingProblemType(true)}>
                        Change
                      </button>
                      {metaQ.data.problem_type_source === "user" && (
                        <button
                          className="btn"
                          disabled={clearProblemTypeMut.isPending}
                          onClick={() => clearProblemTypeMut.mutate()}
                        >
                          Reset to inferred
                        </button>
                      )}
                    </>
                  )}
                </span>
                {metaQ.data.problem_type_reason && (
                  <div className="muted" style={{ marginTop: 4 }}>
                    {metaQ.data.problem_type_reason}
                  </div>
                )}
                {metaQ.data.temporal?.declared ? (
                  <div className="muted" style={{ marginTop: 4 }}>
                    A forecasting task is a <b>regression</b> problem on a time axis: it is
                    scored with MAE/RMSE/MASE over a rolling-origin backtest, never with
                    accuracy. Time structure is configured below.
                  </div>
                ) : (
                  <div className="muted" style={{ marginTop: 4 }}>
                    There is no “time series” choice here: time structure is a{" "}
                    <b>separate axis</b>, not a third learning problem. Declare a time
                    column to forecast{" "}
                    <button
                      className="btn"
                      style={{ padding: "0 6px", fontSize: "0.85em" }}
                      onClick={() => {
                        const card = document.getElementById("temporal-structure");
                        card?.scrollIntoView({ behavior: "smooth", block: "start" });
                        card?.querySelector("select")?.focus();
                      }}
                    >
                      Declare time structure
                    </button>
                  </div>
                )}
                {setProblemTypeMut.isError && (
                  <div className="error-note" style={{ marginTop: 4 }}>
                    {setProblemTypeMut.error instanceof Error
                      ? setProblemTypeMut.error.message
                      : String(setProblemTypeMut.error)}
                  </div>
                )}
              </dd>
              <dt>Time structure</dt>
              <dd>
                {metaQ.data.temporal?.declared ? (
                  <span className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <Badge tone="ok">temporal</Badge>
                    <span className="muted">
                      time column{" "}
                      <code>{String((metaQ.data.temporal.spec as Record<string, unknown> | null)?.["time_column"] ?? "—")}</code>
                      {", horizon "}
                      {String((metaQ.data.temporal.spec as Record<string, unknown> | null)?.["horizon"] ?? 1)}
                      {" — evaluation leads with the rolling-origin backtest"}
                    </span>
                  </span>
                ) : (
                  <span className="muted">
                    none — rows are i.i.d. and cross-validation applies. Declare a time
                    column in the Temporal structure card to forecast instead.
                  </span>
                )}
              </dd>
              <dt>Fingerprint</dt>
              <dd className="mono">{metaQ.data.dataset_fingerprint ?? "—"}</dd>
              <dt>Feature count</dt>
              <dd>
                {fmtNum(metaQ.data.effective_feature_count ?? metaQ.data.dataset_feature_count)}
                {(metaQ.data.ignored_columns ?? []).length > 0 && (
                  <span className="muted">
                    {" "}· {(metaQ.data.ignored_columns ?? []).length} ignored
                  </span>
                )}
              </dd>
              {(metaQ.data.ignored_columns ?? []).length > 0 && (
                <>
                  <dt>Ignored columns</dt>
                  <dd className="mono">{(metaQ.data.ignored_columns ?? []).join(", ")}</dd>
                </>
              )}
            </dl>

            {partitions.length > 0 && (
              <>
                <div className="row" style={{ alignItems: "end", marginTop: 12 }}>
                  <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 180 }}>
                    <label>Validate stored partition</label>
                    <select
                      value={partitions.includes(validationPartition) ? validationPartition : partitions[0]}
                      onChange={(e) => setValidationPartition(e.target.value)}
                    >
                      {partitions.map((p) => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <button
                    className="btn"
                    disabled={validateStoredMut.isPending}
                    onClick={() => validateStoredMut.mutate(
                      partitions.includes(validationPartition) ? validationPartition : partitions[0],
                    )}
                  >
                    {validateStoredMut.isPending ? "Validating…" : "Re-validate partition"}
                  </button>
                </div>
                {validateStoredMut.isError && <ErrorNote error={validateStoredMut.error} />}
                {validateStoredMut.data?.valid && (
                  <div className="success-note">
                    Partition <code>{validateStoredMut.data.source.replace("partition:", "")}</code> matches the registered schema.
                  </div>
                )}

                <h3 style={{ marginTop: 16 }}>Partitions</h3>
                <Table<string>
                  columns={["Partition", "Rows", "Ingested at", "Fingerprint"]}
                  rows={partitions}
                  keyOf={(p) => p}
                  render={(p) => {
                    const info = metaQ.data!.partitions[p];
                    return [
                      <td key="p" className="mono">{p}</td>,
                      <td key="r">{fmtNum(info.rows)}</td>,
                      <td key="i" className="mono">{info.ingestion_timestamp}</td>,
                      <td key="f"><Badge tone="info">{(info.fingerprint ?? "").slice(0, 12)}…</Badge></td>,
                    ];
                  }}
                />

                {previewPartition && (
                  <>
                    <h3 style={{ marginTop: 16 }}>Preview — {previewPartition}</h3>
                    {previewQ.isLoading && <Loading />}
                    {previewQ.isError && <ErrorNote error={previewQ.error} />}
                    {previewQ.data && previewRows.length > 0 && (
                      <Table<Record<string, unknown>>
                        columns={previewCols}
                        rows={previewRows}
                        keyOf={(_, i) => i}
                        render={(r) =>
                          previewCols.map((c) => (
                            <td key={c} className="mono">
                              {String(r[c] ?? "")}
                            </td>
                          ))
                        }
                      />
                    )}
                  </>
                )}

                <div style={{ marginTop: 12 }}>
                  {partitions.map((p) => (
                    <button
                      key={p}
                      className={`btn${previewPartition === p ? " primary" : ""}`}
                      style={{ marginRight: 8 }}
                      onClick={() => setPreviewPartition(previewPartition === p ? null : p)}
                    >
                      Preview {p}
                    </button>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </Card>

      <SyntheticDataCard
        task={task}
        metadata={metaQ.data}
        partitions={partitions}
        onIgnoredChange={() => {
          qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
          qc.invalidateQueries({ queryKey: ["features"] });
          setDataRevision((revision) => revision + 1);
        }}
      />
      <TargetStatsCard task={task} enabled={partitions.length > 0} revision={dataRevision} />
    </div>
  );
}

function SyntheticDataCard({
  task,
  metadata,
  partitions,
  onIgnoredChange,
}: {
  task: string;
  metadata?: DatasetMetadata | null;
  partitions: string[];
  onIgnoredChange: () => void;
}) {
  const qc = useQueryClient();
  const columns = metadata?.dataset_columns ?? [];
  const ignored = metadata?.ignored_columns ?? [];
  const [num, setNum] = useState("100");
  const [partition, setPartition] = useState("train");
  // The select is populated from the registered partitions, so a hard-coded
  // initial "train" could show the first option while still *sending* "train" to
  // the API — a 404 on any task whose partition is named differently.
  const sourcePartition = partitions.includes(partition) ? partition : (partitions[0] ?? "");
  const [distribution, setDistribution] = useState<"marginal" | "joint">("marginal");
  const [types, setTypes] = useState<Record<string, string>>({});
  const typesTaskRef = useRef<string | null>(null);

  useEffect(() => {
    if (!metadata) return;
    // A different task keeps no stale per-column choices: the same column name
    // can be numeric here and text there, and a stale "int" override was a
    // real failure source for text-binary columns.
    const sameTask = typesTaskRef.current === task;
    typesTaskRef.current = task;
    const inferred = Object.fromEntries(columns.map((column) => {
      const schema = metadata.dataset_schema?.[column];
      const dtype = schema && typeof schema === "object" && "dtype" in schema
        && typeof schema.dtype === "string"
        ? schema.dtype.toLowerCase()
        : "";
      const deduced = metadata.dataset_deduced_types?.[column];
      // SynthData's bin/int/real types compare values numerically, so a text
      // column that merely has two distinct values (Gender = Female/Male) must
      // default to "cat", not "bin".
      const numericDtype = /int|float|double|decimal|bool/.test(dtype);
      const deducedType: Record<string, string> = {
        float64: "real",
        int64: "int",
        datetime64: "date",
        binary: numericDtype ? "bin" : "cat",
        label: "cat",
        text: "cat",
        categorical: "cat",
        other: "cat",
        obsolete: "cat",
      };
      const type = (deduced && deducedType[deduced])
        ?? (dtype.includes("datetime") ? "date" : /int/.test(dtype) ? "int" : /float|double|decimal/.test(dtype) ? "real" : "cat");
      return [column, type];
    }));
    setTypes((current) => (sameTask ? { ...inferred, ...current } : inferred));
  }, [metadata, columns, task]);

  const filesQ = useQuery({
    queryKey: ["synthetic-files", task],
    queryFn: () => listSynthetic(task),
    enabled: !!task && partitions.length > 0,
    retry: false,
  });
  const generateMut = useMutation({
    mutationFn: () => generateSynthetic(task, {
      num: Number(num),
      partition: sourcePartition || undefined,
      distribution_type: distribution,
      // Ignored columns are dropped server-side from the source frame; never
      // send a generation type for one (it would be an unknown column).
      type_overrides: Object.fromEntries(
        Object.entries(types).filter(([column]) => !ignored.includes(column)),
      ),
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["synthetic-files", task] }),
  });
  const ignoreMut = useMutation({
    mutationFn: (nextIgnored: string[]) => setIgnoredColumns(task, nextIgnored),
    onSuccess: () => onIgnoredChange(),
  });
  const downloadMut = useMutation({ mutationFn: (filename: string) => downloadSynthetic(task, filename) });
  const preview = generateMut.data?.preview ?? [];
  const previewColumns = generateMut.data?.columns ?? [];
  const rowCount = Number(num);
  const validNum = Number.isInteger(rowCount) && rowCount > 0;

  return (
    <Card title="Synthetic data" sub="Generate and download a sample based on a registered data partition.">
      {!partitions.length && <p className="muted">Register a dataset partition before generating synthetic data.</p>}
      {partitions.length > 0 && (
        <>
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
              <label>Rows</label>
              <input type="number" min={1} step={1} value={num} onChange={(e) => setNum(e.target.value)} />
            </div>
            <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 160 }}>
              <label>Source partition</label>
              <select value={sourcePartition} onChange={(e) => setPartition(e.target.value)}>
                {partitions.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
              <label>Distribution type</label>
              <div className="radio-row">
                <label className="checkbox-row">
                  <input type="radio" name="synthetic-distribution" checked={distribution === "marginal"} onChange={() => setDistribution("marginal")} />
                  Marginal
                </label>
                <label className="checkbox-row">
                  <input type="radio" name="synthetic-distribution" checked={distribution === "joint"} onChange={() => setDistribution("joint")} />
                  Joint
                </label>
              </div>
            </div>
          </div>

          {columns.length > 0 && (
            <>
              <h3 style={{ marginTop: 14 }}>Column generation types</h3>
              <Table<string>
                columns={["Column", "Type"]}
                rows={columns}
                keyOf={(column) => column}
                render={(column) => {
                  const isIgnored = ignored.includes(column);
                  return [
                    <td key="name" className="mono">
                      {isIgnored ? <span className="muted">{column} (ignored)</span> : column}
                    </td>,
                    <td key="type">
                      <select
                        aria-label={`Synthetic type for ${column}`}
                        value={isIgnored ? IGNORE_TYPE : types[column] ?? "cat"}
                        onChange={(event) => {
                          const value = event.target.value;
                          if (value === IGNORE_TYPE) {
                            ignoreMut.mutate([...ignored, column]);
                            return;
                          }
                          if (isIgnored) {
                            ignoreMut.mutate(ignored.filter((name) => name !== column));
                          }
                          setTypes((current) => ({ ...current, [column]: value }));
                        }}
                      >
                        {["bin", "int", "real", "cat", "date"].map((type) => (
                          <option key={type} value={type}>{type}</option>
                        ))}
                        <option value={IGNORE_TYPE}>ignore (exclude from experiments)</option>
                      </select>
                    </td>,
                  ];
                }}
              />
              <div className="hint" style={{ marginTop: 8 }}>
                <code>ignore</code> is a dataset-level choice: the column stays in the stored
                partition (so it can be restored at any time) but is dropped from experiments,
                pipeline optimization, baselines, sensitivity/correlation analysis, profiles,
                synthetic data and monitoring drift. Choose a real type to re-enable it.
              </div>
              {ignoreMut.isError && <ErrorNote error={ignoreMut.error} />}
            </>
          )}

          {generateMut.isError && <ErrorNote error={generateMut.error} />}
          {downloadMut.isError && <ErrorNote error={downloadMut.error} />}
          {generateMut.data && (
            <>
              <div className="success-note" style={{ marginTop: 12 }}>
                Generated {fmtNum(generateMut.data.rows)} rows at <code>{generateMut.data.path}</code>
              </div>
              {preview.length > 0 && (
                <Table<Record<string, unknown>>
                  columns={previewColumns}
                  rows={preview}
                  keyOf={(_, i) => i}
                  render={(row) => previewColumns.map((column) => (
                    <td key={column} className="mono">{String(row[column] ?? "")}</td>
                  ))}
                />
              )}
              <button
                className="btn"
                disabled={downloadMut.isPending}
                onClick={() => downloadMut.mutate(generateMut.data!.path.split("/").pop()!)}
              >
                Download generated CSV
              </button>
            </>
          )}

          <div className="row" style={{ marginTop: 12 }}>
            <button
              className="btn primary"
              disabled={!task || !validNum || !sourcePartition || generateMut.isPending}
              onClick={() => generateMut.mutate()}
            >
              {generateMut.isPending ? "Generating…" : "Generate synthetic data"}
            </button>
          </div>

          {filesQ.isLoading && <Loading />}
          {filesQ.isError && <ErrorNote error={filesQ.error} />}
          {(filesQ.data?.files.length ?? 0) > 0 && (
            <>
              <h3 style={{ marginTop: 16 }}>Previously generated files</h3>
              <Table
                columns={["File", "Rows", "Download"]}
                rows={filesQ.data!.files}
                keyOf={(entry) => entry.name}
                render={(entry) => [
                  <td key="name" className="mono">{entry.name}</td>,
                  <td key="rows">{fmtNum(entry.rows)}</td>,
                  <td key="download">
                    <button className="btn" onClick={() => downloadMut.mutate(entry.name)}>Download</button>
                  </td>,
                ]}
              />
            </>
          )}
        </>
      )}
    </Card>
  );
}

/** Name-based PII / sensitive-column scan returned by dataset registration (Phase 2.4). */
function SensitiveScanNote({ scan }: { scan?: SensitiveScanReport }) {
  if (!scan || (scan.pii_columns.length === 0 && scan.sensitive_columns.length === 0)) return null;
  return (
    <div className="row" style={{ alignItems: "center", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
      <span className="muted" style={{ fontSize: 12.5 }}>Sensitive-column scan:</span>
      {scan.pii_columns.map((c) => (
        <Badge key={`pii-${c}`} tone="warn">PII · {c}</Badge>
      ))}
      {scan.sensitive_columns.map((c) => (
        <Badge key={`sens-${c}`} tone="err">sensitive · {c}</Badge>
      ))}
      <span className="muted" style={{ fontSize: 12.5 }}>
        consider excluding these from the feature set before training.
      </span>
    </div>
  );
}

/** Descriptive statistics of the registered target column. */
function TargetStatsCard({ task, enabled, revision }: { task: string; enabled: boolean; revision: number }) {
  const statsQ = useQuery({
    queryKey: ["target-stats", task, revision],
    queryFn: async () => {
      try {
        return await getTargetStats(task);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task && enabled,
    retry: false,
  });

  const rows = statsQ.data
    ? ([
        ["count", statsQ.data.count],
        ["mean", statsQ.data.mean],
        ["std", statsQ.data.std],
        ["min", statsQ.data.min],
        ["25%", statsQ.data["25%"]],
        ["50%", statsQ.data["50%"]],
        ["75%", statsQ.data["75%"]],
        ["max", statsQ.data.max],
      ] as Array<[string, number | null]>)
    : [];

  return (
    <Card
      title="Target statistics"
      sub={
        statsQ.data
          ? `describe() of “${statsQ.data.target}” over the ${statsQ.data.partition} partition`
          : "Summary of the registered target column"
      }
    >
      {!enabled && <p className="muted">Register a dataset partition to see target statistics.</p>}
      {enabled && statsQ.isLoading && <Loading />}
      {enabled && statsQ.isError && <ErrorNote error={statsQ.error} />}
      {rows.length > 0 && (
        <Table<[string, number | null]>
          columns={["Statistic", "Value"]}
          rows={rows}
          keyOf={(row) => row[0]}
          render={(row) => [
            <td key="k" className="mono">{row[0]}</td>,
            <td key="v">{fmtNum(row[1])}</td>,
          ]}
        />
      )}
    </Card>
  );
}

/**
 * Temporal structure panel (plan phase P7.1).
 *
 * Declares the task's time geometry — opt-in, like the CLI/API path: the time
 * (and optional series-id) columns are excluded from every feature matrix, the
 * declaration gates gaps over `gap_tolerance`, and it unlocks lag features,
 * rolling-origin backtests and multi-step serving. The per-partition report is
 * the server's `validate_temporal_frame` output verbatim; a rejected save
 * surfaces the 422's report in the error note.
 */
function TemporalCard({
  task,
  revision,
  fallbackColumns,
}: {
  task: string;
  revision: number;
  fallbackColumns: string[];
}) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["dataset-temporal", task, revision],
    queryFn: async () => {
      try {
        return await getDatasetTemporal(task);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task,
    retry: false,
  });

  const EMPTY_DRAFT: TemporalSpecState = {
    time_column: "",
    series_id: null,
    freq: null,
    horizon: 1,
    order: "increasing",
    gap_tolerance: 0,
  };
  const [draft, setDraft] = useState<TemporalSpecState>(EMPTY_DRAFT);

  // Re-seed the form whenever the stored declaration changes (save, clear, or a
  // fresh dataset registration bumping `revision`).
  useEffect(() => {
    const spec = q.data?.spec;
    setDraft(
      spec
        ? {
            time_column: spec.time_column,
            series_id: spec.series_id ?? null,
            freq: spec.freq ?? null,
            horizon: spec.horizon ?? 1,
            order: spec.order ?? "increasing",
            gap_tolerance: spec.gap_tolerance ?? 0,
          }
        : EMPTY_DRAFT,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q.data]);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["dataset-temporal", task, revision] });
    // The declaration rewrites the ignored columns and constrains CV (D2).
    qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
    qc.invalidateQueries({ queryKey: ["dataset-cv", task, revision] });
    qc.invalidateQueries({ queryKey: ["features"] });
  };
  const saveMut = useMutation({
    mutationFn: (body: TemporalSpecState) => setDatasetTemporal(task, body),
    onSuccess: refresh,
  });
  const clearMut = useMutation({
    mutationFn: () => clearDatasetTemporal(task),
    onSuccess: refresh,
  });

  const columns = q.data?.columns?.length ? q.data.columns : fallbackColumns;
  const declared = q.data?.spec ?? null;
  const reports = Object.entries(q.data?.validation ?? {}) as Array<
    [string, TemporalValidationReport]
  >;
  const firstInferred = reports.find(([, report]) => report.inferred_freq)?.[1].inferred_freq ?? null;

  const save = () => {
    if (!task || !draft.time_column) return;
    const freq = (draft.freq ?? "").trim();
    saveMut.mutate({
      time_column: draft.time_column,
      series_id: draft.series_id || null,
      freq: freq ? freq : null,
      horizon: Math.max(1, Number(draft.horizon) || 1),
      order: draft.order ?? "increasing",
      gap_tolerance: Math.max(0, Number(draft.gap_tolerance) || 0),
    });
  };

  return (
    <Card
      title="Temporal structure (forecasting)"
      sub="Declare the time column (and optional series id) to enable lag features, rolling-origin backtests and multi-step forecasts. The time and series columns are excluded from every feature matrix."
    >
      {q.isLoading && task && <Loading />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data === null && (
        <EmptyState
          title="No dataset registered for this task yet"
          hint="Register a partition above; the temporal declaration applies to the stored dataset."
        />
      )}
      {q.data && (
        <>
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 160 }}>
              <label>Time column</label>
              <select
                value={draft.time_column}
                onChange={(e) => setDraft((d) => ({ ...d, time_column: e.target.value }))}
              >
                <option value="">— select —</option>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 160 }}>
              <label>Series id (optional)</label>
              <select
                value={draft.series_id ?? ""}
                onChange={(e) => setDraft((d) => ({ ...d, series_id: e.target.value || null }))}
              >
                <option value="">(single series)</option>
                {columns
                  .filter((column) => column !== draft.time_column)
                  .map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 120 }}>
              <label>Frequency</label>
              <input
                placeholder="e.g. D, h, 15min"
                value={draft.freq ?? ""}
                onChange={(e) => setDraft((d) => ({ ...d, freq: e.target.value }))}
              />
            </div>
            <div className="field fixed" style={{ flex: "0 1 120px", minWidth: 90 }}>
              <label>Horizon (steps)</label>
              <input
                type="number"
                min={1}
                step={1}
                value={String(draft.horizon ?? 1)}
                onChange={(e) => setDraft((d) => ({ ...d, horizon: Number(e.target.value) }))}
              />
            </div>
            <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
              <label>Order</label>
              <select
                value={draft.order ?? "increasing"}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, order: e.target.value as "increasing" | "decreasing" }))
                }
              >
                <option value="increasing">increasing</option>
                <option value="decreasing">decreasing</option>
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 130px", minWidth: 100 }}>
              <label>Gap tolerance</label>
              <input
                type="number"
                min={0}
                step={1}
                value={String(draft.gap_tolerance ?? 0)}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, gap_tolerance: Number(e.target.value) }))
                }
              />
            </div>
          </div>
          <p className="muted" style={{ marginBottom: 12 }}>
            Frequency is a pandas offset alias; leave it blank to accept irregular sampling
            (advisory warning only).{firstInferred ? (
              <>
                {" "}
                Inferred from the data: <code>{firstInferred}</code>
                {firstInferred !== draft.freq && (
                  <button
                    className="btn"
                    style={{ marginLeft: 8 }}
                    onClick={() => setDraft((d) => ({ ...d, freq: firstInferred }))}
                  >
                    Use inferred
                  </button>
                )}
              </>
            ) : null}
          </p>

          {declared && (
            <dl className="kv">
              <dt>Declared</dt>
              <dd className="mono">
                time={declared.time_column}
                {declared.series_id ? ` series=${declared.series_id}` : " (single series)"}
                {declared.freq ? ` freq=${declared.freq}` : " (no declared frequency)"} h=
                {declared.horizon ?? 1}
              </dd>
              <dt>Excluded columns</dt>
              <dd className="mono">{q.data.temporal_columns.join(", ") || "—"}</dd>
            </dl>
          )}

          {reports.length > 0 && (
            <>
              <div className="row" style={{ gap: 8, marginBottom: 8 }}>
                <Badge tone={declared?.freq ? "ok" : "muted"}>
                  {declared?.freq ? `regular @ ${declared.freq}` : "irregular allowed"}
                </Badge>
                <Badge tone={q.data.cv_temporal_safe === false ? "warn" : "info"}>
                  CV {q.data.cv_temporal_safe === false ? "not temporal-safe — pick an order-respecting splitter" : "temporal-safe"}
                </Badge>
              </div>
              <Table<[string, TemporalValidationReport]>
                columns={["Partition", "Rows", "Series", "Inferred freq", "Missing periods", "Status"]}
                rows={reports}
                keyOf={([partition]) => partition}
                render={([partition, report]) => [
                  <td key="p" className="mono">{partition}</td>,
                  <td key="r">{fmtNum(report.rows ?? null)}</td>,
                  <td key="s">{report.series_count ?? report.series?.length ?? "—"}</td>,
                  <td key="f" className="mono">{report.inferred_freq ?? "—"}</td>,
                  <td key="m">{report.missing_periods ?? "—"}</td>,
                  <td key="ok">
                    <Badge tone={report.ok ? "ok" : "err"}>{report.ok ? "valid" : "invalid"}</Badge>
                  </td>,
                ]}
              />
              {reports
                .filter(([, report]) => (report.warnings ?? []).length > 0)
                .map(([partition, report]) => (
                  <div key={partition} className="muted" style={{ marginTop: 8 }}>
                    <strong>{partition}</strong> warnings: {(report.warnings ?? []).join("; ")}
                  </div>
                ))}
            </>
          )}

          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          {saveMut.isSuccess && (
            <div className="success-note">
              Saved temporal declaration: time=<code>{saveMut.data.spec?.time_column}</code>
              {saveMut.data.spec?.freq ? `, freq=${saveMut.data.spec.freq}` : ""}.
            </div>
          )}
          {clearMut.isError && <ErrorNote error={clearMut.error} />}
          {clearMut.isSuccess && (
            <div className="success-note">Temporal declaration removed; columns re-enabled as features.</div>
          )}

          <button
            className="btn primary"
            disabled={!task || !draft.time_column || saveMut.isPending}
            onClick={save}
          >
            {saveMut.isPending ? "Validating…" : declared ? "Update temporal structure" : "Declare temporal structure"}
          </button>
          {declared && (
            <button
              className="btn"
              style={{ marginLeft: 8 }}
              disabled={clearMut.isPending}
              onClick={() => clearMut.mutate()}
            >
              {clearMut.isPending ? "Removing…" : "Remove declaration"}
            </button>
          )}
        </>
      )}
    </Card>
  );
}
