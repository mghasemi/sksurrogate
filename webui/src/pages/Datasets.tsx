import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";

import {
  downloadSynthetic,
  generateSynthetic,
  getDatasetCV,
  getDatasetMetadata,
  getTargetStats,
  inspectDataset,
  listSynthetic,
  previewDataset,
  registerDataset,
  setDatasetCV,
  validateDataset,
} from "../api/client";
import type {
  CVParamDef,
  CVSpec,
  DatasetInspectResponse,
  DatasetMetadata,
  DatasetPreview,
  RegisterDatasetResponse,
  SensitiveScanReport,
} from "../api/client";
import { Card, ErrorNote, Loading, Table, Badge, fmtNum } from "../components/ui";
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

  const [target, setTarget] = useState("");
  const [partition, setPartition] = useState("train");
  const [file, setFile] = useState<File | null>(null);
  const [typeOverrides, setTypeOverrides] = useState<Record<string, string>>({});
  const [binarizeLabel, setBinarizeLabel] = useState(false);
  const [validationPartition, setValidationPartition] = useState("train");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const metaQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
  });

  const [previewPartition, setPreviewPartition] = useState<string | null>(null);
  const previewQ = useQuery<DatasetPreview>({
    queryKey: ["dataset-preview", task, previewPartition],
    queryFn: () => previewDataset(task, previewPartition!),
    enabled: !!task && !!previewPartition,
  });

  const inspectMut = useMutation({
    mutationFn: (upload: File) => inspectDataset(task, upload),
    onSuccess: (data: DatasetInspectResponse) => {
      setTypeOverrides(data.deduced_types);
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

  const registerMut = useMutation({
    mutationFn: (vars: { target: string; partition: string; file: File }) =>
      registerDataset(task, vars.target, vars.partition, vars.file, {
        type_overrides: typeOverrides,
        binarize_label: binarizeLabel,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["target-stats", task] });
      setFile(null);
      setTypeOverrides({});
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
    setBinarizeLabel(false);
  };

  const cvQ = useQuery({
    queryKey: ["dataset-cv", task],
    queryFn: () => getDatasetCV(task),
    enabled: !!task,
  });
  const [cvChoice, setCvChoice] = useState<string>("");
  // Editable constructor parameters of the selected splitter (raw input strings).
  const [cvParamValues, setCvParamValues] = useState<Record<string, string>>({});
  const setCvMut = useMutation({
    mutationFn: (vars: { spec: CVSpec; params?: Record<string, unknown> }) =>
      setDatasetCV(task, vars.spec, vars.params),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dataset-cv", task] }),
  });

  const cvOptions = cvQ.data?.options ?? [];
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

      {!task && (
        <div className="error-note">Pick a task name in the top bar to work with datasets.</div>
      )}

      <Card title="Register a partition" sub="Upload, review the inferred schema, then commit the CSV as a task partition.">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 180 }}>
            <label>Target column</label>
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
                return [
                  <td key="name" className="mono">{column}</td>,
                  <td key="type">
                    <select
                      aria-label={`Type for ${column}`}
                      value={typeOverrides[column] ?? inspected.deduced_types[column] ?? "other"}
                      onChange={(e) => setTypeOverrides((current) => ({ ...current, [column]: e.target.value }))}
                    >
                      {DATASET_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
                    </select>
                  </td>,
                  <td key="flags">
                    {isPii && <Badge tone="warn">PII</Badge>}
                    {isSensitive && <Badge tone="warn">sensitive</Badge>}
                    {!isPii && !isSensitive && <span className="muted">—</span>}
                  </td>,
                ];
              }}
            />

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
        {cvQ.data && (
          <>
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

      <Card title="Registered metadata" sub={metaQ.data?.dataset_fingerprint ?? undefined}>
        {metaQ.isLoading && task && <Loading />}
        {metaQ.isError && <ErrorNote error={metaQ.error} />}
        {metaQ.data && (
          <>
            <dl className="kv">
              <dt>Target</dt>
              <dd>{metaQ.data.target_name ?? "—"}</dd>
              <dt>Fingerprint</dt>
              <dd className="mono">{metaQ.data.dataset_fingerprint ?? "—"}</dd>
              <dt>Feature count</dt>
              <dd>{fmtNum(metaQ.data.dataset_feature_count)}</dd>
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

      <SyntheticDataCard task={task} metadata={metaQ.data} partitions={partitions} />
      <TargetStatsCard task={task} enabled={partitions.length > 0} />
    </div>
  );
}

function SyntheticDataCard({
  task,
  metadata,
  partitions,
}: {
  task: string;
  metadata?: DatasetMetadata;
  partitions: string[];
}) {
  const qc = useQueryClient();
  const columns = metadata?.dataset_columns ?? [];
  const [num, setNum] = useState("100");
  const [partition, setPartition] = useState("train");
  const [distribution, setDistribution] = useState<"marginal" | "joint">("marginal");
  const [types, setTypes] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!metadata) return;
    const inferred = Object.fromEntries(columns.map((column) => {
      const schema = metadata.dataset_schema?.[column];
      const dtype = schema && typeof schema === "object" && "dtype" in schema
        && typeof schema.dtype === "string"
        ? schema.dtype.toLowerCase()
        : "";
      const deduced = metadata.dataset_deduced_types?.[column];
      const deducedType: Record<string, string> = {
        float64: "real",
        int64: "int",
        datetime64: "date",
        binary: "bin",
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
    setTypes((current) => ({ ...inferred, ...current }));
  }, [metadata, columns]);

  const filesQ = useQuery({
    queryKey: ["synthetic-files", task],
    queryFn: () => listSynthetic(task),
    enabled: !!task && partitions.length > 0,
    retry: false,
  });
  const generateMut = useMutation({
    mutationFn: () => generateSynthetic(task, {
      num: Number(num),
      partition,
      distribution_type: distribution,
      type_overrides: types,
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["synthetic-files", task] }),
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
              <select value={partition} onChange={(e) => setPartition(e.target.value)}>
                {partitions.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
              <label>Distribution type</label>
              <div className="row" style={{ gap: 12 }}>
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
                render={(column) => [
                  <td key="name" className="mono">{column}</td>,
                  <td key="type">
                    <select
                      aria-label={`Synthetic type for ${column}`}
                      value={types[column] ?? "cat"}
                      onChange={(e) => setTypes((current) => ({ ...current, [column]: e.target.value }))}
                    >
                      {["bin", "int", "real", "cat", "date"].map((type) => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </td>,
                ]}
              />
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
              disabled={!task || !validNum || generateMut.isPending}
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
function TargetStatsCard({ task, enabled }: { task: string; enabled: boolean }) {
  const statsQ = useQuery({
    queryKey: ["target-stats", task],
    queryFn: () => getTargetStats(task),
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
