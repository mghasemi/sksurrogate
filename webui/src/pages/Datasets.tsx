import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";

import {
  getDatasetCV,
  getDatasetMetadata,
  getTargetStats,
  previewDataset,
  registerDataset,
  setDatasetCV,
} from "../api/client";
import type { CVParamDef, CVSpec, DatasetPreview, RegisterDatasetResponse } from "../api/client";
import { Card, ErrorNote, Loading, Table, Badge, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

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

  const registerMut = useMutation({
    mutationFn: (vars: { target: string; partition: string; file: File }) =>
      registerDataset(task, vars.target, vars.partition, vars.file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      setFile(null);
      if (fileRef.current) fileRef.current.value = "";
    },
  });

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

      <Card title="Register a partition" sub="Server infers dtypes and stores the CSV under var/sksurrogate-api/datasets/{task}/">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 180 }}>
            <label>Target column</label>
            <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. target" spellCheck={false} />
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
            if (f) setFile(f);
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
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        {registerMut.isError && <ErrorNote error={registerMut.error} />}
        {registerMut.isSuccess && (
          <div className="success-note">
            Registered {fmtNum((registerMut.data as RegisterDatasetResponse).rows)} rows → fingerprint{" "}
            <code>{(registerMut.data as RegisterDatasetResponse).dataset_fingerprint ?? "n/a"}</code>
          </div>
        )}

        <button
          className="btn primary"
          disabled={!task || !target || !partition || !file || registerMut.isPending}
          onClick={() => file && registerMut.mutate({ target, partition, file })}
        >
          {registerMut.isPending ? "Registering…" : "Register dataset"}
        </button>
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

      <TargetStatsCard task={task} enabled={partitions.length > 0} />
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
