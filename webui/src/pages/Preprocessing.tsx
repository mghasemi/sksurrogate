import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  clipDataset,
  deduplicateDataset,
  getDatasetMetadata,
  getDuplicates,
  getEncodingPreview,
  getPreprocessing,
  imputeDataset,
} from "../api/client";
import type {
  ClipReport,
  DuplicateAnalysis,
  EncodingPreview,
  ImputationReport,
  PreprocessingColumn,
  PreprocessingOverview,
} from "../api/client";
import { Badge, Card, EmptyState, ErrorNote, Loading, Table, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

function pct(value: number | null | undefined): string {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function num(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function summaryLine(summary: PreprocessingColumn["numeric_summary"]): string {
  if (!summary) return "—";
  return `mean ${num(summary.mean)} · sd ${num(summary.std)} · q25 ${num(summary.q25)} · median ${num(summary.q50)} · q75 ${num(summary.q75)} · [${num(summary.min)}, ${num(summary.max)}]`;
}

function topLine(values: PreprocessingColumn["top_values"]): string {
  if (!values || values.length === 0) return "—";
  return values.map((entry) => `${entry.value} (${pct(entry.share)})`).join(", ");
}

function ReportView({ report }: { report: ImputationReport }) {
  const columns = report.columns;
  const numeric = Object.entries(report.numeric_shift);
  const categorical = Object.entries(report.categorical_shift);
  return (
    <>
      <dl className="kv">
        <dt>Rows</dt>
        <dd>
          {fmtNum(report.rows_before)} → {fmtNum(report.rows_after)}
          {report.rows_dropped > 0 && <span className="muted"> ({fmtNum(report.rows_dropped)} dropped)</span>}
        </dd>
        <dt>Cells imputed</dt>
        <dd>{fmtNum(report.cells_imputed)}</dd>
        <dt>Target missing</dt>
        <dd>
          {fmtNum(report.target_missing_before)} → {fmtNum(report.target_missing_after)}
          {report.target_missing_after > 0 && <span className="muted"> — target is never imputed</span>}
        </dd>
      </dl>
      <Table<string>
        columns={["Column", "Missing before", "Missing after"]}
        rows={columns}
        keyOf={(column) => column}
        render={(column) => [
          <td key="c" className="mono">{column}</td>,
          <td key="b">{fmtNum(report.missing_before[column] ?? 0)}</td>,
          <td key="a">{fmtNum(report.missing_after[column] ?? 0)}</td>,
        ]}
      />
      {numeric.length > 0 && (
        <>
          <h4 style={{ marginTop: 14 }}>Distribution shift (numeric)</h4>
          <Table<string>
            columns={["Column", "Mean before → after", "Std before → after"]}
            rows={numeric.map(([column]) => column)}
            keyOf={(column) => column}
            render={(column) => {
              const shift = report.numeric_shift[column];
              return [
                <td key="c" className="mono">{column}</td>,
                <td key="m">{num(shift.before?.mean)} → {num(shift.after?.mean)}</td>,
                <td key="s">{num(shift.before?.std)} → {num(shift.after?.std)}</td>,
              ];
            }}
          />
        </>
      )}
      {categorical.length > 0 && (
        <>
          <h4 style={{ marginTop: 14 }}>Category frequencies</h4>
          {categorical.map(([column, shift]) => (
            <p key={column} className="muted">
              <code>{column}</code>: before {topLine(shift.before)} → after {topLine(shift.after)}
            </p>
          ))}
        </>
      )}
    </>
  );
}

function TargetDistribution({ distribution }: { distribution: DuplicateAnalysis["target_before"] }) {
  if (distribution.kind === "categorical") return <>{topLine(distribution.values ?? null)}</>;
  return <>{summaryLine(distribution.summary ?? null)}</>;
}

function EncodingCard({ task, partition }: { task: string; partition: string }) {
  const previewQ = useQuery<EncodingPreview>({
    queryKey: ["encoding-preview", task, partition],
    queryFn: () => getEncodingPreview(task, partition),
    enabled: !!task && !!partition,
  });
  const data = previewQ.data;
  return (
    <Card
      title="Categorical encoding preview"
      sub="The exact feature matrix the model pipeline builds from this partition. Learned inside each training fold, not once on the whole dataset."
    >
      {previewQ.isLoading && <Loading />}
      {previewQ.isError && <ErrorNote error={previewQ.error} />}
      {data && (
        <>
          <dl className="kv">
            <dt>Input features</dt>
            <dd>{fmtNum(data.input_feature_count)}</dd>
            <dt>Encoded features</dt>
            <dd>
              {fmtNum(data.output_feature_count)}
              {data.sampled && <span className="muted"> · vocabulary learned on {fmtNum(data.sample_rows)} sampled rows</span>}
            </dd>
          </dl>
          <Table<(typeof data.per_column)[number]>
            columns={["Column", "Detected kind", "Encoding", "Levels", "Output features", "Encoded names"]}
            rows={data.per_column}
            keyOf={(entry) => entry.column}
            render={(entry) => [
              <td key="c" className="mono">
                {entry.column}{" "}
                {entry.high_cardinality && <Badge tone="warn">high cardinality</Badge>}
              </td>,
              <td key="k">{entry.kind}</td>,
              <td key="e">{entry.encoding}</td>,
              <td key="l">
                {fmtNum(entry.cardinality)}
                {entry.missing > 0 && <span className="muted"> · {fmtNum(entry.missing)} missing</span>}
              </td>,
              <td key="o">{fmtNum(entry.output_feature_count)}</td>,
              <td key="n" className="mono muted">
                {entry.output_columns.length ? entry.output_columns.join(", ") : "—"}
                {entry.output_columns_truncated > 0 && ` … (+${entry.output_columns_truncated})`}
              </td>,
            ]}
          />
          {data.preview.length > 0 && (
            <>
              <h4 style={{ marginTop: 14 }}>Encoded rows (first {data.preview.length})</h4>
              <Table<Record<string, unknown>>
                columns={data.shown_output_features.slice(0, 12)}
                rows={data.preview}
                keyOf={(_, index) => index}
                render={(row) =>
                  data.shown_output_features.slice(0, 12).map((name) => (
                    <td key={name} className="mono">
                      {row[name] === null || row[name] === undefined
                        ? "—"
                        : typeof row[name] === "number"
                          ? num(row[name] as number, 3)
                          : String(row[name])}
                    </td>
                  ))
                }
              />
              {data.output_feature_count > 12 && (
                <p className="hint">Showing 12 of {fmtNum(data.output_feature_count)} encoded columns.</p>
              )}
            </>
          )}
          <ul className="hint" style={{ marginTop: 10 }}>
            {data.notes.map((note) => <li key={note}>{note}</li>)}
          </ul>
        </>
      )}
    </Card>
  );
}

function DuplicatesCard({
  task,
  partition,
  qc,
}: {
  task: string;
  partition: string;
  qc: ReturnType<typeof useQueryClient>;
}) {
  const duplicatesQ = useQuery<DuplicateAnalysis>({
    queryKey: ["duplicates", task, partition],
    queryFn: () => getDuplicates(task, partition),
    enabled: !!task && !!partition,
  });
  const [keep, setKeep] = useState<"first" | "last">("first");
  const [outputName, setOutputName] = useState(`${partition}_deduplicated`);
  useEffect(() => setOutputName(`${partition}_deduplicated`), [partition]);

  const previewMut = useMutation({
    mutationFn: () => deduplicateDataset(task, { partition, keep, preview_only: true }),
  });
  const saveMut = useMutation({
    mutationFn: () => deduplicateDataset(task, { partition, keep, output_partition: outputName.trim() || null }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
      qc.invalidateQueries({ queryKey: ["duplicates", task, partition] });
    },
  });
  const report = saveMut.data ?? previewMut.data ?? duplicatesQ.data;
  const analysis = duplicatesQ.data;

  return (
    <Card
      title="Duplicate rows"
      sub="Exact duplicate feature rows, their target consistency and what removing them would change. Removal is never silent: read the warnings first."
    >
      {duplicatesQ.isLoading && <Loading />}
      {duplicatesQ.isError && <ErrorNote error={duplicatesQ.error} />}
      {analysis && (
        <>
          <dl className="kv">
            <dt>Duplicate rows</dt>
            <dd>
              {fmtNum(analysis.duplicate_rows)} in {fmtNum(analysis.duplicate_groups)} group(s) ·{" "}
              {fmtNum(analysis.removable_rows)} removable
            </dd>
            <dt>Full-row duplicates</dt>
            <dd>{fmtNum(analysis.full_duplicate_rows)}</dd>
            <dt>Conflicting labels</dt>
            <dd>
              {fmtNum(analysis.conflicting_label_groups)} group(s), {fmtNum(analysis.conflicting_label_rows)} rows
              {analysis.conflicting_label_groups > 0 && <Badge tone="err">label contradiction</Badge>}
            </dd>
            <dt>Target before → after</dt>
            <dd>
              <TargetDistribution distribution={analysis.target_before} /> →{" "}
              <TargetDistribution distribution={analysis.target_after} />
            </dd>
            <dt>Rows before → after</dt>
            <dd>
              {fmtNum(analysis.rows_before)} → {fmtNum(analysis.rows_after)} (
              {fmtNum(analysis.rows_removed)} removed, {pct(analysis.rows_removed_pct)})
            </dd>
          </dl>

          {analysis.warnings.length > 0 && (
            <div className="warn-note" style={{ marginTop: 8 }}>
              <strong>Implications of removing duplicates</strong>
              <ul style={{ margin: "6px 0 0 18px" }}>
                {analysis.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </div>
          )}

          {Object.keys(analysis.numeric_shift).length > 0 && (
            <>
              <h4 style={{ marginTop: 14 }}>Numeric feature shift</h4>
              <Table<string>
                columns={["Feature", "Mean before → after", "Relative shift"]}
                rows={Object.keys(analysis.numeric_shift)}
                keyOf={(column) => column}
                render={(column) => {
                  const shift = analysis.numeric_shift[column];
                  return [
                    <td key="c" className="mono">{column}</td>,
                    <td key="m">{num(shift.before?.mean)} → {num(shift.after?.mean)}</td>,
                    <td key="d">
                      {shift.mean_delta_pct === null
                        ? "—"
                        : `${shift.mean_delta_pct > 0 ? "+" : ""}${(shift.mean_delta_pct * 100).toFixed(2)}%`}
                    </td>,
                  ];
                }}
              />
            </>
          )}

          <div className="row top" style={{ marginTop: 12 }}>
            <div className="field fixed" style={{ flex: "0 1 180px", minWidth: 140 }}>
              <label>Keep</label>
              <select value={keep} onChange={(e) => setKeep(e.target.value as "first" | "last")}>
                <option value="first">first occurrence</option>
                <option value="last">last occurrence</option>
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
              <label>Output partition</label>
              <input value={outputName} onChange={(e) => setOutputName(e.target.value)} spellCheck={false} />
              <span className="hint">Written as a new partition; the source is never modified.</span>
            </div>
          </div>
          {previewMut.isError && <ErrorNote error={previewMut.error} />}
          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          <div className="row">
            <button className="btn" onClick={() => previewMut.mutate()} disabled={previewMut.isPending || saveMut.isPending}>
              {previewMut.isPending ? "Computing…" : "Preview removal"}
            </button>
            <button
              className="btn primary"
              onClick={() => saveMut.mutate()}
              disabled={previewMut.isPending || saveMut.isPending || analysis.rows_removed === 0}
            >
              {saveMut.isPending ? "Saving…" : "Save deduplicated partition"}
            </button>
          </div>
          {saveMut.data?.written && (
            <div className="success-note">
              Wrote partition <code>{saveMut.data.output_partition}</code>
              {saveMut.data.dataset_fingerprint && (
                <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
              )}
            </div>
          )}
          {report && report !== analysis && (
            <>
              <h4 style={{ marginTop: 14 }}>{report.written ? "Saved result" : "Preview"}</h4>
              <dl className="kv">
                <dt>Rows</dt>
                <dd>{fmtNum(report.rows_before)} → {fmtNum(report.rows_after)}</dd>
                <dt>Removed</dt>
                <dd>{fmtNum(report.rows_removed)} ({pct(report.rows_removed_pct)})</dd>
              </dl>
            </>
          )}
        </>
      )}
    </Card>
  );
}

function ClipCard({
  task,
  partition,
  overview,
  qc,
}: {
  task: string;
  partition: string;
  overview: PreprocessingOverview;
  qc: ReturnType<typeof useQueryClient>;
}) {
  const [method, setMethod] = useState(overview.clip_methods[0]?.name ?? "iqr");
  const activeMethod = overview.clip_methods.find((entry) => entry.name === method);
  const [values, setValues] = useState<Record<string, string>>({});
  const numericColumns = useMemo(
    () =>
      overview.missing.columns.filter(
        (column) => !column.is_target && !column.ignored && column.numeric_summary !== null,
      ),
    [overview],
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [outputName, setOutputName] = useState(`${partition}_clipped`);

  useEffect(() => {
    setSelected(numericColumns.map((column) => column.column));
  }, [numericColumns]);

  useEffect(() => {
    setOutputName(`${partition}_clipped`);
    setValues({});
  }, [partition, method]);

  const params = useMemo(() => {
    const result: Record<string, number> = {};
    for (const param of activeMethod?.params ?? []) {
      const raw = values[param.name];
      const parsed = raw === undefined || raw === "" ? Number(param.default ?? 0) : Number(raw);
      result[param.name] = parsed;
    }
    return result;
  }, [activeMethod, values]);

  const previewMut = useMutation({
    mutationFn: () =>
      clipDataset(task, { partition, method, columns: selected, params, preview_only: true }),
  });
  const saveMut = useMutation({
    mutationFn: () =>
      clipDataset(task, {
        partition,
        method,
        columns: selected,
        params,
        output_partition: outputName.trim() || null,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
    },
  });
  const report: ClipReport | null = saveMut.data ?? previewMut.data ?? null;

  return (
    <Card
      title="Outlier clipping / winsorizing"
      sub="Cap extreme numeric values at robust bounds. Clipping changes the tails, so the preview reports how many values move and how the mean and standard deviation shift."
    >
      {numericColumns.length === 0 ? (
        <p className="muted">No numeric feature columns to clip.</p>
      ) : (
        <>
          <div className="row top">
            <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
              <label>Method</label>
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {overview.clip_methods.map((entry) => (
                  <option key={entry.name} value={entry.name}>{entry.label}</option>
                ))}
              </select>
            </div>
            {(activeMethod?.params ?? []).map((param) => (
              <div key={param.name} className="field fixed" style={{ flex: "0 1 130px", minWidth: 105 }}>
                <label>{param.name}</label>
                <input
                  type="number"
                  min={param.min}
                  max={param.max}
                  step="any"
                  value={values[param.name] ?? String(param.default ?? "")}
                  onChange={(e) => setValues((current) => ({ ...current, [param.name]: e.target.value }))}
                />
                {param.help && <span className="hint">{param.help}</span>}
              </div>
            ))}
            <div className="field fixed" style={{ flex: "0 1 190px", minWidth: 160 }}>
              <label>Output partition</label>
              <input value={outputName} onChange={(e) => setOutputName(e.target.value)} spellCheck={false} />
            </div>
          </div>

          <h4 style={{ marginTop: 12 }}>Numeric columns</h4>
          <Table<PreprocessingColumn>
            columns={["", "Column", "Median", "IQR range", "Min / max"]}
            rows={numericColumns}
            keyOf={(column) => column.column}
            render={(column) => [
              <td key="sel">
                <input
                  type="checkbox"
                  aria-label={`Clip ${column.column}`}
                  checked={selected.includes(column.column)}
                  onChange={() =>
                    setSelected((current) =>
                      current.includes(column.column)
                        ? current.filter((name) => name !== column.column)
                        : [...current, column.column],
                    )
                  }
                />
              </td>,
              <td key="c" className="mono">{column.column}</td>,
              <td key="m">{num(column.numeric_summary?.q50)}</td>,
              <td key="q">
                {num(column.numeric_summary?.q25)} … {num(column.numeric_summary?.q75)}
              </td>,
              <td key="r">{num(column.numeric_summary?.min)} / {num(column.numeric_summary?.max)}</td>,
            ]}
          />

          {previewMut.isError && <ErrorNote error={previewMut.error} />}
          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          <div className="row">
            <button
              className="btn"
              onClick={() => previewMut.mutate()}
              disabled={selected.length === 0 || previewMut.isPending || saveMut.isPending}
            >
              {previewMut.isPending ? "Computing…" : "Preview clipping"}
            </button>
            <button
              className="btn primary"
              onClick={() => saveMut.mutate()}
              disabled={selected.length === 0 || previewMut.isPending || saveMut.isPending}
            >
              {saveMut.isPending ? "Saving…" : "Save clipped partition"}
            </button>
          </div>
          {saveMut.data?.written && (
            <div className="success-note">
              Wrote partition <code>{saveMut.data.output_partition}</code>
              {saveMut.data.dataset_fingerprint && (
                <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
              )}
            </div>
          )}
          {report && (
            <>
              <h4 style={{ marginTop: 14 }}>
                {report.written ? "Saved result" : "Preview"} · {report.method} · {fmtNum(report.cells_clipped)} values clipped
              </h4>
              <Table<string>
                columns={["Column", "Bounds", "Clipped low / high", "Share", "Mean before → after", "Std before → after"]}
                rows={report.columns}
                keyOf={(column) => column}
                render={(column) => {
                  const stats = report.columns_report[column];
                  return [
                    <td key="c" className="mono">{column}</td>,
                    <td key="b">{num(stats.lower)} … {num(stats.upper)}</td>,
                    <td key="n">{fmtNum(stats.clipped_low)} / {fmtNum(stats.clipped_high)}</td>,
                    <td key="s">{pct(stats.clipped_pct)}</td>,
                    <td key="m">{num(stats.before?.mean)} → {num(stats.after?.mean)}</td>,
                    <td key="d">{num(stats.before?.std)} → {num(stats.after?.std)}</td>,
                  ];
                }}
              />
            </>
          )}
        </>
      )}
    </Card>
  );
}

export default function PreprocessingPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const metaQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: async () => {
      try {
        return await getDatasetMetadata(task);
      } catch {
        return null;
      }
    },
    enabled: !!task,
    retry: false,
  });
  const partitions = useMemo(() => Object.keys(metaQ.data?.partitions ?? {}), [metaQ.data]);

  const [partition, setPartition] = useState("train");
  useEffect(() => {
    if (partitions.length && !partitions.includes(partition)) setPartition(partitions[0]);
  }, [partitions, partition]);

  const prepQ = useQuery<PreprocessingOverview>({
    queryKey: ["preprocessing", task, partition],
    queryFn: () => getPreprocessing(task, partition),
    enabled: !!task && !!partition && partitions.includes(partition),
  });
  const data = prepQ.data;

  const [method, setMethod] = useState("most_frequent");
  const [selected, setSelected] = useState<string[]>([]);
  const [constantValue, setConstantValue] = useState("");
  const [nNeighbors, setNNeighbors] = useState("5");
  const [maxIter, setMaxIter] = useState("10");
  const [outputName, setOutputName] = useState("train_imputed");
  const [dropTarget, setDropTarget] = useState(false);

  // Default selection: every active (non-target, non-ignored) column with
  // missing values. The user can narrow it; ignored columns are not offered.
  useEffect(() => {
    if (!data) return;
    setSelected(
      data.missing.columns
        .filter((column) => !column.is_target && !column.ignored && column.missing > 0)
        .map((column) => column.column),
    );
  }, [data]);

  useEffect(() => {
    setOutputName(`${partition}_imputed`);
    setDropTarget(false);
  }, [partition]);

  const selectedMethod = data?.methods.find((entry) => entry.name === method);
  const requiresNumeric = selectedMethod?.numeric_only ?? false;
  const numericSelected =
    !requiresNumeric ||
    selected.every((name) => {
      const column = data?.missing.columns.find((entry) => entry.column === name);
      return column ? column.numeric_summary !== null : false;
    });

  const params = useMemo(() => {
    if (method === "constant") return { constant_value: constantValue };
    if (method === "knn") return { n_neighbors: Number(nNeighbors) };
    if (method === "iterative") return { max_iter: Number(maxIter) };
    return {};
  }, [method, constantValue, nNeighbors, maxIter]);

  const body = (previewOnly: boolean) => ({
    partition,
    method,
    columns: selected,
    params,
    output_partition: outputName.trim() || null,
    preview_only: previewOnly,
    drop_missing_target: dropTarget,
  });

  const previewMut = useMutation({
    mutationFn: () => imputeDataset(task, body(true)),
  });
  const saveMut = useMutation({
    mutationFn: () => imputeDataset(task, body(false)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
    },
  });
  const report = saveMut.data ?? previewMut.data ?? null;
  const canSubmit =
    !!data && selected.length > 0 && !!method && (method !== "constant" || constantValue.trim() !== "");

  const toggleColumn = (column: string) => {
    setSelected((current) =>
      current.includes(column) ? current.filter((name) => name !== column) : [...current, column],
    );
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Data Preprocessing</h1>
        <span className="desc">
          Missing-data profile and materialized imputation for task “{task || "…"}”. The source
          partition is never modified; imputed data is saved as a new partition.
        </span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}
      {task && metaQ.data === null && (
        <EmptyState
          title="No dataset registered for this task yet"
          hint="Register a CSV partition on the Datasets page first."
        />
      )}

      {task && partitions.length > 0 && (
        <Card title="Partition">
          <div className="row fixed">
            <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
              <label>Stored partition</label>
              <select value={partition} onChange={(e) => setPartition(e.target.value)}>
                {partitions.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </div>
            {data && (
              <span className="muted" style={{ alignSelf: "flex-end", marginBottom: 8 }}>
                target <code>{data.target}</code> · {fmtNum(data.rows)} rows · {fmtNum(data.columns.length)} columns
              </span>
            )}
          </div>
        </Card>
      )}

      {prepQ.isLoading && <Loading />}
      {prepQ.isError && <ErrorNote error={prepQ.error} />}

      {data && (
        <>
          <Card title="Missing data overview" sub="Per-column counts and the share of complete rows. Ignored columns are excluded from modeling but shown here.">
            <dl className="kv">
              <dt>Rows</dt>
              <dd>{fmtNum(data.missing.rows)}</dd>
              <dt>Missing cells</dt>
              <dd>
                {fmtNum(data.missing.missing_cells)} of {fmtNum(data.missing.total_cells)} ({pct(data.missing.missing_pct)})
              </dd>
              <dt>Complete rows</dt>
              <dd>
                {fmtNum(data.missing.complete_rows)} ({pct(data.missing.complete_pct)}) ·{" "}
                {fmtNum(data.missing.rows_with_any_missing)} rows ({pct(data.missing.rows_with_any_missing_pct)}) have a missing feature
              </dd>
              <dt>Target missing</dt>
              <dd>{fmtNum(data.missing.target_missing)}</dd>
            </dl>

            <h4 style={{ marginTop: 14 }}>Columns and implied types</h4>
            <Table<PreprocessingColumn>
              columns={["Column", "Detected type", "Implied type", "Missing", "Unique", "Distribution"]}
              rows={data.missing.columns}
              keyOf={(column) => column.column}
              render={(column) => [
                <td key="c" className="mono">
                  {column.column}{" "}
                  {column.is_target && <Badge tone="info">target</Badge>}{" "}
                  {column.ignored && <Badge tone="warn">ignored</Badge>}{" "}
                  {column.all_missing && <Badge tone="err">all missing</Badge>}
                  {!column.all_missing && column.constant && <Badge tone="warn">constant</Badge>}
                </td>,
                <td key="d" className="mono">{column.dtype}</td>,
                <td key="i">{column.deduced_type ?? "—"}</td>,
                <td key="m">
                  {fmtNum(column.missing)} ({pct(column.missing_pct)})
                </td>,
                <td key="u">{fmtNum(column.unique)}</td>,
                <td key="dist" className="muted">
                  {column.numeric_summary ? summaryLine(column.numeric_summary) : topLine(column.top_values)}
                </td>,
              ]}
            />
            {data.profile_features_truncated > 0 && (
              <p className="hint">
                {fmtNum(data.profile_features_truncated)} further feature(s) are not profiled here (limit 100);
                per-column missingness above still covers every column.
              </p>
            )}
          </Card>

          <Card title="Data quality" sub="Computed on a bounded row sample; duplicates and constants are descriptive flags, not automatic exclusions.">
            {data.quality ? (
              <dl className="kv">
                <dt>Duplicate rows</dt>
                <dd>
                  {fmtNum(data.quality.duplicate_rows.count)} ({pct(data.quality.duplicate_rows.rate)})
                </dd>
                <dt>Constant features</dt>
                <dd className="mono">
                  {data.quality.constant_features.length ? data.quality.constant_features.join(", ") : "—"}
                </dd>
                <dt>All-missing features</dt>
                <dd className="mono">
                  {data.quality.all_missing_features.length ? data.quality.all_missing_features.join(", ") : "—"}
                </dd>
                <dt>Duplicate features</dt>
                <dd className="mono">
                  {data.quality.duplicate_features.length
                    ? data.quality.duplicate_features
                        .map((entry) => `${entry.feature} = ${entry.duplicate_of}`)
                        .join(", ")
                    : "—"}
                </dd>
                {data.target_profile && (
                  <>
                    <dt>Target distribution</dt>
                    <dd className="muted">
                      {String(
                        (data.target_profile as Record<string, unknown>).kind === "numeric"
                          ? summaryLine((data.target_profile as Record<string, unknown>)
                              .distribution as PreprocessingColumn["numeric_summary"])
                          : topLine((data.target_profile as Record<string, unknown>)
                              .distribution as PreprocessingColumn["top_values"]),
                      )}
                    </dd>
                  </>
                )}
              </dl>
            ) : (
              <p className="muted">No active feature column to profile (all features are ignored or the dataset has only the target).</p>
            )}
            {data.missingness_patterns && data.missingness_patterns.patterns.length > 0 && (
              <>
                <h4 style={{ marginTop: 14 }}>Missingness patterns</h4>
                <Table<(typeof data.missingness_patterns.patterns)[number]>
                  columns={["Missing columns", "Rows", "Share"]}
                  rows={data.missingness_patterns.patterns}
                  keyOf={(pattern, index) => `${index}-${pattern.missing_features.join(",")}`}
                  render={(pattern) => [
                    <td key="f" className="mono">{pattern.missing_features.join(", ")}</td>,
                    <td key="c">{fmtNum(pattern.count)}</td>,
                    <td key="p">{pct(pattern.proportion)}</td>,
                  ]}
                />
              </>
            )}
          </Card>

          <DuplicatesCard task={task} partition={partition} qc={qc} />

          <EncodingCard task={task} partition={partition} />

          <Card title="Dataset preview" sub={`First ${fmtNum(data.preview.length)} rows of the stored partition. “—” is a missing value.`}>
            <Table<Record<string, unknown>>
              columns={data.columns}
              rows={data.preview}
              keyOf={(_, index) => index}
              render={(row) =>
                data.columns.map((column) => (
                  <td key={column} className="mono">
                    {row[column] === null || row[column] === undefined ? "—" : String(row[column])}
                  </td>
                ))
              }
            />
          </Card>

          <Card
            title="Impute missing data"
            sub="Preview the result first; saving writes an imputed copy as a NEW partition and records its provenance. The target is never imputed."
          >
            <div className="warn-note">
              Materialized imputation changes the data. Experiments and cross-validation still use
              fold-local imputation on the original partition unless you deliberately select the
              derived partition later.
            </div>
            <div className="row top">
              <div className="field fixed" style={{ flex: "0 1 300px", minWidth: 240 }}>
                <label>Method</label>
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                  {data.methods.map((entry) => (
                    <option key={entry.name} value={entry.name}>{entry.label}</option>
                  ))}
                </select>
                {requiresNumeric && (
                  <span className="hint">This method accepts numeric columns only.</span>
                )}
              </div>
              {method === "constant" && (
                <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 160 }}>
                  <label>Constant value</label>
                  <input value={constantValue} onChange={(e) => setConstantValue(e.target.value)} spellCheck={false} />
                </div>
              )}
              {method === "knn" && (
                <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 110 }}>
                  <label>Neighbours</label>
                  <input type="number" min={1} max={50} value={nNeighbors} onChange={(e) => setNNeighbors(e.target.value)} />
                </div>
              )}
              {method === "iterative" && (
                <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 110 }}>
                  <label>Max iterations</label>
                  <input type="number" min={1} max={100} value={maxIter} onChange={(e) => setMaxIter(e.target.value)} />
                </div>
              )}
              <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
                <label>Output partition</label>
                <input
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  spellCheck={false}
                  disabled={saveMut.isPending}
                />
                <span className="hint">Must be a new partition name; the source is never overwritten.</span>
              </div>
            </div>

            <label className="checkbox-row" style={{ marginTop: 8 }}>
              <input type="checkbox" checked={dropTarget} onChange={(e) => setDropTarget(e.target.checked)} />
              Drop rows whose target value is missing
            </label>

            <h4 style={{ marginTop: 14 }}>Columns to impute</h4>
            {data.missing.columns.filter((column) => !column.is_target && !column.ignored).length === 0 ? (
              <p className="muted">No active feature columns to impute — ignored columns are not imputed.</p>
            ) : (
              <Table<PreprocessingColumn>
                columns={["", "Column", "Missing", "Type"]}
                rows={data.missing.columns.filter((column) => !column.is_target && !column.ignored)}
                keyOf={(column) => column.column}
                render={(column) => [
                  <td key="sel">
                    <input
                      type="checkbox"
                      aria-label={`Impute ${column.column}`}
                      checked={selected.includes(column.column)}
                      onChange={() => toggleColumn(column.column)}
                    />
                  </td>,
                  <td key="c" className="mono">{column.column}</td>,
                  <td key="m">{fmtNum(column.missing)} ({pct(column.missing_pct)})</td>,
                  <td key="t">{column.numeric_summary ? "numeric" : "categorical/text"}</td>,
                ]}
              />
            )}

            {!numericSelected && (
              <p className="warn-note">
                The selected method is numeric-only but the selection includes categorical/text columns.
                Deselect those or choose most-frequent/constant.
              </p>
            )}
            {previewMut.isError && <ErrorNote error={previewMut.error} />}
            {saveMut.isError && <ErrorNote error={saveMut.error} />}
            <div className="row">
              <button
                className="btn"
                disabled={!canSubmit || previewMut.isPending || saveMut.isPending}
                onClick={() => previewMut.mutate()}
              >
                {previewMut.isPending ? "Computing…" : "Preview imputation"}
              </button>
              <button
                className="btn primary"
                disabled={!canSubmit || saveMut.isPending || previewMut.isPending}
                onClick={() => saveMut.mutate()}
              >
                {saveMut.isPending ? "Saving…" : "Save as new partition"}
              </button>
            </div>
            {saveMut.data && (
              <div className="success-note">
                Wrote partition <code>{saveMut.data.output_partition}</code>
                {saveMut.data.dataset_fingerprint && (
                  <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
                )}
              </div>
            )}
            {report && (
              <>
                <h4 style={{ marginTop: 16 }}>
                  {report.written ? "Saved result" : "Preview"} · {report.method}
                </h4>
                <ReportView report={report} />
              </>
            )}
          </Card>

          <ClipCard task={task} partition={partition} overview={data} qc={qc} />
        </>
      )}
    </div>
  );
}
