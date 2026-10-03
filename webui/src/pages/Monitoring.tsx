import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import {
  checkDelayedLabels,
  checkDrift,
  checkFairness,
  checkPredictionDrift,
  checkSubgroupPerformance,
  getDatasetMetadata,
  listBundles,
  monitorSummary,
  scanSensitiveFeatures,
} from "../api/client";
import type {
  DriftReport,
  FairnessReport,
  MonitorSummary,
  PredictionSource,
  SubgroupMetric,
  SubgroupPerformanceReport,
} from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, Loading, Stat, Table, fmtNum, fmtPct } from "../components/ui";
import { useTask } from "../lib/task-context";

const CHART_COLORS = ["#4f8cff", "#7c5cff", "#34d399", "#fbbf24", "#f87171"];

export default function MonitoringPage() {
  const { task, setTask } = useTask();
  const [searchParams, setSearchParams] = useSearchParams();
  const taskParam = searchParams.get("task");
  const versionParam = searchParams.get("version") ?? "";

  useEffect(() => {
    if (!taskParam) return;
    if (taskParam !== task) {
      setTask(taskParam);
      return;
    }
    const params = new URLSearchParams(searchParams);
    params.delete("task");
    setSearchParams(params, { replace: true });
  }, [searchParams, setSearchParams, setTask, task, taskParam]);

  const bundlesQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });
  const datasetQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
    retry: false,
  });

  const versions = useMemo(() => bundlesQ.data?.bundles ?? [], [bundlesQ.data]);
  const partitions = useMemo(() => Object.keys(datasetQ.data?.partitions ?? {}), [datasetQ.data]);
  const columns = useMemo(() => datasetQ.data?.dataset_columns ?? [], [datasetQ.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Monitoring</h1>
        <span className="desc">Drift, fairness and serving health for task “{task || "…"}”.</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {task && (
        <>
          <SummaryCard task={task} versions={versions} initialVersion={versionParam} />
          <DriftCard task={task} partitions={partitions} versions={versions} />
          <PredictionDriftCard task={task} partitions={partitions} columns={columns} versions={versions} />
          <SubgroupPerformanceCard task={task} partitions={partitions} columns={columns} versions={versions} />
          <SensitiveScanCard task={task} partitions={partitions} columns={columns} />
          <FairnessCard task={task} />
          <DelayedLabelCard task={task} versions={versions} />
        </>
      )}
    </div>
  );
}

function SummaryCard({ task, versions, initialVersion }: { task: string; versions: string[]; initialVersion: string }) {
  const [version, setVersion] = useState(initialVersion);
  const q = useQuery<MonitorSummary>({
    queryKey: ["monitor-summary", task, version],
    queryFn: () => monitorSummary(task, version),
    enabled: !!task && !!version,
    retry: false,
  });

  return (
    <Card title="Serving summary" sub="Aggregated latency / throughput / error rate from logged inference calls.">
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
      </div>

      {version && <LineageRail task={task} modelVersion={version} />}

      {q.isLoading && version && <Loading />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && (
        <div className="grid cols-4">
          <Stat label="Requests" value={fmtNum(q.data.requests)} />
          <Stat label="Rows served" value={fmtNum(q.data.rows)} />
          <Stat label="Error rate" value={fmtPct(q.data.error_rate)} />
          <Stat label="Mean latency" value={`${fmtNum(q.data.mean_latency_ms)} ms`} sub={`${fmtNum(q.data.throughput_rows_per_second)} rows/s`} />
        </div>
      )}
    </Card>
  );
}

function DriftCard({ task, partitions, versions }: { task: string; partitions: string[]; versions: string[] }) {
  const [reference, setReference] = useState("");
  const [current, setCurrent] = useState("");
  const [modelVersion, setModelVersion] = useState("");
  const [psiThreshold, setPsiThreshold] = useState("0.2");
  const [categoricalThreshold, setCategoricalThreshold] = useState("0.2");
  const [missingnessThreshold, setMissingnessThreshold] = useState("0.1");
  const [rangeThreshold, setRangeThreshold] = useState("0.1");

  const mut = useMutation({
    mutationFn: () =>
      checkDrift(task, {
        reference_partition: reference,
        current_partition: current,
        model_version: modelVersion || null,
        psi_threshold: Number(psiThreshold) || 0.2,
        categorical_threshold: Number(categoricalThreshold) || 0.2,
        missingness_threshold: Number(missingnessThreshold) || 0.1,
        range_threshold: Number(rangeThreshold) || 0.1,
      }),
  });

  const report = mut.data as DriftReport | undefined;
  const numericEntries = report ? Object.entries(report.drift.numeric) : [];
  const maxPsi = Math.max(0.1, ...numericEntries.map(([, v]) => (v.value ?? 0)));

  return (
    <Card title="Drift check" sub="Compares a reference partition against a current one: PSI / categorical / range / missingness.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
          <label>Reference partition</label>
          <select value={reference} onChange={(e) => setReference(e.target.value)}>
            <option value="">— select —</option>
            {partitions.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
          <label>Current partition</label>
          <select value={current} onChange={(e) => setCurrent(e.target.value)}>
            <option value="">— select —</option>
            {partitions.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
          <label>Model version (optional)</label>
          <select value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>
            <option value="">— none —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
          <label>PSI threshold</label>
          <input type="number" step="0.05" min="0" value={psiThreshold} onChange={(e) => setPsiThreshold(e.target.value)} />
        </div>
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 130 }}>
          <label>Categorical threshold</label>
          <input type="number" step="0.05" min="0" value={categoricalThreshold} onChange={(e) => setCategoricalThreshold(e.target.value)} />
        </div>
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 130 }}>
          <label>Missingness threshold</label>
          <input type="number" step="0.05" min="0" value={missingnessThreshold} onChange={(e) => setMissingnessThreshold(e.target.value)} />
        </div>
        <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
          <label>Range threshold</label>
          <input type="number" step="0.05" min="0" value={rangeThreshold} onChange={(e) => setRangeThreshold(e.target.value)} />
        </div>
      </div>

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || !reference || !current || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Computing…" : "Run drift check"}
      </button>

      {report && (
        <div style={{ marginTop: 14 }}>
          <div className="row" style={{ alignItems: "center", marginBottom: 12 }}>
            {report.alerts.length === 0 ? (
              <Badge tone="ok">No drift alerts</Badge>
            ) : (
              report.alerts.map((a, i) => (
                <Badge key={i} tone="err">{a.type}: {a.column}</Badge>
              ))
            )}
          </div>

          {report.schema_changes.missing_columns.length > 0 && (
            <p className="muted">Missing columns: {report.schema_changes.missing_columns.join(", ")}</p>
          )}
          {report.schema_changes.extra_columns.length > 0 && (
            <p className="muted">Extra columns: {report.schema_changes.extra_columns.join(", ")}</p>
          )}

          {numericEntries.length > 0 && (
            <>
              <h3 style={{ marginTop: 12 }}>Numeric drift (PSI)</h3>
              {numericEntries.map(([col, v]) => (
                <div className="bar-row" key={col}>
                  <span className="mono">{col}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${Math.min(100, ((v.value ?? 0) / maxPsi) * 100)}%` }} />
                  </div>
                  <span className="mono">{fmtNum(v.value)}</span>
                </div>
              ))}
            </>
          )}

          {Object.keys(report.drift.categorical).length > 0 && (
            <>
              <h3 style={{ marginTop: 14 }}>Categorical drift</h3>
              <Table<string>
                columns={["Column", "Metric", "Value"]}
                rows={Object.entries(report.drift.categorical).map(([k, v]) => [k, v.metric, v.value] as unknown as string)}
                keyOf={(r) => r[0]}
                render={(r) => [
                  <td key="c" className="mono">{r[0]}</td>,
                  <td key="m">{r[1]}</td>,
                  <td key="v" className="mono">{fmtNum(Number(r[2]))}</td>,
                ]}
              />
            </>
          )}

          {Object.keys(report.data_quality.missingness).length > 0 && (
            <>
              <h3 style={{ marginTop: 14 }}>Missingness</h3>
              <Table<string>
                columns={["Column", "Reference", "Current", "Δ"]}
                rows={Object.entries(report.data_quality.missingness).map(([k, v]) => [k, v.reference, v.current, v.delta] as unknown as string)}
                keyOf={(r) => r[0]}
                render={(r) => [
                  <td key="c" className="mono">{r[0]}</td>,
                  <td key="ref" className="mono">{fmtPct(Number(r[1]))}</td>,
                  <td key="cur" className="mono">{fmtPct(Number(r[2]))}</td>,
                  <td key="d" className="mono">{fmtNum(Number(r[3]))}</td>,
                ]}
              />
            </>
          )}
        </div>
      )}
    </Card>
  );
}

/** Source kind for a prediction series picker. */
type SourceKind = "partition" | "request_id" | "values";

function SourcePicker({
  label,
  partitions,
  columns,
  allowColumn = false,
  placeholderValues = "[0.9, 0.2, …]",
  onPick,
}: {
  label: string;
  partitions: string[];
  columns: string[];
  allowColumn?: boolean;
  placeholderValues?: string;
  onPick: (source: PredictionSource | null) => void;
}) {
  const [kind, setKind] = useState<SourceKind>("partition");
  const [value, setValue] = useState("");
  const [column, setColumn] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Report the current selection upward whenever any part of it changes.
  const emit = (k: SourceKind, v: string, c: string) => {
    if (k === "values") {
      if (!v.trim()) {
        setError(null);
        onPick(null);
        return;
      }
      const parsed = parseJsonArraySafe(v);
      if (parsed === null) {
        setError(`${label}: expected a JSON array of numbers`);
        onPick(null);
      } else {
        setError(null);
        // Send values verbatim: numbers stay numbers, group labels stay strings.
        onPick({ values: parsed as Array<number | string> });
      }
    } else if (!v) {
      setError(null);
      onPick(null);
    } else if (k === "partition") {
      setError(null);
      onPick({ partition: v, column: c || null });
    } else {
      setError(null);
      onPick({ request_id: v, column: c || null });
    }
  };

  return (
    <div className="field fixed" style={{ flex: "1 1 260px", minWidth: 240 }}>
      <label>{label}</label>
      <div className="row">
        <select value={kind} onChange={(e) => { const k = e.target.value as SourceKind; setKind(k); emit(k, value, column); }} style={{ maxWidth: 150 }}>
          <option value="partition">partition</option>
          <option value="request_id">artifact id</option>
          <option value="values">inline values</option>
        </select>
        {kind === "partition" && (
          <>
            <select value={value} onChange={(e) => { setValue(e.target.value); emit(kind, e.target.value, column); }}>
              <option value="">— select —</option>
              {partitions.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
            {allowColumn && (
              <select value={column} onChange={(e) => { setColumn(e.target.value); emit(kind, value, e.target.value); }}>
                <option value="">target col</option>
                {columns.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            )}
          </>
        )}
        {kind === "request_id" && (
          <>
            <input value={value} onChange={(e) => { setValue(e.target.value); emit(kind, e.target.value, column); }} placeholder="request id from predict-batch" spellCheck={false} />
            {allowColumn && (
              <select value={column} onChange={(e) => { setColumn(e.target.value); emit(kind, value, e.target.value); }}>
                <option value="">prediction</option>
                {columns.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            )}
          </>
        )}
        {kind === "values" && (
          <input className="mono" value={value} onChange={(e) => { setValue(e.target.value); emit(kind, e.target.value, column); }} placeholder={placeholderValues} spellCheck={false} />
        )}
      </div>
      {error && <p style={{ fontSize: 12.5, color: "var(--err)" }}>{error}</p>}
    </div>
  );
}

/** Returns null instead of throwing (for live picker state). */
function parseJsonArraySafe(s: string): unknown[] | null {
  try {
    const v = JSON.parse(s);
    return Array.isArray(v) ? v : null;
  } catch {
    return null;
  }
}

function PredictionDriftCard({ task, partitions, columns, versions }: { task: string; partitions: string[]; columns: string[]; versions: string[] }) {
  const [version, setVersion] = useState("");
  const [threshold, setThreshold] = useState("0.2");
  const [bins, setBins] = useState("10");
  const [reference, setReference] = useState<PredictionSource | null>(null);
  const [current, setCurrent] = useState<PredictionSource | null>(null);

  const mut = useMutation({
    mutationFn: () => {
      if (!version) throw new Error("Pick a model version");
      if (!reference || !current) throw new Error("Pick both a reference and a current source");
      return checkPredictionDrift(task, version, {
        reference_source: reference,
        current_source: current,
        threshold: Number(threshold) || 0.2,
        bins: Math.max(2, Math.min(50, Number(bins) || 10)),
      });
    },
  });

  const report = mut.data as DriftReport | undefined;
  const psiEntry = report ? Object.entries(report.drift.numeric)[0] : undefined;
  const psiValue = psiEntry?.[1].value ?? null;

  return (
    <Card title="Prediction distribution drift" sub="PSI between two prediction series — stored partitions, persisted inference artifacts, or inline arrays.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
          <label>Model version</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 130px", minWidth: 110 }}>
          <label>PSI threshold</label>
          <input type="number" step="0.05" min="0" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
        </div>
        <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 90 }}>
          <label>Bins</label>
          <input type="number" step="1" min="2" max="50" value={bins} onChange={(e) => setBins(e.target.value)} />
        </div>
      </div>

      <SourcePicker label="Reference predictions" partitions={partitions} columns={columns} allowColumn placeholderValues="[0.9, 0.2, …]" onPick={setReference} />
      <SourcePicker label="Current predictions" partitions={partitions} columns={columns} allowColumn placeholderValues="[0.8, 0.3, …]" onPick={setCurrent} />

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || !version || !reference || !current || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Computing…" : "Run prediction drift"}
      </button>

      {report && (
        <div style={{ marginTop: 14 }}>
          <div className="row" style={{ alignItems: "center", marginBottom: 12 }}>
            {report.alerts.length === 0 ? (
              <Badge tone="ok">No prediction drift</Badge>
            ) : (
              report.alerts.map((a, i) => (
                <Badge key={i} tone="err">{a.type}: {a.column}</Badge>
              ))
            )}
          </div>

          {psiValue !== null && psiEntry && (
            <div className="bar-row">
              <span className="mono">prediction PSI</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, (psiValue / Math.max((Number(threshold) || 0.2) * 2, psiValue)) * 100)}%` }} />
              </div>
              <span className="mono">{fmtNum(psiValue)}</span>
            </div>
          )}

          {Object.keys(report.drift.categorical).length > 0 && (
            <>
              <h3 style={{ marginTop: 12 }}>Categorical drift</h3>
              <Table<string>
                columns={["Column", "Metric", "Value"]}
                rows={Object.entries(report.drift.categorical).map(([k, v]) => [k, v.metric, v.value] as unknown as string)}
                keyOf={(r) => r[0]}
                render={(r) => [
                  <td key="c" className="mono">{r[0]}</td>,
                  <td key="m">{r[1]}</td>,
                  <td key="v" className="mono">{fmtNum(Number(r[2]))}</td>,
                ]}
              />
            </>
          )}

          {Object.keys(report.data_quality.missingness).length > 0 && (
            <>
              <h3 style={{ marginTop: 14 }}>Missingness</h3>
              <Table<string>
                columns={["Column", "Reference", "Current", "Δ"]}
                rows={Object.entries(report.data_quality.missingness).map(([k, v]) => [k, v.reference, v.current, v.delta] as unknown as string)}
                keyOf={(r) => r[0]}
                render={(r) => [
                  <td key="c" className="mono">{r[0]}</td>,
                  <td key="ref" className="mono">{fmtPct(Number(r[1]))}</td>,
                  <td key="cur" className="mono">{fmtPct(Number(r[2]))}</td>,
                  <td key="d" className="mono">{fmtNum(Number(r[3]))}</td>,
                ]}
              />
            </>
          )}
        </div>
      )}
    </Card>
  );
}

function SubgroupPerformanceCard({ task, partitions, columns, versions }: { task: string; partitions: string[]; columns: string[]; versions: string[] }) {
  const [version, setVersion] = useState("");
  const [metric, setMetric] = useState<SubgroupMetric>("accuracy");
  const [yTrue, setYTrue] = useState<PredictionSource | null>(null);
  const [yPred, setYPred] = useState<PredictionSource | null>(null);
  const [groups, setGroups] = useState<PredictionSource | null>(null);

  const mut = useMutation({
    mutationFn: () => {
      if (!yTrue || !yPred || !groups) throw new Error("Pick y_true, y_pred and a groups source");
      return checkSubgroupPerformance(task, version, {
        y_true_source: yTrue,
        y_pred_source: yPred,
        groups_source: groups,
        metric,
      });
    },
  });

  const report = mut.data as SubgroupPerformanceReport | undefined;
  const chartRows = report
    ? Object.entries(report.groups).map(([g, s]) => ({ group: g, value: (s[report.metric as SubgroupMetric] ?? 0) as number, count: s.count }))
    : [];

  return (
    <Card title="Subgroup performance" sub={`Per-group ${metric} and the fairness gap between best and worst subgroup.`}>
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
          <label>Model version</label>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
          <label>Metric</label>
          <select value={metric} onChange={(e) => setMetric(e.target.value as SubgroupMetric)}>
            {(["accuracy", "precision", "recall", "f1", "loss"] as const).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      <SourcePicker label="y_true" partitions={partitions} columns={columns} allowColumn placeholderValues="[1, 0, 1, …]" onPick={setYTrue} />
      <SourcePicker label="y_pred" partitions={partitions} columns={columns} allowColumn placeholderValues="[1, 0, 0, …]" onPick={setYPred} />
      <SourcePicker label="groups (column or inline)" partitions={partitions} columns={columns} allowColumn placeholderValues='["a", "b", "a", …]' onPick={setGroups} />

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || !version || !yTrue || !yPred || !groups || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Computing…" : "Run subgroup report"}
      </button>

      {report && (
        <div style={{ marginTop: 14 }}>
          <div className="grid cols-2" style={{ marginBottom: 12 }}>
            <Stat label={`Fairness gap (${metric})`} value={fmtNum(report.fairness_gap)} sub="max − min across groups" />
            <Stat label="Groups" value={String(Object.keys(report.groups).length)} />
          </div>

          {chartRows.length > 0 && (
            <div style={{ height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                  <XAxis dataKey="group" tick={{ fill: "#8b95ab", fontSize: 12 }} />
                  <YAxis domain={["auto", "auto"]} tick={{ fill: "#8b95ab", fontSize: 12 }} />
                  <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                  <Bar dataKey="value" name={metric} radius={[3, 3, 0, 0]}>
                    {chartRows.map((row, i) => (
                      <Cell key={row.group} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {chartRows.length > 0 && (
            <Table<string[]>
              columns={["Group", metric, "Count"]}
              rows={chartRows.map((r) => [r.group, r.value, r.count] as unknown as string[])}
              keyOf={(r) => r[0]}
              render={(r) => [
                <td key="g" className="mono">{r[0]}</td>,
                <td key="v" className="mono">{fmtNum(Number(r[1]))}</td>,
                <td key="n" className="mono">{fmtNum(Number(r[2]))}</td>,
              ]}
            />
          )}
        </div>
      )}
    </Card>
  );
}

function SensitiveScanCard({ task, partitions, columns }: { task: string; partitions: string[]; columns: string[] }) {
  const [partition, setPartition] = useState("");
  const [extraColumns, setExtraColumns] = useState("");
  const [sensitiveFeatures, setSensitiveFeatures] = useState("");

  const mut = useMutation({
    mutationFn: () => {
      if (!partition && !extraColumns.trim()) throw new Error("Pick a partition or list column names");
      return scanSensitiveFeatures(task, {
        partition: partition || null,
        columns: extraColumns.split(",").map((c) => c.trim()).filter(Boolean),
        sensitive_features: sensitiveFeatures.split(",").map((c) => c.trim()).filter(Boolean),
      });
    },
  });

  const report = mut.data as { pii_columns: string[]; sensitive_columns: string[]; warnings: string[] } | undefined;

  return (
    <Card title="PII & sensitive columns" sub="Name-based scan of a stored partition — flags PII-like and explicitly sensitive column names. No cell values are read.">
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
          <label>Partition</label>
          <select value={partition} onChange={(e) => setPartition(e.target.value)}>
            <option value="">— none (use column list) —</option>
            {partitions.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </div>
        <div className="field fixed" style={{ flex: "1 1 260px", minWidth: 240 }}>
          <label>Columns (comma-separated, optional)</label>
          <input value={extraColumns} onChange={(e) => setExtraColumns(e.target.value)} placeholder={columns.slice(0, 5).join(", ") + "…"} spellCheck={false} />
        </div>
        <div className="field fixed" style={{ flex: "1 1 260px", minWidth: 240 }}>
          <label>Extra sensitive names (comma-separated)</label>
          <input value={sensitiveFeatures} onChange={(e) => setSensitiveFeatures(e.target.value)} placeholder="employee_id, …" spellCheck={false} />
        </div>
      </div>

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || (!partition && !extraColumns.trim()) || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Scanning…" : "Run scan"}
      </button>

      {report && (
        <div style={{ marginTop: 14 }}>
          {report.pii_columns.length === 0 && report.sensitive_columns.length === 0 ? (
            <Badge tone="ok">No PII-like or sensitive columns detected</Badge>
          ) : (
            <>
              {report.pii_columns.length > 0 && (
                <div className="row" style={{ marginBottom: 8 }}>
                  <span className="muted" style={{ fontSize: 12.5, marginRight: 6 }}>PII-like:</span>
                  {report.pii_columns.map((c) => (
                    <Badge key={c} tone="warn">{c}</Badge>
                  ))}
                </div>
              )}
              {report.sensitive_columns.length > 0 && (
                <div className="row" style={{ marginBottom: 8 }}>
                  <span className="muted" style={{ fontSize: 12.5, marginRight: 6 }}>Sensitive:</span>
                  {report.sensitive_columns.map((c) => (
                    <Badge key={c} tone="err">{c}</Badge>
                  ))}
                </div>
              )}
            </>
          )}
          {report.warnings.map((w, i) => (
            <p className="muted" key={i}>{w}</p>
          ))}
        </div>
      )}
    </Card>
  );
}

function FairnessCard({ task }: { task: string }) {
  const [yTrue, setYTrue] = useState("");
  const [yPred, setYPred] = useState("");
  const [groups, setGroups] = useState("");

  const mut = useMutation({
    mutationFn: () => {
      const parseArr = (s: string, name: string) => {
        try {
          const v = JSON.parse(s);
          if (!Array.isArray(v)) throw new Error(`${name} must be a JSON array`);
          return v;
        } catch (e) {
          throw new Error(`Invalid ${name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      };
      return checkFairness(task, { y_true: parseArr(yTrue, "y_true"), y_pred: parseArr(yPred, "y_pred"), groups: parseArr(groups, "groups") });
    },
  });

  const report = mut.data as FairnessReport | undefined;
  const groupRows = report ? Object.entries(report.groups) : [];

  return (
    <Card title="Fairness" sub="Group-wise selection / TPR / accuracy and parity gaps. Provide y_true, y_pred and groups as JSON arrays of equal length.">
      <div className="grid cols-3">
        <div>
          <label style={{ fontSize: 12.5, color: "var(--muted)" }}>y_true (JSON array)</label>
          <textarea className="mono" rows={4} value={yTrue} onChange={(e) => setYTrue(e.target.value)} placeholder="[1,0,1,…]" spellCheck={false} />
        </div>
        <div>
          <label style={{ fontSize: 12.5, color: "var(--muted)" }}>y_pred (JSON array)</label>
          <textarea className="mono" rows={4} value={yPred} onChange={(e) => setYPred(e.target.value)} placeholder="[1,0,0,…]" spellCheck={false} />
        </div>
        <div>
          <label style={{ fontSize: 12.5, color: "var(--muted)" }}>groups (JSON array)</label>
          <textarea className="mono" rows={4} value={groups} onChange={(e) => setGroups(e.target.value)} placeholder="[0,1,0,…]" spellCheck={false} />
        </div>
      </div>

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || !yTrue || !yPred || !groups || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Computing…" : "Run fairness check"}
      </button>

      {report && (
        <div style={{ marginTop: 14 }}>
          <div className="grid cols-2" style={{ marginBottom: 12 }}>
            <Stat label="Demographic parity gap" value={fmtNum(report.fairness.demographic_parity_gap)} />
            <Stat label="Equal opportunity gap" value={fmtNum(report.fairness.equal_opportunity_gap)} />
          </div>
          {groupRows.length > 0 && (
            <Table<string[]>
              columns={["Group", "Count", "Selection rate", "TPR", "Accuracy"]}
              rows={groupRows.map(([g, s]) => [g, s.count, s.selection_rate, s.true_positive_rate, s.accuracy] as unknown as string[])}
              keyOf={(r) => r[0]}
              render={(r) => [
                <td key="g" className="mono">{r[0]}</td>,
                <td key="n" className="mono">{fmtNum(Number(r[1]))}</td>,
                <td key="sr" className="mono">{fmtPct(Number(r[2]))}</td>,
                <td key="tpr" className="mono">{fmtPct(Number(r[3]))}</td>,
                <td key="acc" className="mono">{fmtPct(Number(r[4]))}</td>,
              ]}
            />
          )}
        </div>
      )}
    </Card>
  );
}

function DelayedLabelCard({ task, versions }: { task: string; versions: string[] }) {
  const [predictions, setPredictions] = useState("");
  const [labels, setLabels] = useState("");
  const [modelVersion, setModelVersion] = useState("");
  const [taskType, setTaskType] = useState<"classification" | "regression">("classification");

  const mut = useMutation({
    mutationFn: () => {
      const parseArr = (s: string, name: string) => {
        try {
          const v = JSON.parse(s);
          if (!Array.isArray(v)) throw new Error(`${name} must be a JSON array`);
          return v;
        } catch (e) {
          throw new Error(`Invalid ${name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      };
      return checkDelayedLabels(task, { predictions: parseArr(predictions, "predictions"), labels: parseArr(labels, "labels"), model_version: modelVersion || null, task_type: taskType });
    },
  });

  const res = mut.data as { rows: number; metrics: Record<string, number> } | undefined;

  return (
    <Card title="Delayed-label performance" sub="Score predictions against labels that arrive late.">
      <div className="grid cols-3">
        <div>
          <label style={{ fontSize: 12.5, color: "var(--muted)" }}>predictions (JSON array)</label>
          <textarea className="mono" rows={4} value={predictions} onChange={(e) => setPredictions(e.target.value)} placeholder="[0.9,0.2,…]" spellCheck={false} />
        </div>
        <div>
          <label style={{ fontSize: 12.5, color: "var(--muted)" }}>labels (JSON array)</label>
          <textarea className="mono" rows={4} value={labels} onChange={(e) => setLabels(e.target.value)} placeholder="[1,0,…]" spellCheck={false} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <label style={{ fontSize: 12.5, color: "var(--muted)" }}>Task type</label>
            <select value={taskType} onChange={(e) => setTaskType(e.target.value as "classification" | "regression")}>
              <option value="classification">classification</option>
              <option value="regression">regression</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12.5, color: "var(--muted)" }}>Model version (optional)</label>
            <select value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>
              <option value="">— none —</option>
              {versions.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || !predictions || !labels || mut.isPending} onClick={() => mut.mutate()}>
        {mut.isPending ? "Computing…" : "Evaluate"}
      </button>

      {res && (
        <div style={{ marginTop: 14 }}>
          <dl className="kv">
            <dt>Rows</dt>
            <dd>{fmtNum(res.rows)}</dd>
            {Object.entries(res.metrics).map(([k, v]) => (
              <MetricRow key={k} k={k} v={v} />
            ))}
          </dl>
        </div>
      )}
    </Card>
  );
}

function MetricRow({ k, v }: { k: string; v: number }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className="mono">{fmtNum(v)}</dd>
    </>
  );
}
