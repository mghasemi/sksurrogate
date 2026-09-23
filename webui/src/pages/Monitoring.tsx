import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  checkDelayedLabels,
  checkDrift,
  checkFairness,
  getDatasetMetadata,
  listBundles,
  monitorSummary,
} from "../api/client";
import type { DriftReport, FairnessReport, MonitorSummary } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, Loading, Stat, Table, fmtNum, fmtPct } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function MonitoringPage() {
  const { task } = useTask();

  const bundlesQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });
  const datasetQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
    retry: false,
  });

  const versions = useMemo(() => bundlesQ.data?.bundles ?? [], [bundlesQ.data]);
  const partitions = useMemo(() => Object.keys(datasetQ.data?.partitions ?? {}), [datasetQ.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Monitoring</h1>
        <span className="desc">Drift, fairness and serving health for task “{task || "…"}”.</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {task && (
        <>
          <SummaryCard task={task} versions={versions} />
          <DriftCard task={task} partitions={partitions} versions={versions} />
          <FairnessCard task={task} />
          <DelayedLabelCard task={task} versions={versions} />
        </>
      )}
    </div>
  );
}

function SummaryCard({ task, versions }: { task: string; versions: string[] }) {
  const [version, setVersion] = useState("");
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

  const mut = useMutation({ mutationFn: () => checkDrift(task, { reference_partition: reference, current_partition: current, model_version: modelVersion || null }) });

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
