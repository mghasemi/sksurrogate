import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { checkQuality, listBundles } from "../api/client";
import type { QualityGateReport } from "../api/client";
import { Card, ErrorNote, StatusBadge, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function QualityGatesPage() {
  const { task } = useTask();

  const listQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });
  const bundles = listQ.data?.bundles ?? [];

  const [modelVersion, setModelVersion] = useState("");
  const version = modelVersion || bundles[0] || "";

  const [thresholdsJson, setThresholdsJson] = useState("{}");
  const [maxSize, setMaxSize] = useState("");

  const checkMut = useMutation({
    mutationFn: () => {
      let thresholds: Record<string, number> = {};
      try {
        thresholds = JSON.parse(thresholdsJson || "{}") as Record<string, number>;
      } catch (e) {
        throw new Error(`Invalid metric_thresholds JSON: ${e instanceof Error ? e.message : String(e)}`);
      }
      return checkQuality(task, version, {
        metric_thresholds: thresholds,
        max_bundle_size_bytes: maxSize ? Number(maxSize) : null,
      });
    },
  });

  const report = checkMut.data as QualityGateReport | undefined;

  if (!task) {
    return <div className="error-note">Pick a task name in the top bar first.</div>;
  }

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Quality Gates</h1>
        <span className="desc">CI-style checks on stored bundles — must pass before promotion in the Registry</span>
      </div>

      <Card title="Run a gate check">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
            <label>Bundle (model version)</label>
            {bundles.length > 0 ? (
              <select value={version} onChange={(e) => setModelVersion(e.target.value)}>
                {[...bundles].reverse().map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            ) : (
              <input value={version} onChange={(e) => setModelVersion(e.target.value)} placeholder="model_version" spellCheck={false} />
            )}
          </div>
          <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
            <label>Max bundle size (bytes, optional)</label>
            <input type="number" value={maxSize} onChange={(e) => setMaxSize(e.target.value)} placeholder="unlimited" />
          </div>
        </div>

        <div className="field">
          <label>{'Metric thresholds (JSON — e.g. {"train_score": 0.5})'}</label>
          <textarea className="mono" rows={3} value={thresholdsJson} onChange={(e) => setThresholdsJson(e.target.value)} spellCheck={false} />
        </div>

        {checkMut.isError && <ErrorNote error={checkMut.error} />}

        <button className="btn primary" disabled={!version || checkMut.isPending} onClick={() => checkMut.mutate()}>
          {checkMut.isPending ? "Checking…" : "Run quality gate"}
        </button>
      </Card>

      {report && (
        <Card title={`Gate result — ${report.model_version}`}>
          <div style={{ marginBottom: 12 }}>
            {report.passed ? <StatusBadge status="passed" /> : <StatusBadge status="failed" />}
            <span className="muted" style={{ marginLeft: 10 }}>
              {fmtNum(report.checks.length - report.failures.length)}/{fmtNum(report.checks.length)} checks passed
            </span>
          </div>

          <table className="tbl">
            <thead>
              <tr>
                <th>Check</th>
                <th>Status</th>
                <th>Metric</th>
                <th>Value</th>
                <th>Bounds</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody>
              {report.checks.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.passed ? <StatusBadge status="passed" /> : <StatusBadge status="failed" />}</td>
                  <td className="mono">{c.metric ?? "—"}</td>
                  <td>{fmtNum(c.value)}</td>
                  <td className="muted">
                    {c.minimum !== undefined ? `≥ ${fmtNum(c.minimum)}` : ""}
                    {c.maximum !== undefined ? ` ≤ ${fmtNum(c.maximum)}` : ""}
                  </td>
                  <td style={{ color: "var(--err)" }}>{c.error ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {report.failures.length > 0 && (
            <div className="error-note" style={{ marginTop: 12 }}>
              Failures block promotion in the Registry stage.
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
