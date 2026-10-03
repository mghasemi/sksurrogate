import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { checkQuality, getBundle, listBundles } from "../api/client";
import type { BundleSummary, QualityGateReport } from "../api/client";
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

  // The gate only accepts metric names that the bundle actually recorded, so the
  // bundle's own metrics are the authoritative list of valid threshold keys.
  const detailQ = useQuery<BundleSummary>({
    queryKey: ["bundle-detail", task, version],
    queryFn: () => getBundle(task, version),
    enabled: !!task && !!version,
  });

  const bundleMetrics = useMemo(
    () =>
      Object.entries(detailQ.data?.metrics ?? {}).filter(
        (entry): entry is [string, number] => typeof entry[1] === "number",
      ),
    [detailQ.data],
  );

  // Live-parse the textarea so the shortcuts can mirror whatever is currently set,
  // including hand-typed keys the bundle does not know about.
  const parsed = useMemo<{ ok: boolean; value: Record<string, number> }>(() => {
    try {
      const raw: unknown = JSON.parse(thresholdsJson.trim() || "{}");
      if (raw && typeof raw === "object" && !Array.isArray(raw)) {
        return { ok: true, value: raw as Record<string, number> };
      }
    } catch {
      /* invalid JSON — surfaced in the UI, shortcuts disabled */
    }
    return { ok: false, value: {} };
  }, [thresholdsJson]);

  const removeThreshold = (name: string) => {
    if (!parsed.ok) return;
    const next = { ...parsed.value };
    delete next[name];
    setThresholdsJson(JSON.stringify(next, null, 2));
  };

  const toggleMetric = (name: string, current: number) => {
    if (!parsed.ok) return;
    if (name in parsed.value) {
      removeThreshold(name);
      return;
    }
    // Default the minimum to the bundle's current value, floored to 4 dp so the
    // check passes on the stored value instead of tripping over rounding.
    const suggested = Math.floor(current * 1e4) / 1e4;
    setThresholdsJson(JSON.stringify({ ...parsed.value, [name]: suggested }, null, 2));
  };

  const thresholdKeys = parsed.ok ? Object.keys(parsed.value) : [];
  const unknownThresholds = thresholdKeys.filter((k) => !bundleMetrics.some(([n]) => n === k));

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
          <label>{'Metric thresholds (JSON object — each metric maps to the minimum value it must reach, e.g. {"train_score": 0.5})'}</label>
          <textarea className="mono" rows={3} value={thresholdsJson} onChange={(e) => setThresholdsJson(e.target.value)} spellCheck={false} />

          {detailQ.isLoading && !!version && <span className="hint">Loading metrics for this bundle…</span>}
          {detailQ.isError && <span className="hint">Could not read this bundle’s metrics — type metric names manually.</span>}

          {bundleMetrics.length > 0 && (
            <>
              <span className="hint">Metrics recorded in this bundle — click one to add (or remove) its threshold:</span>
              <div className="chips">
                {bundleMetrics.map(([name, current]) => {
                  const active = parsed.ok && name in parsed.value;
                  return (
                    <button
                      type="button"
                      key={name}
                      className={`chip metric-chip${active ? " active" : ""}`}
                      disabled={!parsed.ok}
                      title={
                        active
                          ? `Remove the "${name}" threshold`
                          : `Require ${name} ≥ ${current} (its current value in this bundle)`
                      }
                      onClick={() => toggleMetric(name, current)}
                    >
                      <span className="chip-metric">{name}</span>
                      <span className="chip-val">{active ? `≥ ${fmtNum(parsed.value[name])}` : fmtNum(current)}</span>
                    </button>
                  );
                })}
              </div>
            </>
          )}

          {detailQ.data && bundleMetrics.length === 0 && (
            <span className="hint">This bundle records no numeric metrics, so there are no shortcuts — type metric names manually.</span>
          )}

          {unknownThresholds.length > 0 && (
            <div className="chips">
              {unknownThresholds.map((name) => (
                <button
                  type="button"
                  key={name}
                  className="chip metric-chip unknown"
                  title={`"${name}" is not a metric recorded in this bundle — this check will fail. Click to remove.`}
                  onClick={() => removeThreshold(name)}
                >
                  <span className="chip-metric">{name}</span>
                  <span className="chip-val">not in bundle</span>
                </button>
              ))}
            </div>
          )}

          {!parsed.ok && (
            <span className="hint" style={{ color: "var(--err)" }}>
              Invalid JSON — the gate run will error until this parses (shortcuts are disabled meanwhile).
            </span>
          )}

          {parsed.ok && thresholdKeys.length > 0 && (
            <span className="hint">
              {thresholdKeys.length} threshold{thresholdKeys.length === 1 ? "" : "s"} set.{" "}
              <button type="button" className="link-btn" onClick={() => setThresholdsJson("{}")}>
                Clear all
              </button>
            </span>
          )}
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
