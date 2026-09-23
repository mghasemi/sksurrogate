import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  asSensitivityResult,
  getCorrelationMatrix,
  getWeightsHeatmap,
  runCorrelationThreshold,
  runSensitivity,
} from "../api/client";
import type { CorrelationThresholdResult, SensitivityJobResult } from "../api/client";
import { Heatmap } from "../components/Heatmap";
import { Card, ErrorNote, Loading, EmptyState, Tabs, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { useTask } from "../lib/task-context";

const METHODS = ["sobol", "morris", "delta-mmnt"] as const;
const TABS = ["ranking", "correlation", "weights"] as const;
type TabId = (typeof TABS)[number];

export default function FeatureAnalysisPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const [method, setMethod] = useState<(typeof METHODS)[number]>("sobol");
  const [nFeatures, setNFeatures] = useState(5);
  const [trainPartition, setTrainPartition] = useState("train");
  const [jobId, setJobId] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(0.7);
  const [corrResult, setCorrResult] = useState<CorrelationThresholdResult | null>(null);

  // Result of the most recent sensitivity job, captured for the ranking heatmap.
  const [sensResult, setSensResult] = useState<SensitivityJobResult | null>(null);

  const sensMut = useMutation({
    mutationFn: () => runSensitivity(task, { method, n_features_to_select: nFeatures, train_partition: trainPartition }),
    onSuccess: (res) => {
      setJobId(res.job_id);
      setSensResult(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const corrMut = useMutation({
    mutationFn: () => runCorrelationThreshold(task, threshold, trainPartition),
    onSuccess: (res) => setCorrResult(res),
  });

  // Matrix views are opt-in per tab (wide datasets are rejected with a 400 upstream).
  const [tab, setTab] = useState<TabId>("ranking");

  const corrMatrixQ = useQuery({
    queryKey: ["correlation-matrix", task, trainPartition],
    queryFn: () => getCorrelationMatrix(task, trainPartition),
    enabled: !!task && tab === "correlation",
    retry: false,
  });

  const weightsQ = useQuery({
    queryKey: ["weights-heatmap", task, trainPartition],
    queryFn: () => getWeightsHeatmap(task, trainPartition),
    enabled: !!task && tab === "weights",
    retry: false,
  });

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Feature Analysis</h1>
        <span className="desc">Sensitivity analysis (job-based) and fast correlation pruning for task “{task || "…"}”</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      <div className="grid cols-2">
        <Card title="Sensitivity analysis" sub="Ranks features via Sobol / Morris / delta-mmnt. Runs as a background job — can be slow on wide feature sets.">
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 130 }}>
              <label>Method</label>
              <select value={method} onChange={(e) => setMethod(e.target.value as (typeof METHODS)[number])}>
                {METHODS.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 130px", minWidth: 100 }}>
              <label>Top-N features</label>
              <input type="number" min={1} value={nFeatures} onChange={(e) => setNFeatures(Number(e.target.value))} />
            </div>
            <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
              <label>Train partition</label>
              <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
            </div>
          </div>

          {sensMut.isError && <ErrorNote error={sensMut.error} />}

          <button className="btn primary" disabled={!task || sensMut.isPending} onClick={() => sensMut.mutate()}>
            {sensMut.isPending ? "Submitting…" : "Run sensitivity analysis"}
          </button>

          {jobId && (
            <>
              <div style={{ height: 14 }} />
              <JobTracker
                jobId={jobId}
                onResult={(result) => {
                  const decoded = asSensitivityResult(result);
                  if (!decoded) return;
                  setSensResult(decoded);
                  setTab("ranking");
                }}
              />
            </>
          )}
        </Card>

        <Card title="Correlation pruning" sub="Synchronous — drops features whose |corr| exceeds the threshold.">
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
              <label>Threshold</label>
              <input type="number" step={0.05} min={0} max={1} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            </div>
          </div>

          {corrMut.isError && <ErrorNote error={corrMut.error} />}

          <button className="btn primary" disabled={!task || corrMut.isPending} onClick={() => corrMut.mutate()}>
            {corrMut.isPending ? "Pruning…" : "Run correlation threshold"}
          </button>

          {corrResult && (
            <>
              <div style={{ height: 14 }} />
              <p className="muted">
                Kept <strong>{fmtNum(corrResult.kept_feature_names.length)}</strong>, dropped{" "}
                <strong>{fmtNum(corrResult.dropped_feature_names.length)}</strong> of {fmtNum(corrResult.feature_columns.length)} features.
              </p>
              <div className="chips">
                {corrResult.feature_columns.map((f) => (
                  <span key={f} className={`chip${corrResult.dropped_feature_names.includes(f) ? " dropped" : ""}`}>
                    {f}
                  </span>
                ))}
              </div>
            </>
          )}
        </Card>
      </div>

      <Card
        title="Heatmaps"
        sub="Rendered from the same source as SKSurrogate's mltrack.heatmap. Each matrix loads on demand when its tab is opened."
      >
        {!task && <p className="muted">Pick a task name to render its feature matrices.</p>}

        {task && (
          <>
            <Tabs
              active={tab}
              onChange={(id) => setTab(id as TabId)}
              tabs={[
                { id: "ranking", label: "Sensitivity ranking", badge: sensResult?.method },
                { id: "correlation", label: "Correlation matrix", badge: "Pearson" },
                { id: "weights", label: "Feature weights", badge: weightsQ.data?.source },
              ]}
            />

            {tab !== "ranking" && (
              <div className="row">
                <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
                  <label>Train partition</label>
                  <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
                </div>
              </div>
            )}

            {tab === "ranking" &&
              (sensResult ? (
                <SensitivityHeatmap result={sensResult} />
              ) : (
                <EmptyState
                  title="No sensitivity run yet"
                  hint="Run a sensitivity analysis above — its ranking heatmap appears here as soon as the job completes."
                />
              ))}

            {tab === "correlation" && (
              <>
                <h4 className="hm-title">Pearson correlation matrix</h4>
                {corrMatrixQ.isLoading && <Loading label="Computing correlations…" />}
                {corrMatrixQ.isError && <ErrorNote error={corrMatrixQ.error} />}
                {corrMatrixQ.data && <Heatmap data={corrMatrixQ.data} mode="diverging" />}
                <p className="muted" style={{ marginTop: 6 }}>
                  Pairwise correlation of the <code>{trainPartition}</code> partition, from{" "}
                  <code>/api/sensitivity/{task}/correlation-matrix</code>.
                </p>
              </>
            )}

            {tab === "weights" && (
              <>
                <h4 className="hm-title">Feature weights</h4>
                {weightsQ.isLoading && <Loading />}
                {weightsQ.isError && <ErrorNote error={weightsQ.error} />}
                {weightsQ.data && !weightsQ.data.available && (
                  <EmptyState
                    title="No weights available for this task"
                    hint="Register a dataset partition so weights can be derived."
                  />
                )}
                {weightsQ.data?.available && weightsQ.data.values.length > 0 && (
                  <>
                    <Heatmap data={weightsQ.data} mode="sequential" />
                    <p className="muted" style={{ marginTop: 6 }}>
                      {weightsQ.data.source === "stored"
                        ? "Read from the task's stored weights table (mltrack.heatmap's default source)."
                        : "Computed on demand: Pearson correlation with the target and feature variance. Store weights via mltrack.FeatureWeights (sobol / morris / relieff / …) to add columns."}
                    </p>
                  </>
                )}
              </>
            )}
          </>
        )}
      </Card>

      <p className="muted">
        Tip: the top-N features from a sensitivity run can be fed into the Experiments stage to constrain the search space.
      </p>
    </div>
  );
}

/** Single-column heatmap of one sensitivity run, sorted by descending |weight|. */
function SensitivityHeatmap({ result }: { result: SensitivityJobResult }) {
  const rows = result.feature_columns
    .map((name, i) => ({ name, weight: result.weights[i] ?? null }))
    .sort((a, b) => Math.abs(b.weight ?? 0) - Math.abs(a.weight ?? 0));
  const top = new Set(result.top_feature_names);
  const unit =
    result.method === "sobol"
      ? "total-order Sobol index Sₜ"
      : result.method === "morris"
        ? "Morris μ*"
        : "delta moment-independent measure";

  return (
    <>
      <p className="muted">
        Weights for all {fmtNum(result.feature_columns.length)} features, strongest first; the top{" "}
        {fmtNum(result.n_features_to_select)} are marked with ★.
      </p>
      <Heatmap
        mode="sequential"
        data={{
          labels: rows.map((r) => (top.has(r.name) ? `★ ${r.name}` : r.name)),
          columns: [result.method],
          values: rows.map((r) => [r.weight]),
        }}
      />
      <p className="muted" style={{ marginTop: 6 }}>
        Weight = {unit}. Hover a cell for the exact value; the raw payload is under “Show result”.
      </p>
    </>
  );
}
