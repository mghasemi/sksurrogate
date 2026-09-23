import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { runCorrelationThreshold, runSensitivity } from "../api/client";
import type { CorrelationThresholdResult } from "../api/client";
import { Card, ErrorNote, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { useTask } from "../lib/task-context";

const METHODS = ["sobol", "morris", "delta-mmnt"] as const;

export default function FeatureAnalysisPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const [method, setMethod] = useState<(typeof METHODS)[number]>("sobol");
  const [nFeatures, setNFeatures] = useState(5);
  const [trainPartition, setTrainPartition] = useState("train");
  const [jobId, setJobId] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(0.7);
  const [corrResult, setCorrResult] = useState<CorrelationThresholdResult | null>(null);

  const sensMut = useMutation({
    mutationFn: () => runSensitivity(task, { method, n_features_to_select: nFeatures, train_partition: trainPartition }),
    onSuccess: (res) => {
      setJobId(res.job_id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const corrMut = useMutation({
    mutationFn: () => runCorrelationThreshold(task, threshold, trainPartition),
    onSuccess: (res) => setCorrResult(res),
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
              <JobTracker jobId={jobId} />
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

      <p className="muted">
        Tip: the top-N features from a sensitivity run can be fed into the Experiments stage to constrain the search space.
      </p>
    </div>
  );
}
