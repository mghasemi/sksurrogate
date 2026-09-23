import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { DEFAULT_EXPERIMENT_CONFIG, runExperiment } from "../api/client";
import type { ParamSpec, RunExperimentRequest } from "../api/client";
import { Card, ErrorNote } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { useTask } from "../lib/task-context";

function defaultConfigJson(): string {
  return JSON.stringify(DEFAULT_EXPERIMENT_CONFIG, null, 2);
}

export default function ExperimentsPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const [configJson, setConfigJson] = useState(defaultConfigJson);
  const [length, setLength] = useState(2);
  const [maxGeneration, setMaxGeneration] = useState(3);
  const [numParents, setNumParents] = useState(4);
  const [trainPartition, setTrainPartition] = useState("train");
  const [scoring, setScoring] = useState("accuracy");
  const [jobId, setJobId] = useState<string | null>(null);

  const [configError, setConfigError] = useState<string | null>(null);

  const runMut = useMutation({
    mutationFn: () => {
      let config: Record<string, Record<string, ParamSpec>>;
      try {
        config = JSON.parse(configJson) as Record<string, Record<string, ParamSpec>>;
      } catch (e) {
        throw new Error(`Invalid search-space JSON: ${e instanceof Error ? e.message : String(e)}`);
      }
      const body: RunExperimentRequest = {
        config,
        length,
        max_generation: maxGeneration,
        num_parents: numParents,
        train_partition: trainPartition,
        scoring,
      };
      return runExperiment(task, body);
    },
    onSuccess: (res) => {
      setJobId(res.job_id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const onSubmit = () => {
    try {
      JSON.parse(configJson);
      setConfigError(null);
      runMut.mutate();
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Experiments</h1>
        <span className="desc">AML/EOA pipeline search over a JSON-defined space — runs as a background job for task “{task || "…"}”</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      <Card title="Search space" sub="Keys are fully-qualified estimator class names; values map hyperparameters to real / integer / categorical specs.">
        <textarea
          className="mono"
          rows={16}
          value={configJson}
          onChange={(e) => setConfigJson(e.target.value)}
          spellCheck={false}
        />
        {configError && (
          <div className="error-note" style={{ marginTop: 8 }}>
            {configError}
          </div>
        )}
      </Card>

      <Card title="Run settings">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 120px", minWidth: 90 }}>
            <label>Length (trials)</label>
            <input type="number" min={1} value={length} onChange={(e) => setLength(Number(e.target.value))} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 100 }}>
            <label>Max generation</label>
            <input type="number" min={1} value={maxGeneration} onChange={(e) => setMaxGeneration(Number(e.target.value))} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 130px", minWidth: 100 }}>
            <label>Num parents</label>
            <input type="number" min={2} value={numParents} onChange={(e) => setNumParents(Number(e.target.value))} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
            <label>Train partition</label>
            <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
            <label>Scoring</label>
            <input value={scoring} onChange={(e) => setScoring(e.target.value)} spellCheck={false} />
          </div>
        </div>

        {runMut.isError && <ErrorNote error={runMut.error} />}

        <button className="btn primary" disabled={!task || runMut.isPending} onClick={onSubmit}>
          {runMut.isPending ? "Submitting…" : "Start experiment"}
        </button>
      </Card>

      {jobId && (
        <>
          <JobTracker jobId={jobId} />
          <p className="muted">
            On completion the job result contains <code>model_version</code>, <code>train_score</code> and the full{" "}
            <code>evaluation_history</code>. The new bundle appears under Model Bundles.
          </p>
        </>
      )}
    </div>
  );
}
