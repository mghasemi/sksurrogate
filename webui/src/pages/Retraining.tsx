import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  BASELINE_ESTIMATORS,
  DEFAULT_EXPERIMENT_CONFIG,
  REGISTRY_STATES,
  getDatasetMetadata,
  listJobs,
  runRetraining,
} from "../api/client";
import type { JobRecord, ParamSpec, RunRetrainingRequest } from "../api/client";
import { Card, ErrorNote, StatusBadge, Table } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { ScoringSelect } from "../components/ScoringSelect";
import { useTask } from "../lib/task-context";

type TrainerKind = "baseline" | "experiment";
type Trigger = "always" | "scheduled" | "on_data";

function defaultConfigJson(): string {
  return JSON.stringify(DEFAULT_EXPERIMENT_CONFIG, null, 2);
}

export default function RetrainingPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  /* Trainer selection */
  const [kind, setKind] = useState<TrainerKind>("baseline");
  const [estimator, setEstimator] = useState<string>(BASELINE_ESTIMATORS[0]);
  const [trainPartition, setTrainPartition] = useState("train");
  const [validationPartition, setValidationPartition] = useState("validation");

  /* Experiment trainer fields */
  const [configJson, setConfigJson] = useState(defaultConfigJson);
  const [length, setLength] = useState(2);
  const [maxGeneration, setMaxGeneration] = useState(3);
  const [numParents, setNumParents] = useState(4);
  const [scoring, setScoring] = useState("accuracy");

  /* Trigger + promotion */
  const [trigger, setTrigger] = useState<Trigger>("always");
  const [due, setDue] = useState(true);
  const [promotionState, setPromotionState] = useState<string>("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  const datasetQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
  });
  const partitions = useMemo(
    () => Object.keys(datasetQ.data?.partitions ?? {}),
    [datasetQ.data],
  );

  const historyQ = useQuery({
    queryKey: ["jobs", task],
    queryFn: () => listJobs(task),
    enabled: !!task,
    refetchInterval: 5000,
  });
  const retrainingJobs = useMemo(
    () => (historyQ.data?.jobs ?? []).filter((j) => j.kind === "retraining"),
    [historyQ.data],
  );

  const runMut = useMutation({
    mutationFn: (): Promise<{ job_id: string; status: string }> => {
      let trainer: RunRetrainingRequest["trainer"];
      if (kind === "baseline") {
        trainer = {
          kind: "baseline",
          estimator,
          train_partition: trainPartition,
          validation_partition: validationPartition || null,
        };
      } else {
        let config: Record<string, Record<string, ParamSpec>>;
        try {
          config = JSON.parse(configJson) as Record<string, Record<string, ParamSpec>>;
        } catch (e) {
          throw new Error(`Invalid search-space JSON: ${e instanceof Error ? e.message : String(e)}`);
        }
        trainer = {
          kind: "experiment",
          config,
          length,
          max_generation: maxGeneration,
          num_parents: numParents,
          train_partition: trainPartition,
          scoring,
        };
      }
      const body: RunRetrainingRequest = {
        trainer,
        trigger,
        due,
        promotion_state: promotionState || null,
      };
      return runRetraining(task, body);
    },
    onSuccess: (res) => {
      setJobId(res.job_id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const onSubmit = () => {
    if (kind === "experiment") {
      try {
        JSON.parse(configJson);
        setConfigError(null);
      } catch (e) {
        setConfigError(e instanceof Error ? e.message : String(e));
        return;
      }
    }
    runMut.mutate();
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Retraining</h1>
        <span className="desc">
          Fit a new bundle for task “{task || "…"}” as a background job — optionally auto-registering it into the model registry.
        </span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      <Card title="Trainer" sub="Baseline fits one estimator; experiment runs an AML/EOA search over a JSON-defined space.">
        <div className="row fixed">
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="radio" name="trainer-kind" checked={kind === "baseline"} onChange={() => setKind("baseline")} />
            Baseline
          </label>
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="radio" name="trainer-kind" checked={kind === "experiment"} onChange={() => setKind("experiment")} />
            Experiment (AML)
          </label>
        </div>

        {kind === "baseline" ? (
          <div className="row">
            <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
              <label>Estimator</label>
              <select value={estimator} onChange={(e) => setEstimator(e.target.value)}>
                {BASELINE_ESTIMATORS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <>
            <textarea
              className="mono"
              rows={12}
              value={configJson}
              onChange={(e) => setConfigJson(e.target.value)}
              spellCheck={false}
            />
            {configError && (
              <div className="error-note" style={{ marginTop: 8 }}>
                {configError}
              </div>
            )}
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
            </div>
          </>
        )}

        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
            <label>Train partition</label>
            {partitions.length > 0 ? (
              <select value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)}>
                {partitions.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            ) : (
              <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
            )}
          </div>
          {kind === "baseline" && (
            <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 130 }}>
              <label>Validation partition</label>
              {partitions.length > 0 ? (
                <select value={validationPartition} onChange={(e) => setValidationPartition(e.target.value)}>
                  <option value="">— none —</option>
                  {partitions.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              ) : (
                <input value={validationPartition} onChange={(e) => setValidationPartition(e.target.value)} spellCheck={false} />
              )}
            </div>
          )}
          {kind === "experiment" && (
            <ScoringSelect
              value={scoring}
              onChange={setScoring}
              disabled={!task}
              style={{ flex: "0 1 170px", minWidth: 140 }}
            />
          )}
        </div>
      </Card>

      <Card title="Trigger & promotion" sub="Triggers stay constrained JSON: “scheduled” / “on_data” evaluate the plain boolean below instead of running a callback.">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 120 }}>
            <label>Trigger</label>
            <select value={trigger} onChange={(e) => setTrigger(e.target.value as Trigger)}>
              <option value="always">always (run now)</option>
              <option value="scheduled">scheduled</option>
              <option value="on_data">on_data</option>
            </select>
          </div>
          {trigger !== "always" && (
            <label style={{ display: "flex", gap: 6, alignItems: "center", alignSelf: "flex-end", marginBottom: 6 }}>
              <input type="checkbox" checked={due} onChange={(e) => setDue(e.target.checked)} />
              due = true (job will run; if false it is skipped)
            </label>
          )}
          <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
            <label>Promotion state</label>
            <select value={promotionState} onChange={(e) => setPromotionState(e.target.value)}>
              <option value="">— register only (candidate) —</option>
              {REGISTRY_STATES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>

        {runMut.isError && <ErrorNote error={runMut.error} />}

        <button className="btn primary" disabled={!task || runMut.isPending} onClick={onSubmit}>
          {runMut.isPending ? "Submitting…" : "Start retraining"}
        </button>
      </Card>

      {jobId && (
        <>
          <JobTracker jobId={jobId} />
          <p className="muted">
            On completion the job result contains the new <code>model_version</code>; with a promotion state set it is also
            registered into the model registry.
          </p>
        </>
      )}

      <Card title="Retraining history" sub={`Recent retraining jobs for task “${task || "…"}” (auto-refreshes).`}>
        {retrainingJobs.length === 0 ? (
          <p className="muted">No retraining jobs yet.</p>
        ) : (
          <Table<JobRecord>
            columns={["Created", "Status", "Result"]}
            rows={retrainingJobs}
            keyOf={(j) => j.job_id}
            render={(j) => [
              <td key="c" className="mono">{new Date(j.created_at).toLocaleString()}</td>,
              <td key="s">
                <StatusBadge status={j.status} />
              </td>,
              <td key="r">{j.result ? "bundle produced" : j.error ? "failed" : "—"}</td>,
            ]}
          />
        )}
      </Card>
    </div>
  );
}
