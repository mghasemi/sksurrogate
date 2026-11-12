import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  addProjectArtifact,
  DEFAULT_EXPERIMENT_CONFIG,
  STACKING_COMPONENT,
  getProject,
  getPresets,
  getScoringOptions,
  optimizePipeline,
  runExperiment,
} from "../api/client";
import type { ExperimentResult, JobRecord, ParamSpec, RunExperimentRequest, SearchSpace } from "../api/client";
import { Card, ErrorNote, Table, Tabs, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { ScoringSelect } from "../components/ScoringSelect";
import { fmtElapsed, useElapsed } from "../lib/hooks";
import { useTask } from "../lib/task-context";

function defaultConfigJson(): string {
  return JSON.stringify(DEFAULT_EXPERIMENT_CONFIG, null, 2);
}

interface ParsedSpace {
  config: SearchSpace;
  /** Top-level "forbidden" rules, e.g. [["penalty", "l1"]] or [["max_depth", 1]]. */
  forbidden: Array<[string, unknown]> | null;
  /** Union of every parameter name declared in the space. */
  knownParams: Set<string>;
  /** Per-estimator parameter names (for depends_on validation). */
  paramsByEstimator: Record<string, string[]>;
}

function shortName(estimator: string): string {
  const parts = estimator.split(".");
  return parts[parts.length - 1];
}

/** Reserved search-space key for the stacking directive (see `STACKING_COMPONENT`). */
const STACKING_KEY = "stacking";

/** Top-level key holding forbidden parameter combinations inside the JSON editor. */
const FORBIDDEN_KEY = "forbidden";

/** Parse the search-space JSON and collect the parameter references used for live hints. */
function parseSpace(json: string): ParsedSpace | { error: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (e) {
    return { error: `Invalid search-space JSON: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { error: "Search space must be a JSON object mapping estimator names to parameter specs." };
  }
  const document = { ...(raw as Record<string, unknown>) };
  // "forbidden" is a request field rather than a pipeline component, so it is
  // kept in the same JSON document for editability and split out here.
  const forbiddenRaw = document[FORBIDDEN_KEY];
  delete document[FORBIDDEN_KEY];
  let forbidden: Array<[string, unknown]> | null = null;
  if (forbiddenRaw !== undefined && forbiddenRaw !== null) {
    if (!Array.isArray(forbiddenRaw) || forbiddenRaw.some((p) => !Array.isArray(p) || p.length !== 2)) {
      return { error: '"forbidden" must be an array of [parameter, value] pairs.' };
    }
    forbidden = forbiddenRaw as Array<[string, unknown]>;
  }

  const config = document as SearchSpace;
  const knownParams = new Set<string>();
  const paramsByEstimator: Record<string, string[]> = {};
  for (const [estimator, params] of Object.entries(config)) {
    if (estimator === STACKING_KEY) continue; // directive, not a pipeline component
    if (typeof params !== "object" || params === null) continue;
    paramsByEstimator[estimator] = Object.keys(params);
    for (const name of Object.keys(params)) knownParams.add(name);
  }
  return { config, forbidden, knownParams, paramsByEstimator };
}

/** Live validation hints: unknown parameter references in forbidden rules and depends_on entries. */
function spaceWarnings(space: ParsedSpace, forbidden: Array<[string, unknown]> | null): string[] {
  const warnings: string[] = [];
  for (const [estimator, params] of Object.entries(space.config)) {
    for (const [name, spec] of Object.entries(params)) {
      if (!spec || typeof spec !== "object") continue;
      const dependsOn = (spec as ParamSpec).depends_on;
      if (!dependsOn) continue;
      const siblings = space.paramsByEstimator[estimator] ?? [];
      for (const other of Object.keys(dependsOn)) {
        if (!siblings.includes(other)) {
          warnings.push(`“${name}” in ${shortName(estimator)} depends on unknown parameter “${other}”`);
        }
      }
    }
  }
  for (const pair of forbidden ?? []) {
    if (!Array.isArray(pair) || pair.length !== 2) {
      warnings.push("Each forbidden rule must be a [parameter, value] pair");
      continue;
    }
    const param = String(pair[0]);
    if (!space.knownParams.has(param)) {
      warnings.push(`Forbidden rule references unknown parameter “${param}”`);
    }
  }
  return warnings;
}

interface PipelineJob {
  jobId: string;
  seq: string[];
  taskName: string;
}

/** Job statuses that mean "an experiment is still occupying a worker". */
const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);

export default function ExperimentsPage() {
  const { task } = useTask();
  const qc = useQueryClient();
  const [searchParams] = useSearchParams();
  const projectId = searchParams.get("project_id");
  const projectQ = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => {
      if (!projectId) throw new Error("Project id is required.");
      return getProject(projectId);
    },
    enabled: Boolean(projectId),
  });

  const [mode, setMode] = useState<"quick" | "advanced">("quick");
  const [preset, setPreset] = useState<"classification" | "regression">("classification");
  const [quickScoring, setQuickScoring] = useState("");
  const [budget, setBudget] = useState(1);
  const [engine, setEngine] = useState<"eoa" | "surrogate">("eoa");
  const [configJson, setConfigJson] = useState(defaultConfigJson);
  const [length, setLength] = useState(2);
  const [maxGeneration, setMaxGeneration] = useState(3);
  const [numParents, setNumParents] = useState(4);
  const [trainPartition, setTrainPartition] = useState("train");
  const [scoring, setScoring] = useState("accuracy");
  const [backend, setBackend] = useState<"local" | "dask">("local");
  const [strategy, setStrategy] = useState<"eoa" | "surrogate">("eoa");
  const [surrogateItrs, setSurrogateItrs] = useState(20);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobTask, setJobTask] = useState<string | null>(null);
  const [experimentLabel, setExperimentLabel] = useState("");
  const [submittedName, setSubmittedName] = useState<string | null>(null);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [pipelineJobs, setPipelineJobs] = useState<PipelineJob[]>([]);

  const [configError, setConfigError] = useState<string | null>(null);

  /**
   * Minimal snapshot of the experiment job so the banner above the fold can
   * show it. Kept as plain strings (not the whole record) and only replaced on
   * a real change, so the 1.5 s job poll does not re-render this whole page.
   */
  const [activeJob, setActiveJob] = useState<{
    status: string;
    displayName?: string;
    createdAt: string;
    updatedAt: string;
  } | null>(null);
  const experimentActive = Boolean(activeJob && ACTIVE_JOB_STATUSES.has(activeJob.status));
  const activeElapsed = useElapsed(
    activeJob?.createdAt,
    activeJob && !experimentActive ? activeJob.updatedAt : null,
  );
  const handleJobStatus = useCallback((job: JobRecord) => {
    setActiveJob((previous) => {
      if (
        previous &&
        previous.status === job.status &&
        previous.displayName === job.display_name &&
        previous.createdAt === job.created_at &&
        previous.updatedAt === job.updated_at
      ) {
        return previous;
      }
      return {
        status: job.status,
        displayName: job.display_name,
        createdAt: job.created_at,
        updatedAt: job.updated_at,
      };
    });
  }, []);
  /** Bring the live tracker into view right after a submission. */
  const scrollToTracker = () => {
    document.getElementById("experiment-job")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  // Fire after the tracker has actually mounted (a state update plus paint), so
  // the anchor exists; tying this to `jobId` beats a timer inside onSuccess.
  useEffect(() => {
    if (!jobId) return;
    const timer = window.setTimeout(scrollToTracker, 250);
    return () => window.clearTimeout(timer);
  }, [jobId]);
  const linkProjectJobMut = useMutation({
    mutationFn: ({
      completedJobId,
      label,
      taskName,
    }: {
      completedJobId: string;
      label: string;
      taskName: string;
    }) => {
      if (!projectId) throw new Error("Project context is missing.");
      return addProjectArtifact(projectId, {
        kind: "job",
        task_name: taskName,
        artifact_id: completedJobId,
        label,
      });
    },
    onSuccess: (project) => {
      qc.setQueryData(["project", project.project_id], project);
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project-summary", project.project_id] });
    },
  });
  const linkProjectBundleMut = useMutation({
    mutationFn: ({ taskName, modelVersion }: { taskName: string; modelVersion: string }) => {
      if (!projectId) throw new Error("Project context is missing.");
      return addProjectArtifact(projectId, {
        kind: "model_bundle",
        task_name: taskName,
        artifact_id: modelVersion,
      });
    },
    onSuccess: (project) => {
      qc.setQueryData(["project", project.project_id], project);
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project-summary", project.project_id] });
    },
  });
  const linkResultBundle = (taskName: string, result: Record<string, unknown>) => {
    if (typeof result.model_version !== "string" || !result.model_version) return;
    linkProjectBundleMut.mutate({ taskName, modelVersion: result.model_version });
  };

  const backendOptionsQ = useQuery({
    queryKey: ["scoring-options"],
    queryFn: getScoringOptions,
  });
  const availableBackends = backendOptionsQ.data?.backends ?? ["local"];

  const presetsQ = useQuery({ queryKey: ["experiment-presets"], queryFn: getPresets, staleTime: Infinity });
  const presetInfo = presetsQ.data?.presets.find((p) => p.value === preset);
  const parsed = useMemo(() => parseSpace(configJson), [configJson]);
  const warnings = useMemo(
    () => ("error" in parsed ? [] : spaceWarnings(parsed, parsed.forbidden)),
    [parsed],
  );

  const runMut = useMutation({
    mutationFn: async () => {
      if ("error" in parsed) throw new Error(parsed.error);
      const body: RunExperimentRequest = {
        config: parsed.config,
        length,
        max_generations: maxGeneration,
        num_parents: numParents,
        train_partition: trainPartition,
        scoring,
        surrogate_mode: strategy === "surrogate",
        surrogate_itrs: strategy === "surrogate" ? surrogateItrs : null,
        forbidden: parsed.forbidden,
        backend,
        strategy: engine,
        label: experimentLabel.trim() || null,
      };
      return { ...(await runExperiment(task, body)), taskName: task };
    },
    onSuccess: (res) => {
      setJobId(res.job_id);
      setJobTask(res.taskName);
      setSubmittedName(res.display_name ?? null);
      setResult(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const quickMut = useMutation({
    mutationFn: async () => ({
      ...(await runExperiment(task, {
        preset,
        strategy: "surrogate",
        length: budget,
        scoring: quickScoring || presetInfo?.default_scoring,
        random_state: 0,
        label: experimentLabel.trim() || null,
      })),
      taskName: task,
    }),
    onSuccess: (res) => {
      setJobId(res.job_id);
      setJobTask(res.taskName);
      setSubmittedName(res.display_name ?? null);
      setResult(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const optimizeMut = useMutation({
    mutationFn: async ({ seq }: { seq: string[] }) => ({
      ...(await optimizePipeline(task, { seq, label: experimentLabel.trim() || null })),
      taskName: task,
    }),
    onSuccess: (res, vars) => {
      setPipelineJobs((prev) => [
        ...prev,
        { jobId: res.job_id, seq: vars.seq, taskName: res.taskName },
      ]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const onSubmit = () => {
    if ("error" in parsed) {
      setConfigError(parsed.error);
      return;
    }
    setConfigError(null);
    runMut.mutate();
  };

  /** Merge the pre-filled stacking directive (schema for the StackingEstimator wrappers). */
  const insertStackingComponent = () => {
    if ("error" in parsed) return;
    const { config } = parsed;
    if (config[STACKING_KEY]) return;
    const next = { ...config, ...STACKING_COMPONENT };
    setConfigJson(JSON.stringify(next, null, 2));
    setConfigError(null);
  };

  /** Append a placeholder forbidden rule the user can edit in place. */
  const insertForbiddenRule = () => {
    if ("error" in parsed) return;
    const rules = parsed.forbidden ?? [];
    const firstParam = Object.keys(parsed.paramsByEstimator)[0];
    const candidate = firstParam ? Object.keys(parsed.config[firstParam])[0] : "paramA";
    setConfigJson(
      JSON.stringify(
        { ...parsed.config, [FORBIDDEN_KEY]: [...rules, [candidate ?? "paramA", "value"]] },
        null,
        2,
      ),
    );
    setConfigError(null);
  };

  const historyPoints = (result?.evaluation_history ?? [])
    .filter((row) => typeof row.duration === "number" && typeof row.score === "number")
    .map((row) => ({ pipeline: row.pipeline, score: row.score as number, duration: row.duration as number }));

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Experiments</h1>
        <span className="desc">Find the best model for task “{task || "…"}” — runs as a background job</span>
      </div>

      {experimentActive && (
        <div className="run-banner" role="status" aria-live="polite">
          <span className={`pulse${activeJob?.status === "queued" ? " queued" : ""}`} />
          <strong>{activeJob?.status === "queued" ? "Experiment queued" : "Experiment running"}</strong>
          {(submittedName ?? activeJob?.displayName) && (
            <span className="muted">“{submittedName ?? activeJob?.displayName}”</span>
          )}
          <span className="muted">
            {fmtElapsed(activeElapsed)} elapsed{jobId ? ` · job ${jobId.slice(0, 8)}…` : ""}
          </span>
          <span className="spacer" />
          <button className="btn" type="button" onClick={scrollToTracker}>
            Live progress
          </button>
          {jobId && (
            <Link className="btn" to={`/logs?job=${jobId}`}>
              Logs
            </Link>
          )}
        </div>
      )}

      {projectId && (
        <Card title="Research project">
          {projectQ.isLoading && <p className="muted">Loading project…</p>}
          {projectQ.isError && <ErrorNote error={projectQ.error} />}
          {projectQ.data && (
            <p className="muted">
              Completed jobs and produced model bundles will be linked to “{projectQ.data.name}”.
            </p>
          )}
          {linkProjectJobMut.isError && <ErrorNote error={linkProjectJobMut.error} />}
          {linkProjectBundleMut.isError && <ErrorNote error={linkProjectBundleMut.error} />}
        </Card>
      )}

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      <div className="field" style={{ maxWidth: 420 }}>
        <label>Experiment name (optional)</label>
        <input
          value={experimentLabel}
          onChange={(e) => setExperimentLabel(e.target.value)}
          placeholder={task ? `${task} experiment` : "e.g. baseline EOA"}
          maxLength={80}
          spellCheck={false}
        />
        <span className="hint">
          Shown instead of the job id in the tracker, the Jobs page and Logs. The produced model
          bundle stores the same name.
        </span>
      </div>

      <Tabs
        tabs={[
          { id: "quick", label: "Quick start" },
          { id: "advanced", label: "Advanced" },
        ]}
        active={mode}
        onChange={(id) => setMode(id as "quick" | "advanced")}
      />

      {mode === "quick" && (
        <Card title="Quick start" sub="Pick the kind of problem and a time budget. No search space to write — a bounded, safe search is used.">
          <div className="row">
            <div className="field">
              <label>Problem type</label>
              <select value={preset} onChange={(e) => setPreset(e.target.value as "classification" | "regression")}>
                {(presetsQ.data?.presets ?? []).map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
              {presetInfo && <span className="hint">{presetInfo.description}</span>}
            </div>
            <div className="field">
              <label>Budget</label>
              <select value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
                <option value={1}>Fast (1 pipeline stage)</option>
                <option value={2}>Balanced (up to 2 stages)</option>
                <option value={3}>Thorough (up to 3 stages)</option>
              </select>
            </div>
            <ScoringSelect
              value={quickScoring || presetInfo?.default_scoring || "accuracy"}
              onChange={setQuickScoring}
              disabled={!task}
            />
          </div>
          {quickMut.isError && <ErrorNote error={quickMut.error} />}
          <button
            className="btn primary"
            disabled={!task || quickMut.isPending || experimentActive}
            onClick={() => quickMut.mutate()}
          >
            {quickMut.isPending ? "Submitting…" : experimentActive ? "Experiment running…" : "Find best model"}
          </button>
          <p className="muted">
            Need your own search space, stacking or forbidden rules? Switch to the <b>Advanced</b> tab.
          </p>
        </Card>
      )}

      {mode === "advanced" && (<>
      <Card title="Search space" sub="Keys are fully-qualified estimator class names; values map hyperparameters to real / integer / categorical specs. Optional per-param “depends_on” and top-level “forbidden” rules shape the search.">
        <textarea
          className="mono"
          rows={16}
          value={configJson}
          onChange={(e) => setConfigJson(e.target.value)}
          spellCheck={false}
        />
        <div className="row fixed" style={{ marginTop: 8 }}>
          <button className="btn" onClick={insertStackingComponent} disabled={"error" in parsed}>
            Add stacking component
          </button>
          <button className="btn" onClick={insertForbiddenRule} disabled={"error" in parsed}>
            Add forbidden rule
          </button>
          <span className="hint">
            <code>stacking</code> configures the out-of-fold <code>StackingEstimator</code> wrappers for intermediate
            components (<code>res</code>/<code>probs</code>/<code>decision</code>/<code>cv</code>/<code>n_jobs</code>);
            it is a directive, not a pipeline step.
          </span>
        </div>
        {warnings.length > 0 && (
          <div style={{ marginTop: 8 }}>
            {warnings.map((w) => (
              <div key={w} className="hint">
                ⚠ {w}
              </div>
            ))}
          </div>
        )}
        {configError && (
          <div className="error-note" style={{ marginTop: 8 }}>
            {configError}
          </div>
        )}
      </Card>

      <Card
        title="Run settings"
        sub="Each field is one column of the row; the scoring metric becomes the objective the AML/EOA search maximizes."
      >
        <div className="row">
          <div className="field">
            <label>Max pipeline length</label>
            <input type="number" min={1} value={length} onChange={(e) => setLength(Number(e.target.value))} />
            <span className="hint">Maximum number of components in a generated pipeline — not a trial count.</span>
          </div>
          <div className="field">
            <label>Max generation</label>
            <input type="number" min={1} value={maxGeneration} onChange={(e) => setMaxGeneration(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Num parents</label>
            <input type="number" min={2} value={numParents} onChange={(e) => setNumParents(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Train partition</label>
            <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
          </div>
          <ScoringSelect value={scoring} onChange={setScoring} disabled={!task} />
          <div className="field">
            <label>Search engine</label>
            <select value={engine} onChange={(e) => setEngine(e.target.value as "eoa" | "surrogate")}>
              <option value="eoa">Evolutionary (EOA)</option>
              <option value="surrogate">Surrogate only (no evolution)</option>
            </select>
            {engine === "surrogate" && <span className="hint">Generation and parent settings are ignored.</span>}
          </div>
          <div className="field">
            <label>Execution backend</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value as "local" | "dask")}>
              <option value="local">Local</option>
              <option value="dask" disabled={!availableBackends.includes("dask")}>Dask</option>
            </select>
            {backendOptionsQ.isError && (
              <span className="hint">Could not load backend availability; local execution remains available.</span>
            )}
            {backendOptionsQ.data && !availableBackends.includes("dask") && (
              <span className="hint">Dask is unavailable on this server; install dask[distributed] to enable it.</span>
            )}
          </div>
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <div className="field">
            <label>Search strategy</label>
            <span className="row fixed">
              <input type="radio" name="strategy" checked={strategy === "eoa"} onChange={() => setStrategy("eoa")} />
              <span>Evolutionary (EOA)</span>
              <input type="radio" name="strategy" checked={strategy === "surrogate"} onChange={() => setStrategy("surrogate")} />
              <span>Surrogate-assisted</span>
            </span>
          </div>
          {strategy === "surrogate" && (
            <>
              <div className="field">
                <label>Surrogate iterations</label>
                <input type="number" min={1} value={surrogateItrs} onChange={(e) => setSurrogateItrs(Number(e.target.value))} />
              </div>
              <span className="hint">
                A single lightweight regressor replaces the default KRR+GPR pair; more iterations improve the
                surrogate fit but increase runtime per generation.
              </span>
            </>
          )}
        </div>

        {runMut.isError && <ErrorNote error={runMut.error} />}

        <button className="btn primary" disabled={!task || runMut.isPending || experimentActive} onClick={onSubmit}>
          {runMut.isPending ? "Submitting…" : experimentActive ? "Experiment running…" : "Start experiment"}
        </button>
      </Card>
      </>)}

      {jobId && (
        <div id="experiment-job">
          <JobTracker
            jobId={jobId}
            onStatus={handleJobStatus}
            onResume={(resumedId) => {
              // Resume submits a new job that continues from the old
              // checkpoint; follow that job rather than the paused record.
              setJobId(resumedId);
              setResult(null);
              setActiveJob(null);
            }}
            onResult={(r) => {
              setResult(r as unknown as ExperimentResult);
              if (projectId && jobTask) {
                linkProjectJobMut.mutate({
                  completedJobId: jobId,
                  taskName: jobTask,
                  label: "experiment",
                });
                linkResultBundle(jobTask, r);
              }
            }}
          />
          <p className="muted">
            On completion the job result contains <code>model_version</code>, <code>train_score</code>,{" "}
            <code>top_pipelines</code>, <code>pareto</code> and the full <code>evaluation_history</code>. The new
            bundle appears under Model Bundles.
          </p>
        </div>
      )}

      {result && (
        <>
          <Card title="Top pipelines" sub="Best stored pipeline structures from this search — optimize any one of them in place.">
            <Table
              columns={["Pipeline", "Score", ""]}
              rows={result.top_pipelines ?? []}
              keyOf={(row, i) => `${i}-${row.pipeline.join(",")}`}
              render={(row) => [
                <td key="p" className="mono">
                  {row.pipeline.map(shortName).join(" → ")}
                </td>,
                <td key="s">{fmtNum(row.score)}</td>,
                <td key="a">
                  <button
                    className="btn"
                    disabled={optimizeMut.isPending}
                    onClick={() => optimizeMut.mutate({ seq: row.pipeline })}
                  >
                    Optimize this structure
                  </button>
                </td>,
              ]}
            />
            {optimizeMut.isError && <ErrorNote error={optimizeMut.error} />}
          </Card>

          {pipelineJobs.map((pj) => (
            <JobTracker
              key={pj.jobId}
              jobId={pj.jobId}
              onResult={(pipelineResult) => {
                if (projectId) {
                  linkProjectJobMut.mutate({
                    completedJobId: pj.jobId,
                    taskName: pj.taskName,
                    label: "optimize_pipeline",
                  });
                  linkResultBundle(pj.taskName, pipelineResult);
                }
              }}
            />
          ))}

          {historyPoints.length > 0 && (
            <Card
              title="Pareto frontier"
              sub="Score vs. per-candidate duration; highlighted points are non-dominated (best score for their cost)."
            >
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 16, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="duration"
                      name="Duration (s)"
                      domain={["auto", "auto"]}
                      tickFormatter={(v: number) => Number(v).toFixed(1)}
                    />
                    <YAxis type="number" dataKey="score" name="Score" domain={["auto", "auto"]} width={64} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value) => Number(value).toFixed(4)} />
                    <Scatter name="Candidates" data={historyPoints} fill="#9ca3af" opacity={0.55} />
                    {(result.pareto ?? []).length > 0 && (
                      <Scatter name="Frontier" data={result.pareto ?? []} fill="#4f8cff" />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              {(result.pareto ?? []).length > 0 && (
                <Table
                  columns={["Pipeline", "Score", "Duration (s)"]}
                  rows={result.pareto ?? []}
                  keyOf={(row, i) => `${i}-${row.pipeline.join(",")}`}
                  render={(row) => [
                    <td key="p" className="mono">
                      {row.pipeline.map(shortName).join(" → ")}
                    </td>,
                    <td key="s">{fmtNum(row.score)}</td>,
                    <td key="d">{fmtNum(row.duration)}</td>,
                  ]}
                />
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
