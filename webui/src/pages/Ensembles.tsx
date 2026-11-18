import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import {
  fitEnsemble,
  getEnsemble,
  getEnsembleCandidates,
  getEnsembleMethods,
  getScoringOptions,
  listEnsembles,
} from "../api/client";
import type {
  EnsembleCalibrationMember,
  EnsembleCandidate,
  EnsembleDetail,
  EnsembleParameterSpec,
  EnsembleRecord,
} from "../api/client";
import { Badge, Card, EmptyState, ErrorNote, Loading, Table, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { ScoringSelect } from "../components/ScoringSelect";
import { useTask } from "../lib/task-context";

/** Compact "0.9312 ± 0.0040" rendering for a cross-validated score. */
function scoreText(mean: number | null | undefined, std?: number | null) {
  if (mean === null || mean === undefined) return "—";
  return std === null || std === undefined ? fmtNum(mean) : `${fmtNum(mean)} ± ${fmtNum(std)}`;
}

/**
 * Render one parameter control from the server's schema, so the page never
 * hard-codes a combiner's options: a change in
 * ``SKSurrogate.ensembling.METHOD_PARAMETERS`` shows up here automatically.
 */
function ParameterField({
  spec,
  value,
  onChange,
  disabled,
}: {
  spec: EnsembleParameterSpec;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}) {
  const width = { flex: "0 1 190px", minWidth: 150 };
  if (spec.type === "boolean") {
    return (
      <label
        className="field"
        style={{ display: "flex", gap: 6, alignItems: "center", alignSelf: "flex-end", marginBottom: 6 }}
        title={spec.help}
      >
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        {spec.label ?? spec.name}
      </label>
    );
  }
  return (
    <div className="field fixed" style={width} title={spec.help}>
      <label>{spec.label ?? spec.name}</label>
      {spec.choices ? (
        <select
          value={String(value ?? "")}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          {(spec.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      ) : spec.type === "integer" || spec.type === "float" ? (
        <input
          type="number"
          step={spec.type === "integer" ? 1 : "any"}
          min={spec.min}
          max={spec.max}
          disabled={disabled}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        />
      ) : spec.type === "floats" ? (
        // Weights are positional, so they are typed as a comma-separated list in
        // member order; an empty box means equal weights.
        <input
          disabled={disabled}
          value={Array.isArray(value) ? (value as number[]).join(", ") : ""}
          placeholder="equal weights"
          onChange={(e) =>
            onChange(
              e.target.value
                .split(",")
                .map((item) => item.trim())
                .filter((item) => item.length > 0)
                .map(Number),
            )
          }
        />
      ) : (
        <input value={String(value ?? "")} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
      )}
    </div>
  );
}

export default function EnsemblesPage() {
  const { task } = useTask();
  const qc = useQueryClient();
  const navigate = useNavigate();

  const [selected, setSelected] = useState<string[]>([]);
  const [method, setMethod] = useState("voting");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [scoring, setScoring] = useState("accuracy");
  const [label, setLabel] = useState("");
  const [register, setRegister] = useState(false);
  // The two orthogonal refinements, kept as plain on/off + settings so the page always
  // states what it will send (the server validates the same values again).
  const [calibrationOn, setCalibrationOn] = useState(false);
  const [calibration, setCalibration] = useState({ method: "sigmoid", cv: 5, ensemble: "auto" });
  const [weightsOn, setWeightsOn] = useState(false);
  const [weights, setWeights] = useState({ strategy: "least_squares", samples: 200, rounds: 20 });
  const [honestOn, setHonestOn] = useState(false);
  const [honest, setHonest] = useState({ folds: 5, inner_folds: 3 });
  const [jobId, setJobId] = useState<string | null>(null);
  const [detailVersion, setDetailVersion] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const methodsQ = useQuery({
    queryKey: ["ensemble-methods", task],
    queryFn: () => getEnsembleMethods(task),
    enabled: !!task,
  });
  const candidatesQ = useQuery({
    queryKey: ["ensemble-candidates", task],
    queryFn: () => getEnsembleCandidates(task),
    enabled: !!task,
    refetchInterval: jobId ? 3000 : false,
    // A combination job runs in a background tab as often as a foreground one; without
    // this the member list (and the ensembles list below) stops refreshing when the tab
    // is hidden and the page reads as stuck while the job has actually finished.
    refetchIntervalInBackground: true,
  });
  const ensemblesQ = useQuery({
    queryKey: ["ensembles", task],
    queryFn: () => listEnsembles(task),
    enabled: !!task,
    // While a combination is running the list gains a row, so poll rather than
    // leaving the page showing a stale set.
    refetchInterval: jobId ? 3000 : false,
    refetchIntervalInBackground: true,
  });
  const detailQ = useQuery({
    queryKey: ["ensemble", task, detailVersion],
    queryFn: () => getEnsemble(task, detailVersion as string),
    enabled: !!task && !!detailVersion,
  });
  const scoringQ = useQuery({ queryKey: ["scoring-options"], queryFn: getScoringOptions, staleTime: Infinity });

  // A new task means a new member pool; carrying the previous selection over would
  // request members that do not exist for this task.
  useEffect(() => {
    setSelected([]);
    setDetailVersion(null);
    setJobId(null);
  }, [task]);

  const activeMethod = useMemo(
    () => methodsQ.data?.methods.find((entry) => entry.value === method),
    [methodsQ.data, method],
  );

  /**
   * Weight optimization and an explicit weight vector are alternatives — the server
   * refuses both together — so the page offers only one: enabling optimization disables
   * the typed vector, and a typed vector disables the optimizer.
   */
  const hasExplicitWeights = useMemo(
    () =>
      Object.entries(params).some(
        ([name, value]) => name === "weights" && Array.isArray(value) && (value as number[]).length > 0,
      ),
    [params],
  );

  // Parameters start at the method's defaults (a stacking meta-learner, a boosting
  // round count) so a user can submit without visiting every field.
  useEffect(() => {
    if (!activeMethod) return;
    setParams((current) => {
      const next: Record<string, unknown> = {};
      for (const spec of activeMethod.parameters) {
        const kept = current[spec.name];
        next[spec.name] =
          kept === undefined || kept === null || (Array.isArray(kept) && kept.length === 0)
            ? spec.default
            : kept;
      }
      return next;
    });
  }, [activeMethod]);

  const selectedMembers = useMemo(
    () => (candidatesQ.data?.candidates ?? []).filter((entry) => selected.includes(entry.model_version)),
    [candidatesQ.data, selected],
  );

  const families = useMemo(
    () => new Set(selectedMembers.map((entry) => entry.family).filter(Boolean)),
    [selectedMembers],
  );

  // Stacking learns its own combination in the final estimator, so a member weight
  // vector would be ignored (SKSurrogate.validate_combination refuses it). Calibration
  // maps scores to probabilities, so the server publishes its availability from the
  // task's own family — keying off the *selected* members instead left the control
  // disabled until a member was ticked, which reads as "unavailable".
  const weightsAllowed =
    Boolean(methodsQ.data?.weight_optimization.available) && (activeMethod?.accepts_weights ?? true);
  const calibrationAllowed = Boolean(methodsQ.data?.calibration.available);

  useEffect(() => {
    if (!weightsAllowed) setWeightsOn(false);
    if (!calibrationAllowed) setCalibrationOn(false);
  }, [weightsAllowed, calibrationAllowed]);

  const toggleMember = (version: string) => {
    setSelected((current) =>
      current.includes(version) ? current.filter((item) => item !== version) : [...current, version],
    );
    setFormError(null);
  };

  const fitMut = useMutation({
    mutationFn: () => {
      if (!activeMethod) throw new Error("Pick a combination method first.");
      if (selected.length < activeMethod.minimum_members) {
        throw new Error(`${activeMethod.label} needs at least ${activeMethod.minimum_members} member model(s).`);
      }
      if (families.size > 1) {
        throw new Error("All selected members must be of the same problem family.");
      }
      if (weightsOn && hasExplicitWeights) {
        throw new Error("Either type member weights or optimize them, not both.");
      }
      const cleaned: Record<string, unknown> = {};
      for (const spec of activeMethod.parameters) {
        const value = params[spec.name];
        if (value === undefined || value === null || value === "") continue;
        if (spec.type === "floats") {
          const weights = value as number[];
          if (weights.length === 0) continue;
          if (weights.length !== selected.length) {
            throw new Error(`${spec.label ?? spec.name} needs exactly ${selected.length} value(s), one per member.`);
          }
          if (weights.some((item) => !Number.isFinite(item))) {
            throw new Error(`${spec.label ?? spec.name} must contain only numbers.`);
          }
        }
        cleaned[spec.name] = value;
      }
      return fitEnsemble(task, {
        members: selected,
        method,
        params: cleaned,
        scoring,
        label: label.trim() || null,
        register,
        calibration: calibrationOn ? { ...calibration } : null,
        weight_optimization: weightsOn ? { ...weights } : null,
        honest_cv: honestOn ? { ...honest } : null,
      });
    },
    onSuccess: (res) => {
      setJobId(res.job_id);
      setFormError(null);
      qc.invalidateQueries({ queryKey: ["ensembles", task] });
      qc.invalidateQueries({ queryKey: ["ensemble-candidates", task] });
    },
  });

  const onSubmit = () => {
    try {
      fitMut.mutate();
    } catch (error) {
      setFormError(error instanceof Error ? error.message : String(error));
    }
  };

  const ensembles = ensemblesQ.data?.ensembles ?? [];
  const candidates = candidatesQ.data?.candidates ?? [];

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Ensembles</h1>
        <span className="desc">
          Combine the trained models of task “{task || "…"}” with scikit-learn's own meta-estimators — voting,
          averaging, stacking, bagging or boosting. The result is an ordinary model bundle, so it can be
          evaluated, registered and served like any other.
        </span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {methodsQ.isError && <ErrorNote error={methodsQ.error} />}
      {methodsQ.data && (
        <div className="note">
          <strong>{methodsQ.data.family ?? "No model family detected"}</strong>
          {methodsQ.data.family
            ? ` — the combination methods below apply to this task's ${methodsQ.data.family.toLowerCase()} models.`
            : " — train at least one model for this task first (Experiments, or a baseline on Model Bundles)."}
        </div>
      )}

      <Card title="Members" sub="Select the models to combine. The order matters: weights are applied positionally.">
        {candidatesQ.isLoading ? (
          <Loading />
        ) : candidatesQ.isError ? (
          <ErrorNote error={candidatesQ.error} />
        ) : candidates.length === 0 ? (
          <EmptyState
            title="No models to combine yet"
            hint="Fit one from the Experiments page (or a baseline on Model Bundles), then come back."
          />
        ) : (
          <Table<EnsembleCandidate>
            columns={["", "#", "Model", "Family", "Kind", "Stored score", "Created", "Source"]}
            rows={candidates}
            keyOf={(entry) => entry.model_version}
            render={(entry) => {
              const position = selected.indexOf(entry.model_version);
              return [
                <td key="pick">
                  <input
                    type="checkbox"
                    checked={position >= 0}
                    onChange={() => toggleMember(entry.model_version)}
                    aria-label={`Select ${entry.model_version}`}
                  />
                </td>,
                <td key="pos" className="mono">
                  {position >= 0 ? position + 1 : "—"}
                </td>,
                <td key="model">
                  <div className="mono">{entry.model_version}</div>
                  {entry.label && <div className="muted">{entry.label}</div>}
                </td>,
                <td key="family">{entry.family ?? "—"}</td>,
                <td key="kind">
                  {entry.is_ensemble ? (
                    <Badge tone="info">ensemble · {entry.ensemble?.method}</Badge>
                  ) : (
                    <span className="muted">single model</span>
                  )}
                </td>,
                <td key="score" className="mono">
                  {scoreText(entry.metrics?.train_score ?? entry.metrics?.score ?? null)}
                </td>,
                <td key="created" className="mono">
                  {new Date(entry.created_at).toLocaleString()}
                </td>,
                <td key="source">{entry.source}</td>,
              ];
            }}
          />
        )}
      </Card>

      <Card
        title="Combination"
        sub={
          activeMethod
            ? `${activeMethod.description} Aggregation: ${activeMethod.aggregation}.`
            : "Pick a method."
        }
      >
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 190px", minWidth: 150 }}>
            <label>Method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {(methodsQ.data?.methods ?? []).map((entry) => (
                <option key={entry.value} value={entry.value} disabled={!entry.available}>
                  {entry.label}
                  {entry.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
          </div>
          <ScoringSelect value={scoring} onChange={setScoring} disabled={!task} style={{ flex: "0 1 190px", minWidth: 150 }} />
          <div className="field fixed" style={{ flex: "0 1 190px", minWidth: 150 }}>
            <label>Name (optional)</label>
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. lr + trees, soft" />
          </div>
          <label
            className="field"
            style={{ display: "flex", gap: 6, alignItems: "center", alignSelf: "flex-end", marginBottom: 6 }}
          >
            <input type="checkbox" checked={register} onChange={(e) => setRegister(e.target.checked)} />
            register as candidate
          </label>
        </div>

        {activeMethod && activeMethod.parameters.length > 0 && (
          <div className="row">
            {activeMethod.parameters.map((spec) => (
              <ParameterField
                key={spec.name}
                spec={spec}
                value={params[spec.name]}
                disabled={weightsOn && spec.type === "floats"}
                onChange={(value) => setParams((current) => ({ ...current, [spec.name]: value }))}
              />
            ))}
          </div>
        )}

        {selected.length > 0 && (
          <p className="muted">
            Selected in order:{" "}
            {selectedMembers.map((entry, index) => (
              <span key={entry.model_version}>
                {index > 0 ? ", " : ""}
                <code>
                  {index + 1}. {entry.model_version}
                </code>
              </span>
            ))}
            {families.size > 1 && <span className="error-note"> — mixed families; pick models of one family.</span>}
          </p>
        )}

        {scoringQ.data && !scoringQ.data.groups.some((group) => group.options.some((o) => o.value === scoring)) && (
          <p className="hint">The selected metric is not offered by this scikit-learn build.</p>
        )}

        {(formError || fitMut.isError) && <ErrorNote error={formError ?? fitMut.error ?? "Could not submit."} />}

        <button className="btn primary" disabled={!task || fitMut.isPending || candidates.length === 0} onClick={onSubmit}>
          {fitMut.isPending ? "Submitting…" : "Fit ensemble"}
        </button>
      </Card>

      {/*
        Both refinements and the honest estimate are rendered from the server's own
        schema (``methods.calibration`` / ``methods.weight_optimization``), so a new
        strategy or calibration method appears here without touching this file.
      */}
      <Card
        title="Refinements"
        sub="Optional stages around the combination. Each one is recorded with the model, so a score can always be traced to the procedure that produced it."
      >
        <div className="stack">
          <div className="row" style={{ alignItems: "flex-end", gap: 12 }}>
            <label className="field" style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}>
              <input
                type="checkbox"
                checked={calibrationOn}
                disabled={!calibrationAllowed}
                onChange={(e) => setCalibrationOn(e.target.checked)}
              />
              calibrate each member
            </label>
            <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
              <label>Calibration method</label>
              <select
                value={calibration.method}
                disabled={!calibrationOn}
                onChange={(e) => setCalibration({ ...calibration, method: e.target.value })}
              >
                {(methodsQ.data?.calibration.methods ?? [{ value: "sigmoid", label: "Sigmoid (Platt)", help: "" }]).map(
                  (option) => (
                    <option key={option.value} value={option.value} title={option.help}>
                      {option.label}
                    </option>
                  ),
                )}
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
              <label>Calibration folds</label>
              <input
                type="number"
                min={2}
                max={20}
                disabled={!calibrationOn}
                value={calibration.cv}
                onChange={(e) => setCalibration({ ...calibration, cv: Number(e.target.value) })}
              />
            </div>
            <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 130 }}>
              <label>Calibrators</label>
              <select
                value={calibration.ensemble}
                disabled={!calibrationOn}
                onChange={(e) => setCalibration({ ...calibration, ensemble: e.target.value })}
              >
                <option value="auto">auto</option>
                <option value="per_fold">one per fold</option>
                <option value="single">one on OOF scores</option>
              </select>
            </div>
          </div>
          {!calibrationAllowed && (
            <p className="hint">
              This task's models are {methodsQ.data?.family?.toLowerCase() ?? "not classified"} models. Per-member
              calibration maps a classifier's scores to probabilities, so it is available on classification tasks
              only — and the server refuses it for regression members.
            </p>
          )}

          <div className="row" style={{ alignItems: "flex-end", gap: 12 }}>
            <label
              className="field"
              style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}
              title={methodsQ.data?.weight_optimization.note}
            >
              <input
                type="checkbox"
                checked={weightsOn}
                disabled={!weightsAllowed || hasExplicitWeights}
                onChange={(e) => setWeightsOn(e.target.checked)}
              />
              optimize member weights
            </label>
            <div className="field fixed" style={{ flex: "0 1 210px", minWidth: 170 }}>
              <label>Strategy</label>
              <select
                value={weights.strategy}
                disabled={!weightsOn}
                onChange={(e) => setWeights({ ...weights, strategy: e.target.value })}
              >
                {(methodsQ.data?.weight_optimization.strategies ?? [
                  { value: "least_squares", label: "Least squares (Brier/MSE)", help: "" },
                ]).map((option) => (
                  <option key={option.value} value={option.value} title={option.help}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            {weights.strategy === "greedy" ? (
              <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
                <label>Rounds</label>
                <input
                  type="number"
                  min={1}
                  max={500}
                  disabled={!weightsOn}
                  value={weights.rounds}
                  onChange={(e) => setWeights({ ...weights, rounds: Number(e.target.value) })}
                />
              </div>
            ) : weights.strategy === "dirichlet" ? (
              <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
                <label>Samples</label>
                <input
                  type="number"
                  min={10}
                  max={5000}
                  disabled={!weightsOn}
                  value={weights.samples}
                  onChange={(e) => setWeights({ ...weights, samples: Number(e.target.value) })}
                />
              </div>
            ) : (
              <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
                <label>Cross-validation</label>
                <input value="out-of-fold" disabled readOnly />
              </div>
            )}
          </div>
          {!weightsAllowed && (
            <p className="hint">
              Weights merge members by a weighted vote or average, so they do not apply to stacking — it learns
              its own combination in the meta-learner.
            </p>
          )}
          {hasExplicitWeights && (
            <p className="hint">
              A weight vector is already typed in the parameters above. Clear it to let the optimizer fit one
              instead — the server refuses both at once.
            </p>
          )}

          <div className="row" style={{ alignItems: "flex-end", gap: 12 }}>
            <label
              className="field"
              style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}
              title="Refits the members, their calibration and the weights inside every outer fold."
            >
              <input type="checkbox" checked={honestOn} onChange={(e) => setHonestOn(e.target.checked)} />
              honest nested estimate
            </label>
            <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
              <label>Outer folds</label>
              <input
                type="number"
                min={2}
                max={20}
                disabled={!honestOn}
                value={honest.folds}
                onChange={(e) => setHonest({ ...honest, folds: Number(e.target.value) })}
              />
            </div>
            <div className="field fixed" style={{ flex: "0 1 110px", minWidth: 100 }}>
              <label>Inner folds</label>
              <input
                type="number"
                min={2}
                max={20}
                disabled={!honestOn}
                value={honest.inner_folds}
                onChange={(e) => setHonest({ ...honest, inner_folds: Number(e.target.value) })}
              />
            </div>
          </div>
          <p className="hint">
            {honestOn
              ? `Costs about ${honest.folds} × (members${calibrationOn ? " + calibration" : ""}${
                  weightsOn ? " + weight search" : ""
                }) fits, and reports the score of the procedure rather than of this one fitted model.`
              : "Without it, the ensemble's cross-validated score is the one the weight search was fitted against — jointly optimistic with the member scores next to it."}
          </p>
        </div>
      </Card>

      {jobId && (
        <Card title="Combination job">
          <JobTracker
            jobId={jobId}
            onResult={() => {
              qc.invalidateQueries({ queryKey: ["ensembles", task] });
              qc.invalidateQueries({ queryKey: ["ensemble-candidates", task] });
            }}
          />
          <p className="muted">
            On completion the job result names the new <code>model_version</code> and carries the member-versus-ensemble
            comparison. Evaluate it on the <a href="#" onClick={(e) => { e.preventDefault(); navigate("/evaluation"); }}>Evaluation</a>{" "}
            page — including the nested-CV estimate — or promote it under{" "}
            <a href="#" onClick={(e) => { e.preventDefault(); navigate("/registry"); }}>Registry</a>.
          </p>
        </Card>
      )}

      <Card
        title="Ensembles"
        sub={`Combinations stored for task “${task || "…"}”. Scores are cross-validated on identical folds per row.`}
      >
        {methodsQ.data?.caveats?.length ? (
          <div className="note">
            <strong>How to read these scores.</strong>
            <ul style={{ margin: "6px 0 0 18px" }}>
              {methodsQ.data.caveats.map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {ensemblesQ.isLoading ? (
          <Loading />
        ) : ensemblesQ.isError ? (
          <ErrorNote error={ensemblesQ.error} />
        ) : ensembles.length === 0 ? (
          <EmptyState title="No ensembles yet" hint="Select at least two models above and fit one." />
        ) : (
          <Table<EnsembleRecord>
            columns={["Created", "Model", "Method", "Members", "CV score", "Honest (nested)", "Refinements", "Train score", "Source", ""]}
            rows={ensembles}
            keyOf={(entry) => entry.model_version}
            render={(entry) => [
              <td key="c" className="mono">
                {new Date(entry.created_at).toLocaleString()}
              </td>,
              <td key="v">
                <div className="mono">{entry.model_version}</div>
                {entry.label && <div className="muted">{entry.label}</div>}
              </td>,
              <td key="m">
                {entry.method ?? "—"}
                {entry.aggregation && <div className="muted">{entry.aggregation}</div>}
              </td>,
              <td key="n" className="mono">
                {entry.member_count}
              </td>,
              <td key="cv" className="mono">
                {scoreText(
                  (entry.scores?.cv_score as number | null) ?? null,
                  (entry.scores?.cv_score_std as number | null) ?? null,
                )}
              </td>,
              // The honest number, when the fit asked for it: the score of the procedure
              // rather than of this one fitted model. A dash means it was not requested,
              // which is itself the caveat on the CV column.
              <td key="nested" className="mono">
                {entry.nested_score === null || entry.nested_score === undefined
                  ? "—"
                  : scoreText(entry.nested_score)}
              </td>,
              <td key="ref">
                {entry.calibrated ? <Badge tone="info">calibrated</Badge> : null}
                {entry.weighted ? <Badge tone="info">weighted</Badge> : null}
                {!entry.calibrated && !entry.weighted && <span className="muted">plain</span>}
              </td>,
              <td key="train" className="mono">
                {scoreText(entry.metrics?.train_score ?? null)}
              </td>,
              <td key="src">{entry.source}</td>,
              <td key="d">
                <button className="btn" onClick={() => setDetailVersion(entry.model_version)}>
                  Details
                </button>
              </td>,
            ]}
          />
        )}
      </Card>

      {detailVersion && (
        <Card
          title={`Members of ${detailVersion}`}
          sub="Each member was refitted inside every fold, so these are out-of-fold scores on the same folds the ensemble was scored on."
        >
          {detailQ.isLoading ? (
            <Loading />
          ) : detailQ.isError ? (
            <ErrorNote error={detailQ.error} />
          ) : detailQ.data ? (
            <EnsembleDetailView detail={detailQ.data} />
          ) : null}
        </Card>
      )}
    </div>
  );
}

/**
 * Say what the gap between the in-fold and the nested score means, without printing a
 * bare "0" for a gap that is zero to floating-point precision (the two protocols then
 * score the same estimator, which is a finding about the members, not a missing number).
 */
function nestedGapSentence(gap: number) {
  if (Math.abs(gap) < 5e-5) {
    return "The two agree here: with the weights refit inside every outer fold the procedure scores the same, so this fit carries no measurable optimism — which happens when the weight search collapses onto a single member and the ensemble is that member.";
  }
  if (gap > 0) {
    return `This procedure scores ${fmtNum(gap)} lower once the weights, the calibration and the members are refitted inside each outer fold; that difference is what the in-fold number does not pay for.`;
  }
  return `The nested estimate is ${fmtNum(-gap)} higher, which is fold noise rather than evidence that the procedure beats its own fit.`;
}

function EnsembleDetailView({ detail }: { detail: EnsembleDetail }) {
  const { ensemble } = detail;
  const weights = ensemble.weight_optimization?.weights ?? null;
  const rows: Array<{ key: string; name: string; note: string; weight: string; score: string }> = [
    ...ensemble.members.map((member, index) => ({
      key: `m-${member.model_version}`,
      name: member.model_version,
      note: [
        member.error ? `member — ${member.error}` : `member · ${member.source}${member.label ? ` · ${member.label}` : ""}`,
        member.calibrated ? "calibrated" : null,
        weights && weights[index] === 0 ? "dropped (weight 0)" : null,
      ]
        .filter(Boolean)
        .join(" · "),
      weight: weights ? fmtNum(weights[index]) : "—",
      score: scoreText(member.cv_mean, member.cv_std),
    })),
    {
      key: "ensemble",
      name: `ensemble (${ensemble.method})`,
      note: `${ensemble.aggregation} · scored by ${ensemble.scoring}`,
      weight: weights ? fmtNum(weights.reduce((total, value) => total + value, 0)) : "—",
      score: scoreText(ensemble.scores.cv_score, ensemble.scores.cv_score_std),
    },
  ];
  const parameters = Object.entries(ensemble.parameters ?? {}).filter(
    ([, value]) => value !== null && value !== undefined,
  );
  const calibration = ensemble.calibration;
  const nested = ensemble.scores.nested;
  return (
    <div className="stack">
      <Table<{ key: string; name: string; note: string; weight: string; score: string }>
        columns={weights ? ["Model", "Role", "Weight", "CV score"] : ["Model", "Role", "CV score"]}
        rows={rows}
        keyOf={(row) => row.key}
        render={(row) =>
          weights
            ? [
                <td key="n" className="mono">
                  {row.name}
                </td>,
                <td key="r">{row.note}</td>,
                <td key="w" className="mono">
                  {row.weight}
                </td>,
                <td key="s" className="mono">
                  {row.score}
                </td>,
              ]
            : [
                <td key="n" className="mono">
                  {row.name}
                </td>,
                <td key="r">{row.note}</td>,
                <td key="s" className="mono">
                  {row.score}
                </td>,
              ]
        }
      />

      {ensemble.weight_optimization && (
        <p className="muted">
          Weights fitted by <code>{ensemble.weight_optimization.strategy}</code> against{" "}
          {ensemble.weight_optimization.objective_name} → objective{" "}
          <span className="mono">{fmtNum(ensemble.weight_optimization.objective)}</span>
          {ensemble.weight_optimization.folds !== null && ` over ${ensemble.weight_optimization.folds} out-of-fold splits`}
          {ensemble.weight_optimization.dropped.length > 0 &&
            ` · dropped: ${ensemble.weight_optimization.dropped.join(", ")}`}
        </p>
      )}

      {/*
        Calibration is reported as before/after on identical folds, so the claim "this
        member is now calibrated" is a measured pair rather than an assertion.
      */}
      {calibration && (
        <>
          <h4 style={{ marginBottom: 4 }}>
            Per-member calibration ({calibration.method}, {calibration.cv} folds, {calibration.ensemble})
          </h4>
          <Table<EnsembleCalibrationMember>
            columns={["Member", "Brier before", "Brier after", "Log loss before", "Log loss after"]}
            rows={calibration.members}
            keyOf={(row) => row.member}
            render={(row) =>
              row.error ? (
                [
                  <td key="m" className="mono">
                    {row.member}
                  </td>,
                  <td key="e" colSpan={4}>
                    <span className="error-note">{row.error}</span>
                  </td>,
                ]
              ) : (
                [
                  <td key="m" className="mono">
                    {row.member}
                  </td>,
                  <td key="bb" className="mono">
                    {fmtNum(row.brier_before)}
                  </td>,
                  <td key="ba" className="mono">
                    {fmtNum(row.brier_after)}
                  </td>,
                  <td key="lb" className="mono">
                    {fmtNum(row.log_loss_before)}
                  </td>,
                  <td key="la" className="mono">
                    {fmtNum(row.log_loss_after)}
                  </td>,
                ]
              )
            }
          />
          <p className="hint">
            Lower is better. A large drop is the expected repair for an overconfident member;
            a well-calibrated member barely moves, and that is the honest outcome too.
          </p>
        </>
      )}

      {nested && (
        <div className="note">
          <strong>Honest nested estimate</strong> — {scoreText(nested.mean, nested.std)} by{" "}
          {nested.scoring}, {nested.outer_cv} outer × {nested.inner_cv} inner folds.
          {ensemble.scores.cv_score !== null && nested.mean !== null && (
            <>
              {" "}
              The optimistic in-fold score is {scoreText(ensemble.scores.cv_score, ensemble.scores.cv_score_std)}.{" "}
              {nestedGapSentence(ensemble.scores.cv_score - nested.mean)}
            </>
          )}
          {nested.errors.length > 0 && (
            <div className="error-note">
              {nested.errors.length} of the outer folds failed: {nested.errors[0].detail}
            </div>
          )}
          {nested.per_fold_weights && nested.per_fold_weights.length > 0 && (
            <p className="muted">
              Weight vector per outer fold:{" "}
              {nested.per_fold_weights
                .map((entry) => `#${entry.fold + 1} [${entry.weights.map((value) => fmtNum(value)).join(", ")}]`)
                .join(" · ")}
              {" — "}the spread shows how stable the search is across folds; a vector that
              moves fold to fold is a search that is fitting noise.
            </p>
          )}
        </div>
      )}

      {ensemble.caveats.length > 0 && (
        <ul className="muted" style={{ margin: "0 0 0 18px" }}>
          {ensemble.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      )}

      {parameters.length > 0 && (
        <p className="muted">
          Parameters:{" "}
          {parameters
            .map(([name, value]) => {
              const shown = Array.isArray(value)
                ? (value as unknown[]).map((item) => (typeof item === "number" ? fmtNum(item) : String(item))).join("/")
                : typeof value === "number"
                  ? fmtNum(value)
                  : String(value);
              return `${name}=${shown}`;
            })
            .join(", ")}
        </p>
      )}
      {ensemble.scores.cv_error && <div className="error-note">No cross-validated score: {ensemble.scores.cv_error}</div>}
      {ensemble.scores.train_score !== null && (
        <p className="hint">
          Refit training score {fmtNum(ensemble.scores.train_score)} — optimistic by construction; compare the CV column.
        </p>
      )}
    </div>
  );
}
