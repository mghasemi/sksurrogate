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
}: {
  spec: EnsembleParameterSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const width = { flex: "0 1 190px", minWidth: 150 };
  if (spec.type === "boolean") {
    return (
      <label
        className="field"
        style={{ display: "flex", gap: 6, alignItems: "center", alignSelf: "flex-end", marginBottom: 6 }}
        title={spec.help}
      >
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {spec.label ?? spec.name}
      </label>
    );
  }
  return (
    <div className="field fixed" style={width} title={spec.help}>
      <label>{spec.label ?? spec.name}</label>
      {spec.choices ? (
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
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
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        />
      ) : spec.type === "floats" ? (
        // Weights are positional, so they are typed as a comma-separated list in
        // member order; an empty box means equal weights.
        <input
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
        <input value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />
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
            columns={["Created", "Model", "Method", "Members", "CV score", "Train score", "Source", ""]}
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

function EnsembleDetailView({ detail }: { detail: EnsembleDetail }) {
  const { ensemble } = detail;
  const rows: Array<{ key: string; name: string; note: string; score: string }> = [
    ...ensemble.members.map((member) => ({
      key: `m-${member.model_version}`,
      name: member.model_version,
      note: member.error ? `member — ${member.error}` : `member · ${member.source}${member.label ? ` · ${member.label}` : ""}`,
      score: scoreText(member.cv_mean, member.cv_std),
    })),
    {
      key: "ensemble",
      name: `ensemble (${ensemble.method})`,
      note: `${ensemble.aggregation} · scored by ${ensemble.scoring}`,
      score: scoreText(ensemble.scores.cv_score, ensemble.scores.cv_score_std),
    },
  ];
  const parameters = Object.entries(ensemble.parameters ?? {}).filter(
    ([, value]) => value !== null && value !== undefined,
  );
  return (
    <div className="stack">
      <Table<{ key: string; name: string; note: string; score: string }>
        columns={["Model", "Role", "CV score"]}
        rows={rows}
        keyOf={(row) => row.key}
        render={(row) => [
          <td key="n" className="mono">
            {row.name}
          </td>,
          <td key="r">{row.note}</td>,
          <td key="s" className="mono">
            {row.score}
          </td>,
        ]}
      />
      {parameters.length > 0 && (
        <p className="muted">
          Parameters:{" "}
          {parameters
            .map(([name, value]) => `${name}=${Array.isArray(value) ? (value as unknown[]).join("/") : String(value)}`)
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
