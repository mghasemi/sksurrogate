import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  getDatasetMetadata,
  getPreprocessing,
  latinHypercube,
  suggestNextPoint,
} from "../api/client";
import type { DesignVariable } from "../api/client";
import { Card, ErrorNote, JsonView, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

function toCsv(columns: string[], rows: Array<Array<number | string>>): string {
  return [columns.join(","), ...rows.map((r) => r.join(","))].join("\n");
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** Parse "x1,x2,...,y" rows; the last column is the observed objective. */
function parseObservations(text: string, dims: number): { samples: number[][]; values: number[] } {
  const rows = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => l.split(",").map((c) => Number(c.trim())));
  if (rows.some((r) => r.length !== dims + 1 || r.some((v) => !Number.isFinite(v)))) {
    throw new Error(`Each row needs ${dims} input value(s) followed by the measured result, all numeric.`);
  }
  // The surrogate is fitted on observations, so one run cannot be extrapolated
  // from. Say so here instead of letting the API answer 422.
  if (rows.length < 2) {
    throw new Error("Enter at least two runs — the surrogate needs two observations to suggest the next point.");
  }
  return { samples: rows.map((r) => r.slice(0, dims)), values: rows.map((r) => r[dims]) };
}

export default function DesignPage() {
  const { task } = useTask();
  const [variables, setVariables] = useState<DesignVariable[]>([
    { name: "x1", low: 0, high: 1 },
    { name: "x2", low: 0, high: 1 },
  ]);
  const [n, setN] = useState(10);
  const [observations, setObservations] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [partition, setPartition] = useState("train");
  const [chosenColumns, setChosenColumns] = useState<string[]>([]);

  const metaQ = useQuery({
    queryKey: ["design-metadata", task],
    queryFn: async () => {
      try {
        return await getDatasetMetadata(task);
      } catch {
        return null;
      }
    },
    enabled: !!task,
    retry: false,
  });
  const partitions = useMemo(() => Object.keys(metaQ.data?.partitions ?? {}), [metaQ.data]);
  useEffect(() => {
    if (partitions.length && !partitions.includes(partition)) setPartition(partitions[0]);
  }, [partitions, partition]);

  const datasetQ = useQuery({
    queryKey: ["design-preprocessing", task, partition],
    queryFn: () => getPreprocessing(task, partition),
    enabled: !!task && partitions.includes(partition),
  });
  const numericColumns = useMemo(
    () =>
      (datasetQ.data?.missing.columns ?? []).filter(
        (column) =>
          !column.is_target &&
          !column.ignored &&
          column.numeric_summary !== null &&
          column.numeric_summary.min !== null &&
          column.numeric_summary.max !== null &&
          column.numeric_summary.min < column.numeric_summary.max,
      ),
    [datasetQ.data],
  );
  useEffect(() => {
    setChosenColumns([]);
  }, [task, partition]);

  const importRanges = () => {
    const selected = numericColumns.filter((column) => chosenColumns.includes(column.column));
    if (!selected.length) return;
    setVariables(
      selected.map((column) => ({
        name: column.column,
        low: Number(column.numeric_summary?.min),
        high: Number(column.numeric_summary?.max),
      })),
    );
  };

  const update = (i: number, patch: Partial<DesignVariable>) =>
    setVariables((v) => v.map((row, idx) => (idx === i ? { ...row, ...patch } : row)));

  const lhsMut = useMutation({ mutationFn: () => latinHypercube({ variables, n_samples: n }) });
  const validateDesign = () => {
    // Catch what pydantic would reject with a 422, in the page's own words.
    const names = variables.map((v) => v.name.trim());
    if (names.some((n) => !n)) throw new Error("Every variable needs a name.");
    if (new Set(names).size !== names.length) throw new Error("Variable names must be unique.");
    variables.forEach((v, i) => {
      if (!Number.isFinite(v.low) || !Number.isFinite(v.high)) {
        throw new Error(`Variable ${names[i] || i + 1} has a non-numeric bound.`);
      }
      if (!(v.low < v.high)) {
        throw new Error(`Variable ${names[i] || i + 1}: low must be smaller than high.`);
      }
    });
    return variables.map((v, i) => ({ name: names[i], low: v.low, high: v.high }));
  };

  const nextMut = useMutation({
    mutationFn: () => {
      const checked = validateDesign();
      const { samples, values } = parseObservations(observations, checked.length);
      return suggestNextPoint({ variables: checked, samples, values });
    },
  });

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Experiment design</h1>
        <span className="desc">
          Plan which experiments to run next. Continuous numeric variables only; this proposes points, it does not run them.
        </span>
      </div>

      <Card
        title="What this page is for"
        sub="Design of experiments (DoE): choose which runs to perform next, not which model to train."
      >
        <p className="muted" style={{ marginTop: 0 }}>
          Use this page when each run is expensive or slow — a physical experiment, an instrument
          calibration, a simulation sweep — and you must decide the next input combination. It fits
          a Gaussian-process surrogate to the results you paste in and proposes a point that
          balances a low predicted objective with high uncertainty. For hyperparameter search or
          model selection on a registered dataset, use <b>Experiments</b> instead; this page does
          not train, score or register models.
        </p>
        <ol style={{ margin: "0 0 0 18px", lineHeight: 1.65 }}>
          <li>
            <b>Variables</b> — list the continuous inputs you control and their safe bounds. With a
            task selected you can import the numeric columns and their observed ranges.
          </li>
          <li>
            <b>Initial plan</b> — generate a space-filling Latin-hypercube design, download it, and
            run those points.
          </li>
          <li>
            <b>Suggest the next experiment</b> — paste one row per completed run as{" "}
            <code>input1, input2, …, measured_result</code>. The suggestion <b>minimizes</b> the
            result, so encode “larger is better” metrics with a negative sign. At least two runs are
            required; repeat the loop after each measurement.
          </li>
        </ol>
        <p className="hint">
          The page only proposes points; you decide whether and how to measure them. Observed dataset
          ranges are descriptive, not proof that extrapolation is physically safe — widen or tighten
          bounds using domain knowledge, and enforce engineering, ethical and safety constraints
          yourself. Categorical factors, constraints, multi-objective goals and noisy replications
          are not supported yet.
        </p>
      </Card>

      {task && (
        <Card
          title="Use ranges from the current dataset"
          sub={`Optional shortcut: seed the design variables from numeric features of task “${task}”. The target column and ignored columns are excluded.`}
        >
          {metaQ.isLoading && <span className="muted">Loading dataset…</span>}
          {!metaQ.isLoading && partitions.length === 0 && (
            <p className="muted">
              No dataset is registered for this task. Register one on the Datasets page, or define
              variables manually below.
            </p>
          )}
          {partitions.length > 0 && (
            <>
              <div className="row top">
                <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
                  <label>Partition</label>
                  <select value={partition} onChange={(e) => setPartition(e.target.value)}>
                    {partitions.map((name) => <option key={name} value={name}>{name}</option>)}
                  </select>
                </div>
                <div className="field fixed" style={{ flex: "0 0 auto" }}>
                  <label aria-hidden="true">&nbsp;</label>
                  <button
                    className="btn"
                    disabled={chosenColumns.length === 0}
                    onClick={importRanges}
                  >
                    Use selected ranges as variables
                  </button>
                </div>
              </div>
              {datasetQ.isLoading && <span className="muted">Profiling numeric columns…</span>}
              {datasetQ.isError && <ErrorNote error={datasetQ.error} />}
              {datasetQ.data && numericColumns.length === 0 && (
                <p className="muted">
                  No variable-range numeric feature column is available (columns are categorical,
                  constant, ignored, or the target).
                </p>
              )}
              {numericColumns.length > 0 && (
                <>
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th />
                        <th>Column</th>
                        <th>Observed min</th>
                        <th>Observed max</th>
                        <th>Missing</th>
                      </tr>
                    </thead>
                    <tbody>
                      {numericColumns.map((column) => (
                        <tr key={column.column}>
                          <td>
                            <input
                              type="checkbox"
                              aria-label={`Design on ${column.column}`}
                              checked={chosenColumns.includes(column.column)}
                              onChange={() =>
                                setChosenColumns((current) =>
                                  current.includes(column.column)
                                    ? current.filter((name) => name !== column.column)
                                    : [...current, column.column],
                                )
                              }
                            />
                          </td>
                          <td className="mono">{column.column}</td>
                          <td>{fmtNum(column.numeric_summary?.min)}</td>
                          <td>{fmtNum(column.numeric_summary?.max)}</td>
                          <td>{fmtNum(column.missing)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="hint">
                    Importing replaces the variable list with the selected columns and their observed
                    ranges. Constant columns are omitted. Consider widening the upper bound beyond the
                    observed maximum when the current data never explored that region.
                  </p>
                </>
              )}
            </>
          )}
        </Card>
      )}

      <Card title="1. Variables" sub="The inputs you can control and their allowed ranges. Edit imported ranges before generating a plan.">
        <div className="stack">
          {variables.map((v, i) => (
            <div className="row" key={i}>
              <div className="field">
                <label>Name</label>
                <input value={v.name} onChange={(e) => update(i, { name: e.target.value })} />
              </div>
              <div className="field">
                <label>Low</label>
                <input
                  type="number"
                  value={Number.isFinite(v.low) ? v.low : ""}
                  onChange={(e) => update(i, { low: Number(e.target.value) })}
                />
              </div>
              <div className="field">
                <label>High</label>
                <input
                  type="number"
                  value={Number.isFinite(v.high) ? v.high : ""}
                  onChange={(e) => update(i, { high: Number(e.target.value) })}
                />
              </div>
              <button
                className="btn"
                disabled={variables.length <= 1}
                onClick={() => setVariables((vs) => vs.filter((_, idx) => idx !== i))}
              >
                Remove
              </button>
            </div>
          ))}
          <div>
            <button
              className="btn"
              onClick={() => setVariables((vs) => [...vs, { name: `x${vs.length + 1}`, low: 0, high: 1 }])}
            >
              Add variable
            </button>
          </div>
        </div>
      </Card>

      <Card title="2. Initial plan" sub="A space-filling Latin-hypercube design: every range is covered evenly.">
        <div className="row fixed">
          <div className="field">
            <label>Number of experiments</label>
            <input type="number" min={1} max={10000} value={n} onChange={(e) => setN(Number(e.target.value))} />
          </div>
          <button className="btn primary" disabled={lhsMut.isPending} onClick={() => lhsMut.mutate()}>
            Generate plan
          </button>
          {lhsMut.data && (
            <button
              className="btn"
              onClick={() => download("design.csv", toCsv(lhsMut.data.columns, lhsMut.data.points))}
            >
              Download CSV
            </button>
          )}
        </div>
        {lhsMut.isError && <ErrorNote error={lhsMut.error} />}
        {lhsMut.data && (
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>{lhsMut.data.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {lhsMut.data.points.slice(0, 50).map((row, i) => (
                  <tr key={i}>{row.map((v, j) => <td key={j}>{Number(v).toPrecision(5)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="3. Suggest the next experiment"
        sub={`Paste results so far, one run per line: ${variables.map((v) => v.name).join(", ")}, result. The suggestion minimizes the result.`}
      >
        <textarea
          className="mono"
          rows={6}
          value={observations}
          onChange={(e) => setObservations(e.target.value)}
          placeholder={"0.2,0.7,1.35\n0.8,0.1,0.42"}
          spellCheck={false}
        />
        <button
          className="btn primary"
          disabled={nextMut.isPending}
          onClick={() => {
            setParseError(null);
            try {
              parseObservations(observations, variables.length);
            } catch (e) {
              setParseError(e instanceof Error ? e.message : String(e));
              return;
            }
            nextMut.mutate();
          }}
        >
          Suggest next point
        </button>
        {parseError && <div className="error-note">{parseError}</div>}
        {nextMut.isError && <ErrorNote error={nextMut.error} />}
        {nextMut.data && <JsonView data={nextMut.data} />}
      </Card>
    </div>
  );
}
