import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { latinHypercube, suggestNextPoint } from "../api/client";
import type { DesignVariable } from "../api/client";
import { Card, ErrorNote, JsonView } from "../components/ui";

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
  return { samples: rows.map((r) => r.slice(0, dims)), values: rows.map((r) => r[dims]) };
}

export default function DesignPage() {
  const [variables, setVariables] = useState<DesignVariable[]>([
    { name: "x1", low: 0, high: 1 },
    { name: "x2", low: 0, high: 1 },
  ]);
  const [n, setN] = useState(10);
  const [observations, setObservations] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);

  const update = (i: number, patch: Partial<DesignVariable>) =>
    setVariables((v) => v.map((row, idx) => (idx === i ? { ...row, ...patch } : row)));

  const lhsMut = useMutation({ mutationFn: () => latinHypercube({ variables, n_samples: n }) });
  const nextMut = useMutation({
    mutationFn: () => {
      const { samples, values } = parseObservations(observations, variables.length);
      return suggestNextPoint({ variables, samples, values });
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

      <Card title="1. Variables" sub="The inputs you can control and their allowed ranges.">
        <div className="stack">
          {variables.map((v, i) => (
            <div className="row" key={i}>
              <div className="field">
                <label>Name</label>
                <input value={v.name} onChange={(e) => update(i, { name: e.target.value })} />
              </div>
              <div className="field">
                <label>Low</label>
                <input type="number" value={v.low} onChange={(e) => update(i, { low: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label>High</label>
                <input type="number" value={v.high} onChange={(e) => update(i, { high: Number(e.target.value) })} />
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
