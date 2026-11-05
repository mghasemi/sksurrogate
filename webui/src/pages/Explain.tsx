import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { explainBundle, getIntervals, listBundles } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Card, EmptyState, ErrorNote, Loading, Stat, fmtNum, fmtPct } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function ExplainPage() {
  const { task } = useTask();
  const [version, setVersion] = useState("");
  const [alpha, setAlpha] = useState(0.1);

  const bundlesQ = useQuery({
    queryKey: ["bundles", task],
    queryFn: () => listBundles(task),
    enabled: !!task,
  });
  const bundles = bundlesQ.data?.bundles ?? [];
  const selected = version || bundles[0] || "";

  const explainMut = useMutation({ mutationFn: () => explainBundle(task, selected, { top_k: 3 }) });
  const intervalMut = useMutation({ mutationFn: () => getIntervals(task, selected, { alpha }) });

  const importance = (explainMut.data?.importance ?? []).slice(0, 15);
  const intervals = intervalMut.data;

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Explain</h1>
        <span className="desc">Why a model predicts what it does, and how sure it is — for task “{task || "…"}”</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}
      {task && bundlesQ.isLoading && <Loading />}
      {task && bundlesQ.data && bundles.length === 0 && (
        <EmptyState title="No models yet" hint="Run an experiment or train a baseline first." />
      )}

      {bundles.length > 0 && (
        <Card title="Model">
          <div className="row">
            <div className="field">
              <label>Model version</label>
              <select value={selected} onChange={(e) => setVersion(e.target.value)}>
                {bundles.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>
          </div>
        </Card>
      )}

      {selected && <LineageRail task={task} modelVersion={selected} />}

      {selected && (
        <>
          <Card
            title="Feature importance"
            sub="How much the score drops when a feature is shuffled (permutation importance), plus how predictions change across each top feature's range."
            actions={
              <button className="btn primary" disabled={explainMut.isPending} onClick={() => explainMut.mutate()}>
                {explainMut.isPending ? "Computing…" : "Explain model"}
              </button>
            }
          >
            {explainMut.isError && <ErrorNote error={explainMut.error} />}
            {explainMut.data && (
              <>
                <p className="muted">Computed on {explainMut.data.rows} rows of the “{explainMut.data.partition}” partition.</p>
                <div style={{ height: Math.max(180, importance.length * 26) }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={importance} layout="vertical" margin={{ left: 24, right: 16 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={110} interval={0} />
                      <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                      <Bar dataKey="mean" fill="#6366f1" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid cols-3">
                  {explainMut.data.partial_dependence.map((curve) => {
                    const data = curve.grid.map((g, i) => ({ x: g, y: curve.average[i] }));
                    return (
                      <div key={curve.feature}>
                        <strong>{curve.feature}</strong>
                        <div style={{ height: 160 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="x" tickFormatter={(v: number) => Number(v).toPrecision(3)} />
                              <YAxis width={48} domain={["auto", "auto"]} />
                              <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                              <Line type="monotone" dataKey="y" stroke="#10b981" dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {explainMut.data.errors.map((e) => (
                  <div key={e.feature} className="hint">⚠ {e.feature}: {e.detail}</div>
                ))}
              </>
            )}
          </Card>

          <Card
            title="Prediction intervals"
            sub="Regression models only. Split-conformal intervals: a range expected to contain the true value with the chosen coverage, checked on held-out rows."
            actions={
              <button className="btn primary" disabled={intervalMut.isPending} onClick={() => intervalMut.mutate()}>
                {intervalMut.isPending ? "Computing…" : "Compute intervals"}
              </button>
            }
          >
            <div className="row">
              <div className="field">
                <label>Target coverage</label>
                <select value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}>
                  <option value={0.2}>80%</option>
                  <option value={0.1}>90%</option>
                  <option value={0.05}>95%</option>
                </select>
              </div>
            </div>
            {intervalMut.isError && <ErrorNote error={intervalMut.error} />}
            {intervals && (
              <>
                <div className="grid cols-4">
                  <Stat label="Target coverage" value={fmtPct(intervals.nominal_coverage)} />
                  <Stat label="Observed on held-out rows" value={fmtPct(intervals.empirical_coverage)} sub={`${intervals.holdout_rows} rows`} />
                  <Stat label="Interval half-width" value={fmtNum(intervals.half_width)} sub="± around each prediction" />
                  <Stat label="Calibration rows" value={fmtNum(intervals.calibration_size)} />
                </div>
                <p className="muted">{intervals.coverage_assumption}</p>
              </>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
