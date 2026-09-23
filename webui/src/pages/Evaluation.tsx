import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getBundle, getLearningCurveMetrics, getLearningCurves, listBundles, listJobs } from "../api/client";
import type { BundleSummary } from "../api/client";
import { Card, EmptyState, ErrorNote, Loading, Table, Tabs, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

const CHART_COLORS = ["#4f8cff", "#7c5cff", "#34d399", "#fbbf24", "#f87171", "#60a5fa"];

/** Tab id of the side-by-side metric comparison bar chart (every other tab is a learning curve). */
const OVERVIEW_TAB = "overview";

export default function EvaluationPage() {
  const { task } = useTask();

  const [tab, setTab] = useState<string>(OVERVIEW_TAB);
  const [trainPartition, setTrainPartition] = useState("train");

  // Metric tabs describe metrics of *this* task's models, so drop the selection on task change.
  useEffect(() => {
    setTab(OVERVIEW_TAB);
  }, [task]);

  const listQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });

  const bundleQs = useQuery({
    queryKey: ["bundle-details", task, listQ.data?.bundles ?? []],
    queryFn: async (): Promise<BundleSummary[]> => {
      if (!listQ.data) return [];
      const results = await Promise.all(
        listQ.data.bundles.map((v) => getBundle(task, v).catch(() => null)),
      );
      return results.filter((b): b is BundleSummary => b !== null);
    },
    enabled: !!task && (listQ.data?.bundles.length ?? 0) > 0,
  });

  const jobsQ = useQuery({ queryKey: ["jobs", task], queryFn: () => listJobs(task), enabled: !!task });

  const bundles = bundleQs.data ?? [];

  const metricNames = useMemo(() => {
    const names = new Set<string>();
    for (const b of bundles) Object.keys(b.metrics).forEach((k) => names.add(k));
    return [...names];
  }, [bundles]);

  const chartData = useMemo(
    () =>
      metricNames.map((m) => {
        const row: Record<string, string | number> = { metric: m };
        for (const b of bundles) row[b.model_version] = b.metrics[m] ?? null;
        return row;
      }),
    [bundles, metricNames],
  );

  // Relevant metrics for the task's problem family — cheap (no curve computation).
  const lcMetricsQ = useQuery({
    queryKey: ["learning-curve-metrics", task, bundles[0]?.model_version ?? null],
    queryFn: () => getLearningCurveMetrics(task, bundles[0]?.model_version),
    enabled: !!task && bundles.length > 0,
    retry: false,
  });

  const lcMetrics = lcMetricsQ.data?.metrics ?? [];
  const activeMetric = lcMetrics.find((m) => m.value === tab) ?? null;
  const modelVersions = useMemo(() => bundles.map((b) => b.model_version), [bundles]);

  // Curves are expensive, so each metric tab loads on demand only while it is open.
  const curvesQ = useQuery({
    queryKey: ["learning-curves", task, tab, trainPartition, modelVersions],
    queryFn: () => getLearningCurves(task, { scoring: tab, trainPartition, modelVersions }),
    enabled: !!task && activeMetric !== null,
    retry: false,
  });

  const curveRows = useMemo(() => {
    const series = curvesQ.data?.series ?? [];
    if (series.length === 0) return [];
    const pointCount = Math.max(...series.map((s) => s.train_sizes.length));
    return Array.from({ length: pointCount }, (_, i) => {
      const first = series[0];
      const size = first.train_sizes[i];
      const fraction = first.train_sizes_fraction[i];
      const row: Record<string, string | number | null> = {
        size: fraction === undefined ? fmtNum(size) : `${fmtNum(size)} (${(fraction * 100).toFixed(0)}%)`,
      };
      for (const s of series) {
        row[`${s.model_version} (validation)`] = s.validation_scores_mean[i] ?? null;
        row[`${s.model_version} (train)`] = s.train_scores_mean[i] ?? null;
      }
      return row;
    });
  }, [curvesQ.data]);

  const experimentJobs = (jobsQ.data?.jobs ?? []).filter(
    (j) => j.kind === "experiment" && j.status === "completed",
  );

  if (!task) {
    return <div className="error-note">Pick a task name in the top bar to compare models.</div>;
  }

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Evaluation</h1>
        <span className="desc">Side-by-side metric comparison of every bundle for task “{task}”</span>
      </div>

      {listQ.isLoading && <Loading />}
      {listQ.isError && <ErrorNote error={listQ.error} />}

      {bundles.length === 0 && !bundleQs.isLoading && (
        <Card title="No bundles yet">
          <p className="muted">Train a baseline under Model Bundles or run an experiment first — metrics will appear here.</p>
        </Card>
      )}

      {bundles.length > 0 && (
        <>
          <Card
            title={`Metric comparison (${fmtNum(bundles.length)} bundles)`}
            sub="Overview compares stored metrics across bundles; the metric tabs plot each bundle's learning curve for that metric."
          >
            <Tabs
              active={tab}
              onChange={setTab}
              tabs={[
                { id: OVERVIEW_TAB, label: "Overview", badge: fmtNum(metricNames.length) },
                ...lcMetrics.map((m) => ({
                  id: m.value,
                  label: m.label,
                  badge: m.higher_is_better ? undefined : "lower is better",
                })),
              ]}
            />

            {tab === OVERVIEW_TAB && (
              <div style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis dataKey="metric" tick={{ fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {bundles.map((b, i) => (
                      <Bar key={b.model_version} dataKey={b.model_version} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[3, 3, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {activeMetric && (
              <>
                <div className="row">
                  <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
                    <label>Train partition</label>
                    <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
                  </div>
                </div>

                {curvesQ.isLoading && <Loading label="Fitting learning curves…" />}
                {curvesQ.isError && <ErrorNote error={curvesQ.error} />}

                {curvesQ.data && curvesQ.data.series.length === 0 && (
                  <EmptyState
                    title="No learning curves available"
                    hint={`No bundle could be scored for ${activeMetric.label} on the “${trainPartition}” partition.`}
                  />
                )}

                {curveRows.length > 0 && (
                  <>
                    <h4 className="hm-title">
                      {activeMetric.label} vs training size — {lcMetricsQ.data?.family ?? "unknown family"}
                    </h4>
                    <div style={{ height: 320 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={curveRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                          <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                          <XAxis dataKey="size" tick={{ fill: "#8b95ab", fontSize: 12 }} />
                          <YAxis domain={["auto", "auto"]} tick={{ fill: "#8b95ab", fontSize: 12 }} />
                          <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                          <Legend wrapperStyle={{ fontSize: 12 }} />
                          {curvesQ.data?.series.map((s, i) => (
                            <Line
                              key={`${s.model_version}:validation`}
                              type="monotone"
                              dataKey={`${s.model_version} (validation)`}
                              stroke={CHART_COLORS[i % CHART_COLORS.length]}
                              strokeWidth={2}
                              dot={{ r: 2 }}
                            />
                          ))}
                          {curvesQ.data?.series.map((s, i) => (
                            <Line
                              key={`${s.model_version}:train`}
                              type="monotone"
                              dataKey={`${s.model_version} (train)`}
                              stroke={CHART_COLORS[i % CHART_COLORS.length]}
                              strokeDasharray="4 4"
                              strokeWidth={1.5}
                              dot={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="muted" style={{ marginTop: 6 }}>
                      Solid = held-out score (cross-validated with the task's stored splitter), dashed = training score, over the{" "}
                      <code>{curvesQ.data?.train_partition}</code> partition.{" "}
                      {activeMetric.higher_is_better ? "Higher is better." : "Lower is better."}
                    </p>
                  </>
                )}

                {curvesQ.data && curvesQ.data.errors.length > 0 && (
                  <p className="muted">
                    Skipped {fmtNum(curvesQ.data.errors.length)} bundle(s):{" "}
                    {curvesQ.data.errors.map((e) => `${e.model_version.slice(0, 8)}… (${e.detail})`).join("; ")}
                  </p>
                )}
              </>
            )}
          </Card>

          <Card title="Bundle metrics">
            <Table<BundleSummary>
              columns={["Model version", ...metricNames, "Created"]}
              rows={bundles}
              keyOf={(b) => b.model_version}
              render={(b) => [
                <td key="v" className="mono">{b.model_version}</td>,
                ...metricNames.map((m) => (
                  <td key={m}>{fmtNum(b.metrics[m])}</td>
                )),
                <td key="c" className="mono">{b.created_at}</td>,
              ]}
            />
          </Card>

          {experimentJobs.length > 0 && (
            <Card title="Experiment evaluation history" sub="From completed AML/EOA jobs — best pipeline per run">
              <Table<typeof experimentJobs[number]>
                columns={["Job", "Created", "Train score", "History length"]}
                rows={experimentJobs}
                keyOf={(j) => j.job_id}
                render={(j) => {
                  const result = (j.result ?? {}) as Record<string, unknown>;
                  const history = Array.isArray(result.evaluation_history) ? result.evaluation_history : [];
                  return [
                    <td key="id" className="mono">{j.job_id.slice(0, 8)}…</td>,
                    <td key="c" className="mono">{j.created_at}</td>,
                    <td key="s">{fmtNum(result.train_score as number | undefined)}</td>,
                    <td key="h">{fmtNum(history.length)}</td>,
                  ];
                }}
              />
            </Card>
          )}
        </>
      )}
    </div>
  );
}
