import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { getBundle, listBundles, listJobs } from "../api/client";
import type { BundleSummary } from "../api/client";
import { Card, ErrorNote, Loading, Table, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

const CHART_COLORS = ["#4f8cff", "#7c5cff", "#34d399", "#fbbf24", "#f87171", "#60a5fa"];

export default function EvaluationPage() {
  const { task } = useTask();

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
          <Card title={`Metric comparison (${fmtNum(bundles.length)} bundles)`}>
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
