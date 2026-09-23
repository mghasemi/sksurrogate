import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  asNestedCVResult,
  getBundle,
  getBundleCurves,
  getLearningCurveMetrics,
  getLearningCurves,
  listBundles,
  listJobs,
  runNestedCV,
} from "../api/client";
import type { BundleSummary, NestedCVJobResult } from "../api/client";
import { Card, EmptyState, ErrorNote, Loading, Stat, Table, Tabs, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { useTask } from "../lib/task-context";

const CHART_COLORS = ["#4f8cff", "#7c5cff", "#34d399", "#fbbf24", "#f87171", "#60a5fa"];

/** Tab id of the side-by-side metric comparison bar chart (every other tab is a learning curve). */
const OVERVIEW_TAB = "overview";

/** Drop null/NaN points so Recharts lines stay continuous. */
function finitePoints(xs: Array<number | null>, ys: Array<number | null>): Array<{ x: number; y: number }> {
  const rows: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
    const x = xs[i];
    const y = ys[i];
    if (x != null && y != null) rows.push({ x, y });
  }
  return rows;
}

/**
 * Like ``finitePoints`` but carries a second series (the other class's curve),
 * keeping index alignment instead of filtering it away with the first series.
 */
function pairedPoints(
  xs: Array<number | null>,
  ys1: Array<number | null>,
  ys2: Array<number | null>,
): Array<{ x: number; y: number; y1: number | null }> {
  const rows: Array<{ x: number; y: number; y1: number | null }> = [];
  for (let i = 0; i < Math.min(xs.length, ys1.length); i++) {
    const x = xs[i];
    const y = ys1[i];
    if (x != null && y != null) rows.push({ x, y, y1: ys2[i] ?? null });
  }
  return rows;
}

/** Bundle picker shared by the nested-CV and diagnostic-curves cards. */
function BundlePicker({ bundles, value, onChange }: { bundles: BundleSummary[]; value: string; onChange: (v: string) => void }) {
  useEffect(() => {
    if (bundles.length > 0 && !bundles.some((b) => b.model_version === value)) {
      onChange(bundles[0].model_version);
    }
  }, [bundles, value, onChange]);

  return (
    <div className="field fixed" style={{ flex: "1 1 260px", minWidth: 220 }}>
      <label>Bundle</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {bundles.map((b) => (
          <option key={b.model_version} value={b.model_version}>
            {b.model_version.slice(0, 12)}…
          </option>
        ))}
      </select>
    </div>
  );
}

/** Small labelled numeric input used by the nested-CV and curves cards. */
function NumberField({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min: number;
  max: number;
}) {
  return (
    <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 108 }}>
      <label>{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

/** Nested cross-validation: submit as a background job, render the result when it lands. */
function NestedCVCard({ task, bundles }: { task: string; bundles: BundleSummary[] }) {
  const [modelVersion, setModelVersion] = useState(bundles[0]?.model_version ?? "");
  const [innerCv, setInnerCv] = useState(2);
  const [outerCv, setOuterCv] = useState(2);
  const [repeats, setRepeats] = useState(1);
  const [confidenceLevel, setConfidenceLevel] = useState(0.95);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<NestedCVJobResult | null>(null);

  const submit = useMutation({
    mutationFn: () =>
      runNestedCV(task, {
        model_version: modelVersion || null,
        inner_cv: Math.max(1, innerCv),
        outer_cv: Math.max(1, outerCv),
        repeats: Math.max(1, repeats),
        confidence_level: confidenceLevel,
      }),
    onSuccess: (res) => {
      setJobId(res.job_id);
      setResult(null);
    },
  });

  // One bar per outer fold (the first repeat's folds, matching outer_scores).
  const foldRows = useMemo(
    () =>
      (result?.fold_metrics ?? []).map((f) => ({
        name: `fold ${f.outer_fold}`,
        score: f.outer_score ?? null,
      })),
    [result],
  );

  return (
    <Card
      title="Nested CV"
      sub="Inner loop tunes the estimator, outer loop measures it — an unbiased estimate of generalization. Expensive: runs as a background job."
    >
      <div className="row">
        <BundlePicker bundles={bundles} value={modelVersion} onChange={setModelVersion} />
        <NumberField label="Inner CV" value={innerCv} onChange={setInnerCv} min={1} max={10} />
        <NumberField label="Outer CV" value={outerCv} onChange={setOuterCv} min={1} max={10} />
        <NumberField label="Repeats" value={repeats} onChange={setRepeats} min={1} max={20} />
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
          <label>Confidence level ({(confidenceLevel * 100).toFixed(0)}%)</label>
          <input
            type="range"
            min={50}
            max={99.9}
            step={0.1}
            value={confidenceLevel * 100}
            onChange={(e) => setConfidenceLevel(Number(e.target.value) / 100)}
          />
        </div>
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn primary" disabled={submit.isPending || bundles.length === 0} onClick={() => submit.mutate()}>
          {submit.isPending ? "Submitting…" : "Run nested CV"}
        </button>
      </div>

      {submit.isError && <ErrorNote error={submit.error} />}

      {jobId && (
        <JobTracker
          jobId={jobId}
          onResult={(r) => {
            const parsed = asNestedCVResult(r);
            if (parsed) setResult(parsed);
          }}
        />
      )}

      {result && (
        <>
          <div className="stat-row">
            <Stat
              label="Mean outer score"
              value={fmtNum(result.mean_outer_score)}
              sub={
                result.confidence_interval.lower != null && result.confidence_interval.upper != null
                  ? `${(result.confidence_interval.confidence_level * 100).toFixed(0)}% CI [${fmtNum(result.confidence_interval.lower)}, ${fmtNum(result.confidence_interval.upper)}]`
                  : undefined
              }
            />
            <Stat label="Final validation score" value={fmtNum(result.final_validation_score ?? null)} sub="pooled, untouched partition" />
            <Stat label="Repeats" value={fmtNum(result.repeat_count)} sub={`${result.repeated_outer_scores.length} outer means`} />
          </div>

          {foldRows.length > 0 && (
            <>
              <h4 className="hm-title">Outer-fold scores (repeat 1) vs mean outer score</h4>
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={foldRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis domain={["auto", "auto"]} tick={{ fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    {result.mean_outer_score != null && (
                      <ReferenceLine
                        y={result.mean_outer_score}
                        stroke="#fbbf24"
                        strokeDasharray="4 4"
                        label={{ value: "mean", fill: "#fbbf24", fontSize: 11, position: "right" }}
                      />
                    )}
                    <Bar dataKey="score" fill="#4f8cff" radius={[3, 3, 0, 0]} name="Outer score" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {result.fold_metrics.length > 0 && (
            <Table<typeof result.fold_metrics[number]>
              columns={["Outer fold", "Inner scores", "Inner mean", "Outer score"]}
              rows={result.fold_metrics}
              keyOf={(f) => String(f.outer_fold)}
              render={(f) => [
                <td key="fold">{fmtNum(f.outer_fold)}</td>,
                <td key="inner" className="mono">{f.inner_scores.map((s) => fmtNum(s)).join(", ")}</td>,
                <td key="im">{fmtNum(f.inner_score_mean)}</td>,
                <td key="os">{fmtNum(f.outer_score)}</td>,
              ]}
            />
          )}
        </>
      )}
    </Card>
  );
}

/** Diagnostic curves (ROC / calibration / gain / lift) for one classification bundle. */
function CurvesCard({ task, bundles }: { task: string; bundles: BundleSummary[] }) {
  const [modelVersion, setModelVersion] = useState(bundles[0]?.model_version ?? "");
  const [bins, setBins] = useState(10);
  const [tab, setTab] = useState("roc");

  useEffect(() => setTab("roc"), [task]);

  const curvesQ = useQuery({
    queryKey: ["bundle-curves", task, modelVersion, bins],
    queryFn: () => getBundleCurves(task, modelVersion, bins),
    enabled: !!task && bundles.length > 0 && !!modelVersion,
    retry: false,
  });

  const data = curvesQ.data;

  const rocRows = useMemo(() => (data?.roc ? finitePoints(data.roc.fpr, data.roc.tpr) : []), [data]);
  const calibRows = useMemo(
    () => (data?.calibration ? finitePoints(data.calibration.mean_predicted_value, data.calibration.fraction_of_positives) : []),
    [data],
  );
  const histRows = useMemo(() => {
    if (!data?.calibration) return [];
    const { histogram_counts: counts, histogram_edges: edges } = data.calibration;
    return counts.map((count, i) => ({ bin: `${(edges[i] ?? 0).toFixed(2)}–${(edges[i + 1] ?? 1).toFixed(2)}`, count }));
  }, [data]);
  const gainRows = useMemo(() => {
    const g = data?.cumulative_gain;
    return g ? pairedPoints(g.percentages, g.gains_class0, g.gains_class1) : [];
  }, [data]);
  const liftRows = useMemo(() => {
    const l = data?.lift;
    return l ? pairedPoints(l.percentages, l.lifts_class0, l.lifts_class1) : [];
  }, [data]);

  const diagonal = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  /** A random classifier's lift is flat at 1. */
  const liftBaseline = [{ x: 0, y: 1 }, { x: 1, y: 1 }];

  return (
    <Card title="Diagnostic curves" sub="Computed server-side on the bundle's cached 75/25 split — classification bundles only.">
      <div className="row">
        <BundlePicker bundles={bundles} value={modelVersion} onChange={setModelVersion} />
        <NumberField label="Calibration bins" value={bins} onChange={setBins} min={2} max={50} />
      </div>

      <Tabs
        active={tab}
        onChange={setTab}
        tabs={[
          { id: "roc", label: "ROC" },
          { id: "calibration", label: "Calibration" },
          { id: "gain", label: "Cumulative gain" },
          { id: "lift", label: "Lift" },
        ]}
      />

      {curvesQ.isLoading && <Loading label="Computing curves…" />}
      {curvesQ.isError && <ErrorNote error={curvesQ.error} />}

      {data && (
        <>
          {tab === "roc" && data.roc && (
            <>
              <h4 className="hm-title">ROC curve — AUC = {fmtNum(data.roc.auc)}</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "False positive rate", position: "insideBottom", offset: -2, fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "True positive rate", angle: -90, position: "insideLeft", fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Line data={diagonal} stroke="#8b95ab" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="chance" />
                    <Line data={rocRows} stroke="#4f8cff" strokeWidth={2} dot={false} isAnimationActive={false} name="model" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {tab === "calibration" && data.calibration && (
            <>
              <h4 className="hm-title">Reliability curve</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Mean predicted value", position: "insideBottom", offset: -2, fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Fraction of positives", angle: -90, position: "insideLeft", fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Line data={diagonal} stroke="#8b95ab" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="perfectly calibrated" />
                    <Line data={calibRows} stroke="#7c5cff" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} name="model" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <h4 className="hm-title">Predicted-probability histogram</h4>
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis dataKey="bin" tick={{ fill: "#8b95ab", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#8b95ab", fontSize: 12 }} allowDecimals={false} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Bar dataKey="count" fill="#7c5cff" radius={[3, 3, 0, 0]} name="Count" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {tab === "gain" && (data.cumulative_gain ? (
            <>
              <h4 className="hm-title">Cumulative gain — class {data.cumulative_gain.class0} vs class {data.cumulative_gain.class1}</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Percentage of sample", position: "insideBottom", offset: -2, fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Gain", angle: -90, position: "insideLeft", fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line data={diagonal} stroke="#8b95ab" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="baseline" />
                    <Line data={gainRows} dataKey="y" stroke="#34d399" strokeWidth={2} dot={false} isAnimationActive={false} name={`class ${data.cumulative_gain.class0}`} connectNulls />
                    <Line data={gainRows} dataKey="y1" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} name={`class ${data.cumulative_gain.class1}`} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <EmptyState title="Cumulative gain unavailable" hint="Requires exactly 2 classes in the test partition." />
          ))}

          {tab === "lift" && (data.lift ? (
            <>
              <h4 className="hm-title">Lift — class {data.lift.class0} vs class {data.lift.class1}</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Percentage of sample", position: "insideBottom", offset: -2, fill: "#8b95ab", fontSize: 12 }} />
                    <YAxis type="number" dataKey="y" domain={[0, "auto"]} tick={{ fill: "#8b95ab", fontSize: 12 }} label={{ value: "Lift", angle: -90, position: "insideLeft", fill: "#8b95ab", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line data={liftBaseline} stroke="#8b95ab" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="baseline (lift = 1)" />
                    <Line data={liftRows} dataKey="y" stroke="#34d399" strokeWidth={2} dot={false} isAnimationActive={false} name={`class ${data.lift.class0}`} connectNulls />
                    <Line data={liftRows} dataKey="y1" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} name={`class ${data.lift.class1}`} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <EmptyState title="Lift unavailable" hint="Requires exactly 2 classes in the test partition." />
          ))}

          {data.errors.length > 0 && (
            <p className="muted">
              Skipped curve(s): {data.errors.map((e) => `${e.curve} (${e.detail})`).join("; ")}
            </p>
          )}
        </>
      )}
    </Card>
  );
}



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
          <NestedCVCard task={task} bundles={bundles} />

          <CurvesCard task={task} bundles={bundles} />

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
