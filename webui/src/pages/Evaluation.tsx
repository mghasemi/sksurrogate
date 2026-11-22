import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
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
  getDatasetMetadata,
  getLearningCurveMetrics,
  getLearningCurves,
  listBundles,
  listJobs,
  registrySummary,
  retryUnlessNotFound,
  runBacktest,
  runNestedCV,
} from "../api/client";
import type { BacktestResult, BundleSummary, NestedCVJobResult } from "../api/client";
import { Card, EmptyState, ErrorNote, Loading, Stat, Table, Tabs, fmtNum } from "../components/ui";
import { JobTracker } from "../components/JobTracker";
import { useBundleLabel } from "../lib/hooks";
import { useDefaultPartition } from "../lib/partition";
import { useTask } from "../lib/task-context";

const CHART_COLORS = ["#4f8cff", "#7c5cff", "#34d399", "#fbbf24", "#f87171", "#60a5fa"];

/** Tab id of the side-by-side metric comparison bar chart (every other tab is a learning curve). */
const OVERVIEW_TAB = "overview";

/**
 * Which producer writes each metric into ``bundle.metrics``. The Bundle metrics
 * table shows a column for every metric any bundle carries, so a blank cell is a
 * producer decision (this run never measured it), not a failed computation.
 * Values computed on demand — a rolling-origin backtest's MAE/RMSE/MASE/sMAPE, a
 * nested-CV outer score — live on the bundle's ``evaluations`` record instead.
 */
const METRIC_ORIGIN: Record<string, string> = {
  train_score: "refit training score in the estimator's own units (accuracy / R²) — experiments, pipeline optimization and ensembles",
  score: "score on the validation partition — tabular baselines trained with one",
  cv_score: "out-of-fold cross-validated score — ensembles only",
  cv_score_std: "standard deviation of that ensemble CV score across folds",
  mae: "pooled rolling-origin backtest — forecasting baselines",
  rmse: "pooled rolling-origin backtest — forecasting baselines",
  mase: "pooled rolling-origin backtest, seasonal-naive scale — forecasting baselines",
  smape: "pooled rolling-origin backtest — forecasting baselines",
};

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
function BundlePicker({
  bundles,
  value,
  onChange,
  labelOf,
}: {
  bundles: BundleSummary[];
  value: string;
  onChange: (v: string) => void;
  /** Display label for a version — alias-aware; the option's value stays the raw id. */
  labelOf: (version: string) => string;
}) {
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
            {labelOf(b.model_version)}
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
function NestedCVCard({ task, bundles, labelOf }: { task: string; bundles: BundleSummary[]; labelOf: (version: string) => string }) {
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
        <BundlePicker bundles={bundles} value={modelVersion} onChange={setModelVersion} labelOf={labelOf} />
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
function CurvesCard({ task, bundles, labelOf }: { task: string; bundles: BundleSummary[]; labelOf: (version: string) => string }) {
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
        <BundlePicker bundles={bundles} value={modelVersion} onChange={setModelVersion} labelOf={labelOf} />
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
      {curvesQ.isError &&
        (String(curvesQ.error).includes("classification bundle") ? (
          <p className="muted">Diagnostic curves apply to classification bundles. For a regression model, use the metric tabs below and the Explain page.</p>
        ) : (
          <ErrorNote error={curvesQ.error} />
        ))}

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



/** Coerce a job result into the backtest document, or null when it is another job kind. */
function asBacktestResult(value: unknown): BacktestResult | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (!Array.isArray(record.origins) || typeof record.pooled !== "object" || record.pooled === null) {
    return null;
  }
  return record as unknown as BacktestResult;
}

const BASELINE_LABELS: Record<string, string> = {
  naive: "naive (last value)",
  seasonal_naive: "seasonal naive",
  drift: "drift",
  mean: "training mean",
  historical_mean_by_series: "per-series mean",
};

function metricOf(block: Record<string, number | null | undefined> | undefined, key: string): number | null {
  const value = block?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Rolling-origin backtest (plan phase P7.2).
 *
 * Submits `/api/evaluation/{task}/backtest` as a background job (one fit per
 * origin) and renders the result: pooled metrics with every naive baseline
 * scored on the identical origins, a per-horizon table, and a forecast-vs-
 * actual fan chart whose band is the stored conformal calibration. MASE is
 * shown only where the backtest recorded an in-sample scale — it is never
 * replaced by a fabricated denominator.
 */
function BacktestCard({
  task,
  bundles,
  labelOf,
}: {
  task: string;
  bundles: BundleSummary[];
  labelOf: (version: string) => string;
}) {
  const [modelVersion, setModelVersion] = useState(bundles[0]?.model_version ?? "");
  const [horizon, setHorizon] = useState(1);
  const [nSplits, setNSplits] = useState(5);
  const [step, setStep] = useState(0); // 0 → splitter default (horizon)
  const [minTrain, setMinTrain] = useState(0); // 0 → splitter default
  const [expanding, setExpanding] = useState(true);
  const [refit, setRefit] = useState("every_fold");
  // Empty = let the server score the bundle on the labels it was fitted for
  // (an experiment bundle is one-step, so it is backtested recursively).
  const [strategy, setStrategy] = useState<"" | "direct" | "recursive">("");
  const [alpha, setAlpha] = useState(0.1);
  const [intervalMethod, setIntervalMethod] = useState("absolute");
  const [seasonality, setSeasonality] = useState(1);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [seriesPick, setSeriesPick] = useState("");
  // make_supervised kwargs — must reproduce the bundle's feature design, or the
  // cloned estimator sees a different width and every origin fails.
  const [featuresJson, setFeaturesJson] = useState("{}");
  const [featuresError, setFeaturesError] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: (features: Record<string, unknown>) =>
      runBacktest(task, {
        model_version: modelVersion || null,
        horizon: Math.max(1, horizon),
        n_splits: Math.max(1, nSplits),
        step: step > 0 ? step : null,
        min_train: minTrain > 0 ? minTrain : null,
        expanding,
        refit,
        strategy: strategy || undefined,
        alpha,
        interval_method: intervalMethod,
        seasonality: Math.max(1, seasonality),
        features,
      }),
    onSuccess: (res) => {
      setJobId(res.job_id);
      setResult(null);
    },
  });

  const origins = result?.origins ?? [];
  // Panel records carry `series`; `series_id` is kept for older payloads.
  const seriesOf = (o: { series?: unknown; series_id?: unknown }) =>
    String(o.series ?? o.series_id ?? "(single series)");
  const seriesKeys = Array.from(new Set(origins.map(seriesOf)));
  const activeSeries = seriesPick && seriesKeys.includes(seriesPick) ? seriesPick : seriesKeys[0] ?? "";
  const calibration = result?.intervals?.available
    ? (result.intervals.calibration as Record<string, unknown> | null)
    : null;
  const lowerOffset = calibration ? Number(calibration.lower_offset ?? 0) : 0;
  const upperOffset = calibration ? Number(calibration.upper_offset ?? 0) : 0;

  const chartRows = origins
    .filter((o) => seriesOf(o) === activeSeries)
    .map((o, index) => {
      const pred = o.y_pred ?? null;
      const lower = pred != null && calibration ? pred - lowerOffset : null;
      const upper = pred != null && calibration ? pred + upperOffset : null;
      return {
        label: String(o.target_time ?? index + 1),
        step: o.target_step ?? index + 1,
        actual: o.y_true ?? null,
        predicted: pred,
        lower,
        span: lower != null && upper != null ? Math.max(0, upper - lower) : 0,
        baseline: o.baselines?.naive ?? o.baselines?.seasonal_naive ?? null,
      };
    });

  // Plan P7.2: the naive baselines are pinned above the candidate so a model
  // that loses to "last value" cannot be read as the winner of the table.
  const leaderboard = result
    ? [
        ...Object.entries(result.baselines ?? {}).map(([name, metrics]) => ({
          name,
          role: `${BASELINE_LABELS[name] ?? name} · naive`,
          metrics,
        })),
        { name: "model", role: "candidate", metrics: result.pooled },
      ]
    : [];
  const horizonRows = result
    ? Object.entries(result.per_horizon ?? {}).map(([stepKey, metrics]) => ({
        step: Number(stepKey),
        metrics,
      }))
    : [];

  return (
    <Card
      title="Rolling-origin backtest"
      sub="Walk origins forward in time, refit per policy, and score multi-step forecasts against naive baselines on the identical origins. Runs as a background job."
    >
      <div className="row">
        <BundlePicker bundles={bundles} value={modelVersion} onChange={setModelVersion} labelOf={labelOf} />
        <NumberField label="Horizon" value={horizon} onChange={setHorizon} min={1} max={100} />
        <NumberField label="Origins" value={nSplits} onChange={setNSplits} min={1} max={50} />
        <NumberField label="Step (0=horizon)" value={step} onChange={setStep} min={0} max={100} />
        <NumberField label="Min train (0=default)" value={minTrain} onChange={setMinTrain} min={0} max={10000} />
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <label className="checkbox-row" style={{ marginBottom: 12 }}>
          <input type="checkbox" checked={expanding} onChange={(e) => setExpanding(e.target.checked)} />
          expanding window
        </label>
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
          <label>Refit policy</label>
          <select value={refit} onChange={(e) => setRefit(e.target.value)}>
            <option value="every_fold">every_fold</option>
            <option value="first">first</option>
            <option value="every:2">every:2</option>
            <option value="every:3">every:3</option>
            <option value="every:5">every:5</option>
          </select>
        </div>
        <div className="field fixed" style={{ flex: "0 1 190px", minWidth: 160 }}>
          <label>Strategy</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as "" | "direct" | "recursive")}
          >
            <option value="">bundle default (one-step ⇒ recursive)</option>
            <option value="direct">direct (y at t+h)</option>
            <option value="recursive">recursive (replay h steps)</option>
          </select>
        </div>
        <NumberField label="α (interval miss)" value={Math.round(alpha * 100)} onChange={(v) => setAlpha(Math.min(0.99, Math.max(0.001, v / 100)))} min={1} max={99} />
        <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
          <label>Interval method</label>
          <select value={intervalMethod} onChange={(e) => setIntervalMethod(e.target.value)}>
            <option value="absolute">absolute (symmetric)</option>
            <option value="asymmetric">asymmetric</option>
          </select>
        </div>
        <NumberField label="Seasonality m" value={seasonality} onChange={setSeasonality} min={1} max={1000} />
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <div className="field" style={{ flex: "1 1 480px" }}>
          <label>Features (make_supervised kwargs, JSON)</label>
          <textarea
            className="mono"
            rows={3}
            spellCheck={false}
            value={featuresJson}
            onChange={(e) => setFeaturesJson(e.target.value)}
            placeholder='{"lags": [4, 8], "rolling": [[4, "mean"]]}'
          />
          <span className="hint">
            Must reproduce the bundle's feature design (lags/rolling/diffs/calendar/fourier/exogenous) —
            a wider or narrower design makes every origin fail on feature width.
          </span>
        </div>
      </div>

      {featuresError && <div className="error-note">{featuresError}</div>}

      <div className="row" style={{ marginTop: 8 }}>
        <button
          className="btn primary"
          disabled={submit.isPending || bundles.length === 0}
          onClick={() => {
            let features: Record<string, unknown> = {};
            try {
              const parsed = JSON.parse(featuresJson || "{}");
              if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
                throw new Error("expected a JSON object");
              }
              features = parsed as Record<string, unknown>;
            } catch (err) {
              setFeaturesError(`Invalid features JSON: ${err instanceof Error ? err.message : String(err)}`);
              return;
            }
            setFeaturesError(null);
            submit.mutate(features);
          }}
        >
          {submit.isPending ? "Submitting…" : "Run backtest"}
        </button>
      </div>

      {submit.isError && <ErrorNote error={submit.error} />}

      {jobId && (
        <JobTracker
          jobId={jobId}
          onResult={(r) => {
            const parsed = asBacktestResult(r);
            if (parsed) setResult(parsed);
          }}
        />
      )}

      {result && (
        <>
          <div className="stat-row">
            <Stat label="MAE (pooled)" value={fmtNum(metricOf(result.pooled, "mae"))} sub={`${origins.length} origin records`} />
            <Stat label="RMSE (pooled)" value={fmtNum(metricOf(result.pooled, "rmse"))} />
            <Stat
              label="MASE"
              value={fmtNum(metricOf(result.pooled, "mase"))}
              sub="1 = seasonal-naive parity; >1 is worse"
            />
            <Stat
              label="Intervals"
              value={result.intervals?.available ? `${((1 - alpha) * 100).toFixed(0)}%` : "none"}
              sub={result.intervals?.available ? `${intervalMethod}, α=${alpha}` : "calibration unavailable"}
            />
          </div>

          {chartRows.length > 0 && (
            <>
              <div className="row" style={{ marginBottom: 4 }}>
                {seriesKeys.length > 1 && (
                  <div className="field fixed" style={{ flex: "0 1 220px", minWidth: 160 }}>
                    <label>Series</label>
                    <select value={activeSeries} onChange={(e) => setSeriesPick(e.target.value)}>
                      {seriesKeys.map((key) => (
                        <option key={key} value={key}>
                          {key}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
              <h4 className="hm-title">Forecast vs actual — {activeSeries}</h4>
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#232c40" strokeDasharray="3 3" />
                    <XAxis dataKey="label" tick={{ fill: "#8b95ab", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#8b95ab", fontSize: 12 }} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ background: "#131926", border: "1px solid #232c40" }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {calibration && (
                      <Area
                        dataKey="lower"
                        stackId="band"
                        stroke="none"
                        fill="transparent"
                        isAnimationActive={false}
                        legendType="none"
                        name="interval floor"
                      />
                    )}
                    {calibration && (
                      <Area
                        dataKey="span"
                        stackId="band"
                        stroke="none"
                        fill="#4f8cff"
                        fillOpacity={0.18}
                        isAnimationActive={false}
                        name="prediction interval"
                      />
                    )}
                    <Line type="monotone" dataKey="actual" stroke="#e2e8f0" strokeWidth={2} dot={{ r: 2 }} name="actual" connectNulls isAnimationActive={false} />
                    <Line type="monotone" dataKey="predicted" stroke="#4f8cff" strokeWidth={2} dot={{ r: 2 }} name="model" connectNulls isAnimationActive={false} />
                    <Line type="monotone" dataKey="baseline" stroke="#fbbf24" strokeDasharray="4 4" strokeWidth={1.5} dot={false} name="naive" connectNulls isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {leaderboard.length > 0 && (
            <>
              <h4 className="hm-title">Model vs naive baselines (same origins)</h4>
              <Table<{ name: string; role: string; metrics: Record<string, number | null | undefined> }>
                columns={["Method", "Role", "MAE", "RMSE", "Bias", "sMAPE (%)", "MASE"]}
                rows={leaderboard}
                keyOf={(row) => row.name}
                render={(row) => [
                  <td key="n" className="mono">{row.name}</td>,
                  <td key="r">{row.role}</td>,
                  <td key="mae">{fmtNum(metricOf(row.metrics, "mae"))}</td>,
                  <td key="rmse">{fmtNum(metricOf(row.metrics, "rmse"))}</td>,
                  <td key="bias">{fmtNum(metricOf(row.metrics, "bias"))}</td>,
                  <td key="smape">{fmtNum(metricOf(row.metrics, "smape"))}</td>,
                  <td key="mase">{fmtNum(metricOf(row.metrics, "mase"))}</td>,
                ]}
              />
            </>
          )}

          {horizonRows.length > 0 && (
            <>
              <h4 className="hm-title">Error by horizon step</h4>
              <Table<{ step: number; metrics: Record<string, number | null | undefined> }>
                columns={["Step", "MAE", "RMSE", "Bias", "MASE"]}
                rows={horizonRows}
                keyOf={(row) => row.step}
                render={(row) => [
                  <td key="s">{fmtNum(row.step)}</td>,
                  <td key="mae">{fmtNum(metricOf(row.metrics, "mae"))}</td>,
                  <td key="rmse">{fmtNum(metricOf(row.metrics, "rmse"))}</td>,
                  <td key="bias">{fmtNum(metricOf(row.metrics, "bias"))}</td>,
                  <td key="mase">{fmtNum(metricOf(row.metrics, "mase"))}</td>,
                ]}
              />
            </>
          )}

          {(result.skipped_series ?? []).length > 0 && (
            <p className="muted">
              Skipped (too short for the requested geometry): {(result.skipped_series ?? []).join(", ")}
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
  const defaultTrainPartition = useDefaultPartition(task);
  const [trainPartition, setTrainPartition] = useState("");
  const effectiveTrainPartition = trainPartition || defaultTrainPartition;
  const [pinballAlpha, setPinballAlpha] = useState("0.5");

  // Metric tabs describe metrics of *this* task's models, so drop the selection on task change.
  useEffect(() => {
    setTab(OVERVIEW_TAB);
  }, [task]);

  // A typed partition belongs to the task it was typed for.
  useEffect(() => {
    setTrainPartition("");
  }, [task]);

  const listQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });

  // A backtest is a rolling-origin *forecast* protocol: it needs a declared time column,
  // and the control plane refuses a tabular task with 400 at submit. The panel is
  // therefore only offered where ``temporal.declared`` is true.
  const datasetQ = useQuery({
    queryKey: ["dataset-metadata", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });

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

  // Named registry aliases give bundles human labels on this page; the Registry page owns the
  // cache entry under the same key, and a 404 just means "no registry state yet".
  const aliasQ = useQuery({
    queryKey: ["registry-summary", task],
    queryFn: () => registrySummary(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });

  /** Sorted named aliases pointing at each model version (empty when none has been set). */
  const aliasByVersion = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const [name, entry] of Object.entries(aliasQ.data?.named_aliases ?? {})) {
      const names = map.get(entry.model_version) ?? [];
      names.push(name);
      map.set(entry.model_version, names);
    }
    for (const names of map.values()) names.sort();
    return map;
  }, [aliasQ.data]);

  /** Series/table label: the alias if one has been set, otherwise the raw version. */
  const displayNameOf = (version: string) => {
    const aliases = aliasByVersion.get(version);
    return aliases && aliases.length > 0 ? aliases.join(", ") : version;
  };

  /** Bundle picker label (Nested CV / Diagnostic curves): "{alias(es)}-{version}" when an alias is set, otherwise the version. */
  const bundleLabelOf = useBundleLabel(task);

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
  const pinballAlphaValue = Number(pinballAlpha);
  const validPinballAlpha = pinballAlphaValue > 0 && pinballAlphaValue < 1;
  const activePinballAlpha = activeMetric?.requires_alpha
    ? pinballAlphaValue
    : undefined;

  // Curves are expensive, so each metric tab loads on demand only while it is open.
  const curvesQ = useQuery({
    queryKey: ["learning-curves", task, tab, effectiveTrainPartition, modelVersions, activePinballAlpha],
    queryFn: () => getLearningCurves(task, {
      scoring: tab,
      pinballAlpha: activePinballAlpha,
      trainPartition: effectiveTrainPartition || undefined,
      modelVersions,
    }),
    enabled: !!task && activeMetric !== null && (!activeMetric.requires_alpha || validPinballAlpha),
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
          <NestedCVCard task={task} bundles={bundles} labelOf={bundleLabelOf} />

          {datasetQ.data?.temporal?.declared ? (
            <BacktestCard task={task} bundles={bundles} labelOf={bundleLabelOf} />
          ) : datasetQ.data ? (
            <Card
              title="Rolling-origin backtest (forecasting only)"
              sub="Walk origins forward in time and score multi-step forecasts against naive baselines on identical origins."
            >
              <p className="muted">
                Task “{task}” declares no temporal spec, so there is no time order to walk and no
                horizon to score. Declare the time column (and optional series id) on the Datasets
                page to enable backtesting and multi-step forecasts; for a tabular task, the nested CV
                above and the Diagnostic curves below are the evaluation protocols that apply.
              </p>
            </Card>
          ) : null}

          <CurvesCard task={task} bundles={bundles} labelOf={bundleLabelOf} />

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
                      <Bar key={b.model_version} dataKey={b.model_version} name={displayNameOf(b.model_version)} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[3, 3, 0, 0]} />
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
                    <input value={effectiveTrainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
                  </div>
                  {activeMetric.requires_alpha && (
                    <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
                      <label>Pinball quantile (0–1)</label>
                      <input
                        type="number"
                        min="0.001"
                        max="0.999"
                        step="0.05"
                        value={pinballAlpha}
                        onChange={(e) => setPinballAlpha(e.target.value)}
                      />
                    </div>
                  )}
                </div>

                {activeMetric.requires_alpha && !validPinballAlpha && (
                  <p className="muted">Enter a quantile strictly between 0 and 1 to calculate this curve.</p>
                )}
                {curvesQ.isLoading && <Loading label="Fitting learning curves…" />}
                {curvesQ.isError && <ErrorNote error={curvesQ.error} />}

                {curvesQ.data && curvesQ.data.series.length === 0 && (
                  <EmptyState
                    title="No learning curves available"
                    hint={`No bundle could be scored for ${activeMetric.label} on the “${effectiveTrainPartition}” partition.`}
                  />
                )}

                {curveRows.length > 0 && (
                  <>
                    <h4 className="hm-title">
                      {activeMetric.label} vs training size — {lcMetricsQ.data?.family ?? "unknown family"}
                      {activeMetric.requires_alpha && ` (q=${pinballAlpha})`}
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
                              name={`${displayNameOf(s.model_version)} (validation)`}
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
                              name={`${displayNameOf(s.model_version)} (train)`}
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

          <Card
            title="Bundle metrics"
            sub="Only the metrics a bundle actually records. Experiments and pipeline optimization store train_score; ensembles add cv_score / cv_score_std; forecasting baselines and any bundle you run a Rolling-origin backtest on add mae / rmse / mase / sMAPE, stored with the protocol they were measured under. Nested CV stores its honest outer score on the bundle's evaluation record."
          >
            <Table<BundleSummary>
              columns={["Model version", "Alias", ...metricNames, "Created"]}
              rows={bundles}
              keyOf={(b) => b.model_version}
              render={(b) => [
                <td key="v" className="mono">{b.model_version}</td>,
                <td key="alias">
                  {(() => {
                    const aliases = aliasByVersion.get(b.model_version);
                    return aliases && aliases.length > 0 ? aliases.join(", ") : <span className="muted">—</span>;
                  })()}
                </td>,
                ...metricNames.map((m) => {
                  const value = b.metrics[m];
                  const recorded = typeof value === "number";
                  const provenance = b.metrics_provenance?.[m];
                  const title = recorded
                    ? provenance
                      ? `${m}: ${provenance.scope ?? provenance.source}` +
                        (provenance.horizon != null ? ` · horizon ${provenance.horizon}` : "") +
                        (provenance.n_splits != null ? ` · ${provenance.n_splits} origins` : "") +
                        (provenance.strategy ? ` · ${provenance.strategy}` : "") +
                        (provenance.recorded_at ? ` · ${new Date(provenance.recorded_at).toLocaleString()}` : "")
                      : METRIC_ORIGIN[m]
                    : `${m} was not recorded by this bundle's producer (${METRIC_ORIGIN[m] ?? "see the note above"}). Run a Rolling-origin backtest to measure it.`;
                  return (
                    <td key={m} title={title}>
                      {fmtNum(value)}
                    </td>
                  );
                }),
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
