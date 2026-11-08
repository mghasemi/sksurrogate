import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
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

import { computeAle, explainBundle, explainRow, getIntervals, listBundles } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Card, EmptyState, ErrorNote, Loading, Stat, fmtNum, fmtPct } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function ExplainPage() {
  const { task } = useTask();
  const [version, setVersion] = useState("");
  const [alpha, setAlpha] = useState(0.1);
  const [groupBy, setGroupBy] = useState("");
  const [maxRows, setMaxRows] = useState(500);
  const [stabilitySamples, setStabilitySamples] = useState(1);
  const [aleBins, setAleBins] = useState(20);
  const [rowIndex, setRowIndex] = useState(0);
  const [rowOrders, setRowOrders] = useState(5);

  const bundlesQ = useQuery({
    queryKey: ["bundles", task],
    queryFn: () => listBundles(task),
    enabled: !!task,
  });
  const bundles = bundlesQ.data?.bundles ?? [];
  const selected = version || bundles[0] || "";

  const explainMut = useMutation({
    mutationFn: () =>
      explainBundle(task, selected, {
        max_rows: maxRows,
        top_k: 3,
        stability_samples: stabilitySamples,
      }),
  });
  const intervalMut = useMutation({
    mutationFn: () => getIntervals(task, selected, { alpha, group_by: groupBy.trim() || undefined }),
  });
  const aleMut = useMutation({
    mutationFn: () =>
      computeAle(task, selected, {
        max_rows: maxRows,
        top_k: 3,
        n_bins: aleBins,
        stability_samples: stabilitySamples,
      }),
  });

  const rowMut = useMutation({
    mutationFn: () =>
      explainRow(task, selected, {
        row_index: rowIndex,
        orders: rowOrders,
        random_state: 0,
      }),
  });

  const importance = (explainMut.data?.importance ?? []).slice(0, 15);
  const intervals = intervalMut.data;
  const pdpStability = explainMut.data?.partial_dependence_stability;
  const pdpStabilityFor = (feature: string) =>
    pdpStability?.curves?.find((curve) => curve.feature === feature);

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
            title="Model behavior diagnostics"
            sub="Global model attribution (permutation score drop) and fitted-model response profiles (partial dependence); neither estimates causal effects. Optional reference-sample variation is descriptive, not a confidence interval."
            actions={
              <button className="btn primary" disabled={explainMut.isPending} onClick={() => explainMut.mutate()}>
                {explainMut.isPending ? "Computing…" : "Explain model"}
              </button>
            }
          >
            <p className="muted">
              Correlation summarizes observed associations; sensitivity indices
              describe responses over generated ranges; permutation importance
              measures model-score changes under feature shuffling. They need
              not agree and are not interchangeable.
              {" "}
              <Link to="/features">Compare with correlation and input sensitivity</Link>
            </p>
            <div className="row">
              <div className="field">
                <label htmlFor="explanation-max-rows">Maximum reference rows</label>
                <input
                  id="explanation-max-rows"
                  type="number"
                  min={20}
                  max={1000}
                  value={maxRows}
                  onChange={(event) => setMaxRows(Number(event.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="explanation-stability-samples">Reference samples</label>
                <select
                  id="explanation-stability-samples"
                  value={stabilitySamples}
                  onChange={(event) => setStabilitySamples(Number(event.target.value))}
                >
                  <option value={1}>1 (no sample variation estimate)</option>
                  <option value={3}>3 subsamples</option>
                  <option value={5}>5 subsamples</option>
                </select>
              </div>
            </div>
            <p className="muted">
              Comparing reference samples requires more source rows than the selected row cap.
            </p>
            {explainMut.isError && <ErrorNote error={explainMut.error} />}
            {explainMut.data && (
              <>
                <p className="muted">
                  Reference: {explainMut.data.reference_sample.used_rows} of{" "}
                  {explainMut.data.reference_sample.source_rows} rows from the “
                  {explainMut.data.reference_sample.partition}” partition
                  {explainMut.data.reference_sample.sampling === "random_without_replacement"
                    ? ` (sampled without replacement; seed ${explainMut.data.settings.random_state})`
                    : " (all rows)"}
                  . Permutation repeats: {explainMut.data.settings.n_repeats}.
                </p>
                {explainMut.data.reference_sample_stability && (
                  <>
                    <h3>Reference-sample variation</h3>
                    <p className="muted">
                      Descriptive variation across{" "}
                      {explainMut.data.reference_sample_stability.sample_count} random
                      distinct subsamples without replacement (each with{" "}
                      {explainMut.data.reference_sample_stability.sample_size} rows and{" "}
                      {explainMut.data.reference_sample_stability.permutation_repeats_per_sample} permutation
                      repeats); sample SD is not a confidence interval. Showing
                      the 10 features with the highest mean permutation importance.
                    </p>
                    <div className="grid cols-3">
                      {explainMut.data.reference_sample_stability.summary.slice(0, 10).map((item) => (
                        <Stat
                          key={item.feature}
                          label={item.feature}
                          value={fmtNum(item.mean_importance)}
                          sub={`sample SD ${fmtNum(item.sample_std)}; top-ranked in ${fmtPct(item.top_rank_frequency)} of samples`}
                        />
                      ))}
                    </div>
                  </>
                )}
                <ul className="muted">
                  {explainMut.data.interpretation.limitations.map((limitation) => (
                    <li key={limitation}>{limitation}</li>
                  ))}
                </ul>
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
                    const stability = pdpStabilityFor(curve.feature);
                    const data = curve.grid.map((g, i) => {
                      const mean = stability?.mean?.[i] ?? null;
                      const sd = stability?.sample_std?.[i] ?? null;
                      return {
                        x: g,
                        y: curve.average[i],
                        bandLow: mean !== null && sd !== null ? mean - sd : null,
                        bandHigh: mean !== null && sd !== null ? mean + sd : null,
                      };
                    });
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
                              {stability && (
                                <>
                                  <Line
                                    type="monotone"
                                    dataKey="bandLow"
                                    stroke="#94a3b8"
                                    strokeDasharray="4 4"
                                    dot={false}
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="bandHigh"
                                    stroke="#94a3b8"
                                    strokeDasharray="4 4"
                                    dot={false}
                                  />
                                </>
                              )}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {pdpStability?.curves && (
                  <p className="muted">
                    Dashed curves mark the mean ± one sample SD across{" "}
                    {pdpStability.sample_count} distinct reference subsets, each re-evaluated on
                    the same grid with direct model calls. The spread is descriptive — it is not a
                    confidence band and does not guarantee generalization.
                  </p>
                )}
                {pdpStability?.skipped_reason && (
                  <p className="hint">
                    Reference-subset PDP stability not computed: {pdpStability.skipped_reason}
                  </p>
                )}
                {explainMut.data.errors.map((e) => (
                  <div key={e.feature} className="hint">⚠ {e.feature}: {e.detail}</div>
                ))}
              </>
            )}
          </Card>

    {selected && (
        <Card
          title="Accumulated local effects (ALE)"
          sub="Local response differences accumulated over observed feature bins. A supplement to partial dependence when features are dependent — still a fitted-model response profile, not a causal effect."
          actions={
            <button className="btn primary" disabled={aleMut.isPending} onClick={() => aleMut.mutate()}>
              {aleMut.isPending ? "Computing…" : "Compute ALE"}
            </button>
          }
        >
          <p className="muted">
            ALE varies each feature only within its observed intervals and accumulates the local
            model-response change, so it averages over fewer implausible feature combinations than
            marginal partial dependence. It does not isolate variable importance, and nominal
            categorical features are skipped because they have no canonical ordering.
          </p>
          <div className="row">
            <div className="field">
              <label htmlFor="ale-bins">Quantile bins</label>
              <select
                id="ale-bins"
                value={aleBins}
                onChange={(event) => setAleBins(Number(event.target.value))}
              >
                <option value={10}>10 bins</option>
                <option value={20}>20 bins</option>
                <option value={30}>30 bins</option>
              </select>
            </div>
          </div>
          {aleMut.isError && <ErrorNote error={aleMut.error} />}
          {aleMut.data && (
            <>
              <p className="muted">
                Reference: {aleMut.data.reference_sample.used_rows} of{" "}
                {aleMut.data.reference_sample.source_rows} rows from the “
                {aleMut.data.reference_sample.partition}” partition
                {aleMut.data.reference_sample.sampling === "random_without_replacement"
                  ? ` (sampled without replacement; seed ${aleMut.data.settings.random_state})`
                  : " (all rows)"}
                . Response: <code>{aleMut.data.response.output}</code>. Features:{" "}
                {aleMut.data.feature_selection === "explicit"
                  ? "explicit selection"
                  : `top ${aleMut.data.settings.top_k} by permutation importance`}
                . {" "}
                <Link to="/features">Compare with correlation and input sensitivity</Link>
              </p>
              {aleMut.data.skipped.map((item) => (
                <div key={item.feature} className="hint">
                  ⚠ {item.feature}: {item.detail}
                </div>
              ))}
              {aleMut.data.errors.map((item) => (
                <div key={item.feature} className="hint">
                  ⚠ {item.feature}: {item.detail}
                </div>
              ))}
              <div className="grid cols-3">
                {aleMut.data.curves.map((curve) => {
                  const stability = curve.stability;
                  const data = curve.grid.map((g, i) => {
                    const mean = stability?.mean?.[i] ?? null;
                    const sd = stability?.sample_std?.[i] ?? null;
                    return {
                      x: g,
                      y: curve.accumulated[i],
                      bandLow: mean !== null && sd !== null ? mean - sd : null,
                      bandHigh: mean !== null && sd !== null ? mean + sd : null,
                    };
                  });
                  const total = curve.counts.reduce((sum, count) => sum + count, 0);
                  return (
                    <div key={curve.feature}>
                      <strong>{curve.feature}</strong>
                      <p className="hint">
                        {curve.counts.length} bins ({curve.counts.join(" / ")} rows
                        {curve.excluded_rows ? `; ${curve.excluded_rows} non-finite rows excluded` : ""}
                        {total ? "" : "; no reference rows"})
                      </p>
                      <div style={{ height: 160 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" tickFormatter={(v: number) => Number(v).toPrecision(3)} />
                            <YAxis width={48} domain={["auto", "auto"]} />
                            <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                            <Line type="monotone" dataKey="y" stroke="#6366f1" dot={false} />
                            {stability && (
                              <>
                                <Line
                                  type="monotone"
                                  dataKey="bandLow"
                                  stroke="#94a3b8"
                                  strokeDasharray="4 4"
                                  dot={false}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="bandHigh"
                                  stroke="#94a3b8"
                                  strokeDasharray="4 4"
                                  dot={false}
                                />
                              </>
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
              {aleMut.data.curves.some((curve) => curve.stability) && (
                <p className="muted">
                  Dashed curves mark the mean ± one sample SD across the requested reference
                  subsets, accumulated on the primary sample's bin edges. They are descriptive,
                  not confidence bands.
                </p>
              )}
              <ul className="muted">
                {aleMut.data.interpretation.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            </>
          )}
        </Card>
      )}


          <Card
            title="Row-level explanation"
            sub="Ordered-occlusion attribution for one stored row: each feature is credited with the model-response change when it is restored from a baseline row in a given order. The decomposition reconstructs the prediction exactly; the spread across orders discloses order-dependence and is not a confidence interval. This is a model attribution, not a causal effect."
            actions={
              <button className="btn primary" disabled={rowMut.isPending} onClick={() => rowMut.mutate()}>
                {rowMut.isPending ? "Computing…" : "Explain row"}
              </button>
            }
          >
            <p className="muted">
              Contributions telescope exactly, so the sum plus the baseline prediction equals the
              explained prediction for any estimator. Correlated features and interactions are split
              according to feature order; increase the number of orders to see how much each
              attribution depends on that order.
            </p>
            <div className="row">
              <div className="field">
                <label htmlFor="row-explain-index">Row index (0-based)</label>
                <input
                  id="row-explain-index"
                  type="number"
                  min={0}
                  value={rowIndex}
                  onChange={(event) => setRowIndex(Number(event.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="row-explain-orders">Feature orders</label>
                <select
                  id="row-explain-orders"
                  value={rowOrders}
                  onChange={(event) => setRowOrders(Number(event.target.value))}
                >
                  <option value={1}>1 (single order, exact but arbitrary)</option>
                  <option value={5}>5 orders</option>
                  <option value={10}>10 orders</option>
                </select>
              </div>
            </div>
            {rowMut.isError && <ErrorNote error={rowMut.error} />}
            {rowMut.data && (
              <>
                <p className="muted">
                  Row {rowMut.data.row_index} of {rowMut.data.row_count} in “
                  {rowMut.data.partition}”; baseline {rowMut.data.baseline.mode === "row"
                    ? `is row ${rowMut.data.baseline.row_index} of “${rowMut.data.baseline.partition}”`
                    : `is the ${rowMut.data.reference_sample.used_rows}-row reference mean from “${rowMut.data.reference_sample.partition}”`}
                  . Response: <code>{rowMut.data.response.output}</code>;{" "}
                  {rowMut.data.orders.count} distinct feature order
                  {rowMut.data.orders.count === 1 ? "" : "s"}.
                </p>
                <div className="grid cols-3">
                  <Stat label="Baseline prediction" value={fmtNum(rowMut.data.baseline.prediction)} />
                  <Stat label="Row prediction" value={fmtNum(rowMut.data.prediction)} />
                  <Stat
                    label={rowMut.data.explanation_scope === "full_row" ? "Explained prediction" : "Hybrid-row prediction"}
                    value={fmtNum(rowMut.data.explained_prediction)}
                  />
                  <Stat label="Sum of contributions" value={fmtNum(rowMut.data.reconstruction.sum_contributions)} />
                  <Stat
                    label="Reconstruction residual"
                    value={fmtNum(rowMut.data.reconstruction.max_abs_residual)}
                    sub={`tolerance ${fmtNum(rowMut.data.reconstruction.tolerance)}; exact by construction`}
                  />
                </div>
                {rowMut.data.explanation_scope === "subset_hybrid" && (
                  <p className="hint">
                    Only the explained features move off the baseline; the reconstruction describes
                    the hybrid row, not the full prediction.
                  </p>
                )}
                <div style={{ height: Math.max(180, rowMut.data.contributions.length * 26) }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={rowMut.data.contributions}
                      layout="vertical"
                      margin={{ left: 24, right: 16 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={110} interval={0} />
                      <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                      <Bar dataKey="mean" fill="#0ea5e9" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Mean contribution</th>
                        <th>Sample SD across orders</th>
                        <th>Min</th>
                        <th>Max</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rowMut.data.contributions.map((item) => (
                        <tr key={item.feature}>
                          <td>{item.feature}</td>
                          <td>{fmtNum(item.mean)}</td>
                          <td>{fmtNum(item.sample_std)}</td>
                          <td>{fmtNum(item.min)}</td>
                          <td>{fmtNum(item.max)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <ul className="muted">
                  {rowMut.data.interpretation.limitations.map((limitation) => (
                    <li key={limitation}>{limitation}</li>
                  ))}
                </ul>
              </>
            )}
          </Card>

          <Card
            title="Prediction intervals"
            sub="Regression models only. Split-conformal intervals are evaluated on held-out rows; observed coverage is descriptive and shown with a Wilson interval."
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
              <div className="field">
                <label>Coverage subgroup (optional)</label>
                <input
                  value={groupBy}
                  onChange={(e) => setGroupBy(e.target.value)}
                  placeholder="Categorical feature column"
                />
              </div>
            </div>
            {intervalMut.isError && <ErrorNote error={intervalMut.error} />}
            {intervals && (
              <>
                <div className="grid cols-3">
                  <Stat label="Target coverage" value={fmtPct(intervals.nominal_coverage)} />
                  <Stat
                    label="Observed coverage"
                    value={fmtPct(intervals.empirical_coverage)}
                    sub={`Wilson ${(intervals.coverage_interval.confidence_level * 100).toFixed(0)}%: ${fmtPct(intervals.coverage_interval.lower)}–${fmtPct(intervals.coverage_interval.upper)}; ${intervals.holdout_rows} rows`}
                  />
                  <Stat label="Interval half-width" value={fmtNum(intervals.half_width)} sub="± around each prediction" />
                  <Stat label="Mean interval width" value={fmtNum(intervals.mean_interval_width)} sub={`median ${fmtNum(intervals.median_interval_width)}`} />
                  <Stat label="Mean interval score" value={fmtNum(intervals.mean_interval_score)} sub={intervals.interval_score_method} />
                  <Stat label="Calibration rows" value={fmtNum(intervals.calibration_size)} />
                </div>
                <p className="muted">{intervals.coverage_assumption}</p>
                <p className="muted">
                  Fit and calibration rows came from “{intervals.calibration_source_partition}”
                  ({intervals.fit_rows} fit, {intervals.calibration_size} calibration); held-out
                  rows came from “{intervals.holdout_partition}”. {intervals.diagnostic_assumption}
                </p>
                {intervals.subgroup_coverage && (
                  <>
                    <h3>Coverage by {intervals.group_by}</h3>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Group</th>
                            <th>Rows</th>
                            <th>Observed coverage</th>
                            <th>Wilson interval</th>
                          </tr>
                        </thead>
                        <tbody>
                          {intervals.subgroup_coverage.groups.map((group) => (
                            <tr key={group.group}>
                              <td>{group.group}</td>
                              <td>{fmtNum(group.row_count)}</td>
                              <td>{fmtPct(group.empirical_coverage)}</td>
                              <td>
                                {fmtPct(group.coverage_interval.lower)}–{fmtPct(group.coverage_interval.upper)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="muted">{intervals.subgroup_coverage.interpretation}</p>
                  </>
                )}
              </>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
