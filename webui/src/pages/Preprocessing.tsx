import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  clipDataset,
  deduplicateDataset,
  getDatasetMetadata,
  getDuplicates,
  getEncodingPreview,
  getPreprocessing,
  imputeDataset,
} from "../api/client";
import type {
  ClipReport,
  DuplicateAnalysis,
  EdaColumn,
  EdaNumericDistribution,
  EncodingPreview,
  ImputationReport,
  Multicollinearity,
  PreprocessingColumn,
  PreprocessingOverview,
  TargetComposition,
} from "../api/client";
import { Badge, Card, EmptyState, ErrorNote, Loading, Table, fmtNum, fmtSci } from "../components/ui";
import { Heatmap } from "../components/Heatmap";
import { useTask } from "../lib/task-context";

function pct(value: number | null | undefined): string {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function num(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function summaryLine(summary: PreprocessingColumn["numeric_summary"]): string {
  if (!summary) return "—";
  return `mean ${num(summary.mean)} · sd ${num(summary.std)} · q25 ${num(summary.q25)} · median ${num(summary.q50)} · q75 ${num(summary.q75)} · [${num(summary.min)}, ${num(summary.max)}]`;
}

function topLine(values: PreprocessingColumn["top_values"]): string {
  if (!values || values.length === 0) return "—";
  return values.map((entry) => `${entry.value} (${pct(entry.share)})`).join(", ");
}

function ReportView({ report }: { report: ImputationReport }) {
  const columns = report.columns;
  const numeric = Object.entries(report.numeric_shift);
  const categorical = Object.entries(report.categorical_shift);
  return (
    <>
      <dl className="kv">
        <dt>Rows</dt>
        <dd>
          {fmtNum(report.rows_before)} → {fmtNum(report.rows_after)}
          {report.rows_dropped > 0 && <span className="muted"> ({fmtNum(report.rows_dropped)} dropped)</span>}
        </dd>
        <dt>Cells imputed</dt>
        <dd>{fmtNum(report.cells_imputed)}</dd>
        <dt>Target missing</dt>
        <dd>
          {fmtNum(report.target_missing_before)} → {fmtNum(report.target_missing_after)}
          {report.target_missing_after > 0 && <span className="muted"> — target is never imputed</span>}
        </dd>
      </dl>
      <Table<string>
        columns={["Column", "Missing before", "Missing after"]}
        rows={columns}
        keyOf={(column) => column}
        render={(column) => [
          <td key="c" className="mono">{column}</td>,
          <td key="b">{fmtNum(report.missing_before[column] ?? 0)}</td>,
          <td key="a">{fmtNum(report.missing_after[column] ?? 0)}</td>,
        ]}
      />
      {numeric.length > 0 && (
        <>
          <h4 style={{ marginTop: 14 }}>Distribution shift (numeric)</h4>
          <Table<string>
            columns={["Column", "Mean before → after", "Std before → after"]}
            rows={numeric.map(([column]) => column)}
            keyOf={(column) => column}
            render={(column) => {
              const shift = report.numeric_shift[column];
              return [
                <td key="c" className="mono">{column}</td>,
                <td key="m">{num(shift.before?.mean)} → {num(shift.after?.mean)}</td>,
                <td key="s">{num(shift.before?.std)} → {num(shift.after?.std)}</td>,
              ];
            }}
          />
        </>
      )}
      {categorical.length > 0 && (
        <>
          <h4 style={{ marginTop: 14 }}>Category frequencies</h4>
          {categorical.map(([column, shift]) => (
            <p key={column} className="muted">
              <code>{column}</code>: before {topLine(shift.before)} → after {topLine(shift.after)}
            </p>
          ))}
        </>
      )}
    </>
  );
}

function TargetDistribution({ distribution }: { distribution: DuplicateAnalysis["target_before"] }) {
  if (distribution.kind === "categorical") return <>{topLine(distribution.values ?? null)}</>;
  return <>{summaryLine(distribution.summary ?? null)}</>;
}

function isNumericDistribution(
  distribution: EdaColumn["distribution"],
): distribution is EdaNumericDistribution {
  return "median" in distribution;
}

/**
 * Target distribution line. ``dataset_profile`` numeric summaries carry
 * ``median``, categorical ones carry ``top[].proportion`` — different shapes from
 * the ``missing_overview`` summaries the older helpers expected, so this reads
 * the profile shape directly instead of the overview one.
 */
function ProfileDistributionLine({ profile }: { profile: Record<string, unknown> }) {
  const distribution = profile.distribution as EdaColumn["distribution"] | undefined;
  if (!distribution) return <span>—</span>;
  if (isNumericDistribution(distribution)) {
    return (
      <span>
        mean {num(distribution.mean)} · sd {num(distribution.std)} · q25 {num(distribution.q25)} ·{" "}
        median {num(distribution.median)} · q75 {num(distribution.q75)} · [{num(distribution.min)},{" "}
        {num(distribution.max)}]
      </span>
    );
  }
  if (!distribution.top.length) return <span>—</span>;
  return (
    <span>
      {distribution.top
        .map((entry) => `${entry.value} (${pct(entry.proportion)})`)
        .join(", ")}
    </span>
  );
}

/** min → q25 → median → q75 → max rail; the box is the IQR, the tick the median. */
function QuantileRail({ distribution }: { distribution: EdaNumericDistribution }) {
  const { min, q25, median, q75, max } = distribution;
  if (min === null || max === null || q25 === null || median === null || q75 === null) {
    return <span className="muted">no finite values</span>;
  }
  const span = max - min;
  const at = (value: number) =>
    span > 0 ? Math.max(0, Math.min(100, ((value - min) / span) * 100)) : 0;
  return (
    <div
      style={{ position: "relative", height: 18, minWidth: 120 }}
      title={`min ${num(min)} · q25 ${num(q25)} · median ${num(median)} · q75 ${num(q75)} · max ${num(max)}`}
    >
      <div style={{ position: "absolute", left: 0, right: 0, top: 7, height: 6, background: "var(--bg-soft)", borderRadius: 3 }} />
      <div
        style={{
          position: "absolute",
          left: `${at(q25)}%`,
          width: `${Math.max(at(q75) - at(q25), 0.8)}%`,
          top: 7,
          height: 6,
          background: "var(--accent)",
          opacity: 0.6,
          borderRadius: 3,
        }}
      />
      <div style={{ position: "absolute", left: `calc(${at(median)}% - 1px)`, top: 2, width: 2, height: 16, background: "var(--text)" }} />
    </div>
  );
}

const DISTRIBUTION_ROWS = 25;

function DistributionCard({ data }: { data: PreprocessingOverview }) {
  const [showAll, setShowAll] = useState(false);
  const columns = data.column_profiles;
  const sampling = data.row_sampling as { truncated?: boolean; method?: string } | null;
  const shown = showAll ? columns : columns.slice(0, DISTRIBUTION_ROWS);
  return (
    <Card
      title="Distributions"
      sub="Per-column summaries from the bounded profile. The rail spans min → q25 → median → q75 → max; categorical bars are the most frequent levels. Descriptive only."
    >
      {sampling?.truncated && (
        <p className="hint">
          Computed on a row sample (method: {sampling.method ?? "without_replacement"}); the table
          below describes the sample, not the full partition.
        </p>
      )}
      {columns.length === 0 ? (
        <p className="muted">No active feature column to profile.</p>
      ) : (
        <Table<EdaColumn>
          columns={["Column", "Kind", "Distribution", "Outliers"]}
          rows={shown}
          keyOf={(column) => column.column}
          render={(column) => {
            const distribution = column.distribution;
            return [
              <td key="c" className="mono">
                {column.column}{" "}
                {column.constant && <Badge tone="warn">constant</Badge>}{" "}
                {column.all_missing && <Badge tone="err">all missing</Badge>}
              </td>,
              <td key="k">{column.kind}</td>,
              <td key="d">
                {isNumericDistribution(distribution) ? (
                  <>
                    <QuantileRail distribution={distribution} />
                    <span className="muted" style={{ display: "block", marginTop: 2 }}>
                      {num(distribution.min)} · {num(distribution.q25)} · {num(distribution.median)} ·{" "}
                      {num(distribution.q75)} · {num(distribution.max)}
                    </span>
                  </>
                ) : (
                  <>
                    {distribution.top.slice(0, 4).map((entry) => (
                      <div className="bar-row" key={entry.value} style={{ gridTemplateColumns: "1fr 90px 44px" }}>
                        <span className="mono" title={entry.value}>{entry.value}</span>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${Math.min(100, entry.proportion * 100)}%` }} />
                        </div>
                        <span className="mono">{pct(entry.proportion)}</span>
                      </div>
                    ))}
                    {distribution.other_count > 0 && (
                      <span className="hint">{fmtNum(distribution.other_count)} further value(s)</span>
                    )}
                  </>
                )}
              </td>,
              <td key="o">
                {isNumericDistribution(distribution) ? (
                  distribution.iqr_outlier_count > 0 ? (
                    <Badge tone="warn">
                      {fmtNum(distribution.iqr_outlier_count)} IQR ({pct(distribution.iqr_outlier_rate)})
                    </Badge>
                  ) : (
                    <span className="muted">0 IQR</span>
                  )
                ) : (
                  <span className="muted">{fmtNum(distribution.unique)} levels</span>
                )}
              </td>,
            ];
          }}
        />
      )}
      {!showAll && columns.length > DISTRIBUTION_ROWS && (
        <p className="hint">
          Showing {DISTRIBUTION_ROWS} of {fmtNum(columns.length)} profiled columns.{" "}
          <button className="link-btn" onClick={() => setShowAll(true)}>Show all</button>
        </p>
      )}
      {shown.some((column) => column.missing_rate > 0) && (
        <p className="hint">Distributions exclude missing values; see the missing-data overview for rates.</p>
      )}
    </Card>
  );
}

function TargetCard({ composition }: { composition: TargetComposition | null }) {
  if (!composition) {
    return (
      <Card title="Target analysis" sub="Composition of the modeled target.">
        <p className="muted">No target composition available for this partition.</p>
      </Card>
    );
  }
  if (composition.kind === "categorical") {
    const feasible = composition.stratification_feasible;
    return (
      <Card
        title="Target analysis"
        sub="Exact class counts over the full partition. Stratified k-fold needs every class to appear in every training fold."
      >
        <dl className="kv">
          <dt>Classes</dt>
          <dd>{fmtNum(composition.distinct_classes)}</dd>
          <dt>Minority share</dt>
          <dd>{pct(composition.minority_share)} ({fmtNum(composition.min_class_count)} rows)</dd>
          <dt>Imbalance ratio</dt>
          <dd>{composition.imbalance_ratio ? `${num(composition.imbalance_ratio, 2)} : 1` : "—"}</dd>
          <dt>Recommended folds</dt>
          <dd>
            {composition.recommended_folds ?? "none"}{" "}
            {feasible ? (
              <Badge tone="ok">stratification feasible</Badge>
            ) : (
              <Badge tone="err">a class occurs once — stratification impossible</Badge>
            )}
          </dd>
          <dt>Missing target</dt>
          <dd>{fmtNum(composition.missing)}</dd>
        </dl>
        <h4 style={{ marginTop: 14 }}>Class composition</h4>
        {composition.classes.map((entry) => (
          <div className="bar-row" key={entry.label}>
            <span className="mono" title={entry.label}>{entry.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.min(100, entry.share * 100)}%` }} />
            </div>
            <span className="mono">{pct(entry.share)}</span>
          </div>
        ))}
        {composition.classes_truncated > 0 && (
          <p className="hint">{fmtNum(composition.classes_truncated)} further class(es) not shown.</p>
        )}
        <p className="hint">{composition.interpretation}</p>
      </Card>
    );
  }
  const distribution = composition.distribution;
  return (
    <Card
      title="Target analysis"
      sub="Regression-shaped target: exact counts plus descriptive shape statistics from the bounded profile sample."
    >
      <dl className="kv">
        <dt>Distinct values</dt>
        <dd>
          {fmtNum(composition.distinct_values)}{" "}
          {composition.binary_like && <Badge tone="info">2-valued — classification-shaped</Badge>}
        </dd>
        <dt>Missing</dt>
        <dd>{fmtNum(composition.missing)}</dd>
        {distribution && (
          <>
            <dt>Mean · std</dt>
            <dd>{num(distribution.mean)} · {num(distribution.std)}</dd>
            <dt>Range</dt>
            <dd>[{num(distribution.min)}, {num(distribution.max)}]</dd>
            <dt>Skewness</dt>
            <dd>{num(distribution.skewness, 3)}</dd>
            <dt>Excess kurtosis</dt>
            <dd>{num(distribution.excess_kurtosis, 3)}</dd>
            <dt>IQR outliers</dt>
            <dd>
              {fmtNum(distribution.iqr_outlier_count)} ({pct(distribution.iqr_outlier_rate)})
            </dd>
          </>
        )}
      </dl>
      {distribution && (
        <>
          <h4 style={{ marginTop: 14 }}>Quantiles</h4>
          <QuantileRail distribution={distribution} />
        </>
      )}
      <p className="hint">{composition.interpretation}</p>
    </Card>
  );
}

const HEATMAP_FEATURES = 10;

function AssociationsCard({ data }: { data: PreprocessingOverview }) {
  const ranked = data.target_associations.filter((entry) => entry.value !== null);
  const maxAbs = Math.max(1e-9, ...ranked.map((entry) => Math.abs(entry.value ?? 0)));
  const heatmap = useMemo(() => {
    // Only Pearson pairs form one comparable matrix: correlation ratio and
    // Cramér's V live on different scales and must not share a colour ramp.
    const pearson = data.feature_associations.filter(
      (entry) => entry.method === "pearson" && entry.value !== null,
    );
    const strength = new Map<string, number>();
    for (const entry of pearson) {
      const value = Math.abs(entry.value ?? 0);
      strength.set(entry.feature_a, Math.max(strength.get(entry.feature_a) ?? 0, value));
      strength.set(entry.feature_b, Math.max(strength.get(entry.feature_b) ?? 0, value));
    }
    const labels = [...strength.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, HEATMAP_FEATURES)
      .map(([label]) => label);
    if (labels.length < 2) return null;
    const index = new Map(labels.map((label, position) => [label, position]));
    const values: Array<Array<number | null>> = labels.map((_, row) =>
      labels.map((__, column) => (row === column ? 1 : null)),
    );
    for (const entry of pearson) {
      const row = index.get(entry.feature_a);
      const column = index.get(entry.feature_b);
      if (row === undefined || column === undefined) continue;
      values[row][column] = entry.value;
      values[column][row] = entry.value;
    }
    return { labels, columns: labels, values };
  }, [data.feature_associations]);

  return (
    <Card
      title="Associations"
      sub="Exploratory association with the target and between features. Every row names its statistic; these are not causal effects or hypothesis tests."
    >
      {ranked.length === 0 ? (
        <p className="muted">No association could be computed from the profiled columns.</p>
      ) : (
        <>
          <h4>With the target</h4>
          <Table<(typeof ranked)[number]>
            columns={["Feature", "Statistic", "Value"]}
            rows={ranked.slice(0, 25)}
            keyOf={(entry) => entry.feature}
            render={(entry) => [
              <td key="f" className="mono">{entry.feature}</td>,
              <td key="m"><Badge tone="muted">{entry.method}</Badge></td>,
              <td key="v">
                <div className="bar-row" style={{ gridTemplateColumns: "120px 1fr 64px" }}>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${(Math.abs(entry.value ?? 0) / maxAbs) * 100}%` }}
                    />
                  </div>
                  <span className="mono">{num(entry.value, 3)}</span>
                </div>
              </td>,
            ]}
          />
        </>
      )}

      {heatmap && (
        <>
          <h4 style={{ marginTop: 16 }}>Numeric feature pairs (Pearson)</h4>
          <Heatmap data={heatmap} mode="diverging" />
          <p className="hint">
            Categorical and mixed pairs use correlation ratio or Cramér's V and are left out of this
            matrix so one colour ramp never mixes statistics.
          </p>
        </>
      )}

      {data.feature_associations_truncated > 0 && (
        <p className="hint">
          {fmtNum(data.feature_associations_truncated)} further feature pair(s) exceeded the{" "}
          {fmtNum(data.association_limit)}-pair payload limit.
        </p>
      )}

      {data.rank_associations?.target.length ? (
        <details style={{ marginTop: 12 }}>
          <summary className="muted">Rank association with the target (Spearman / Kendall)</summary>
          <Table<{ feature: string; method: string; value: number; n: number }>
            columns={["Feature", "Statistic", "Value", "n"]}
            rows={data.rank_associations.target}
            keyOf={(entry, index) => `${entry.feature}-${entry.method}-${index}`}
            render={(entry) => [
              <td key="f" className="mono">{entry.feature}</td>,
              <td key="m">{entry.method}</td>,
              <td key="v">{num(entry.value, 3)}</td>,
              <td key="n">{fmtNum(entry.n)}</td>,
            ]}
          />
          <p className="hint">{data.rank_associations.interpretation}</p>
        </details>
      ) : null}
    </Card>
  );
}

function vifTone(value: number | null): "ok" | "warn" | "err" | "muted" {
  if (value === null) return "muted";
  if (value >= 10) return "err";
  if (value >= 5) return "warn";
  return "ok";
}

function CollinearityCard({ multicollinearity }: { multicollinearity: Multicollinearity | null }) {
  return (
    <Card
      title="Collinearity"
      sub="Variance inflation factors and the design-matrix condition number on complete finite rows of the sampled columns."
    >
      {!multicollinearity || multicollinearity.numeric_features.length < 2 ? (
        <p className="muted">
          Needs at least two numeric feature columns; this partition profiles{" "}
          {fmtNum(multicollinearity?.numeric_features.length ?? 0)}.
        </p>
      ) : (
        <>
          <dl className="kv">
            <dt>Numeric features</dt>
            <dd>{fmtNum(multicollinearity.numeric_features.length)}</dd>
            <dt>Complete rows</dt>
            <dd>{fmtNum(multicollinearity.complete_case_rows)}</dd>
            <dt>Condition number</dt>
            <dd>
              {multicollinearity.condition_number === null
                ? (multicollinearity.condition_number_reason ?? "—")
                : fmtSci(multicollinearity.condition_number)}
            </dd>
          </dl>
          <h4 style={{ marginTop: 14 }}>Variance inflation factors</h4>
          <Table<Multicollinearity["vif"][number]>
            columns={["Feature", "VIF", "R²", "Reading"]}
            rows={multicollinearity.vif}
            keyOf={(entry) => entry.feature}
            render={(entry) => [
              <td key="f" className="mono">{entry.feature}</td>,
              <td key="v">
                {entry.value === null ? (
                  <Badge tone="err">{entry.reason ?? "undefined"}</Badge>
                ) : (
                  <Badge tone={vifTone(entry.value)}>{num(entry.value, 2)}</Badge>
                )}
              </td>,
              <td key="r">{num(entry.r_squared, 3)}</td>,
              <td key="t" className="muted">
                {entry.value === null
                  ? "near-perfect linear dependence"
                  : entry.value >= 10
                    ? "strong — consider dropping or combining"
                    : entry.value >= 5
                      ? "moderate"
                      : "low"}
              </td>,
            ]}
          />
          <p className="hint">{multicollinearity.scope}</p>
          <p className="hint">{multicollinearity.interpretation}</p>
        </>
      )}
    </Card>
  );
}

function EncodingCard({ task, partition }: { task: string; partition: string }) {
  const previewQ = useQuery<EncodingPreview>({
    queryKey: ["encoding-preview", task, partition],
    queryFn: () => getEncodingPreview(task, partition),
    enabled: !!task && !!partition,
  });
  const data = previewQ.data;
  return (
    <Card
      title="Categorical encoding preview"
      sub="The exact feature matrix the model pipeline builds from this partition. Learned inside each training fold, not once on the whole dataset."
    >
      {previewQ.isLoading && <Loading />}
      {previewQ.isError && <ErrorNote error={previewQ.error} />}
      {data && (
        <>
          <dl className="kv">
            <dt>Input features</dt>
            <dd>{fmtNum(data.input_feature_count)}</dd>
            <dt>Encoded features</dt>
            <dd>
              {fmtNum(data.output_feature_count)}
              {data.sampled && <span className="muted"> · vocabulary learned on {fmtNum(data.sample_rows)} sampled rows</span>}
            </dd>
          </dl>
          <Table<(typeof data.per_column)[number]>
            columns={["Column", "Detected kind", "Encoding", "Levels", "Output features", "Encoded names"]}
            rows={data.per_column}
            keyOf={(entry) => entry.column}
            render={(entry) => [
              <td key="c" className="mono">
                {entry.column}{" "}
                {entry.high_cardinality && <Badge tone="warn">high cardinality</Badge>}
              </td>,
              <td key="k">{entry.kind}</td>,
              <td key="e">{entry.encoding}</td>,
              <td key="l">
                {fmtNum(entry.cardinality)}
                {entry.missing > 0 && <span className="muted"> · {fmtNum(entry.missing)} missing</span>}
              </td>,
              <td key="o">{fmtNum(entry.output_feature_count)}</td>,
              <td key="n" className="mono muted">
                {entry.output_columns.length ? entry.output_columns.join(", ") : "—"}
                {entry.output_columns_truncated > 0 && ` … (+${entry.output_columns_truncated})`}
              </td>,
            ]}
          />
          {data.preview.length > 0 && (
            <>
              <h4 style={{ marginTop: 14 }}>Encoded rows (first {data.preview.length})</h4>
              <Table<Record<string, unknown>>
                columns={data.shown_output_features.slice(0, 12)}
                rows={data.preview}
                keyOf={(_, index) => index}
                render={(row) =>
                  data.shown_output_features.slice(0, 12).map((name) => (
                    <td key={name} className="mono">
                      {row[name] === null || row[name] === undefined
                        ? "—"
                        : typeof row[name] === "number"
                          ? num(row[name] as number, 3)
                          : String(row[name])}
                    </td>
                  ))
                }
              />
              {data.output_feature_count > 12 && (
                <p className="hint">Showing 12 of {fmtNum(data.output_feature_count)} encoded columns.</p>
              )}
            </>
          )}
          <ul className="hint" style={{ marginTop: 10 }}>
            {data.notes.map((note) => <li key={note}>{note}</li>)}
          </ul>
        </>
      )}
    </Card>
  );
}

function DuplicatesCard({
  task,
  partition,
  qc,
}: {
  task: string;
  partition: string;
  qc: ReturnType<typeof useQueryClient>;
}) {
  const duplicatesQ = useQuery<DuplicateAnalysis>({
    queryKey: ["duplicates", task, partition],
    queryFn: () => getDuplicates(task, partition),
    enabled: !!task && !!partition,
  });
  const [keep, setKeep] = useState<"first" | "last">("first");
  const [outputName, setOutputName] = useState(`${partition}_deduplicated`);
  useEffect(() => setOutputName(`${partition}_deduplicated`), [partition]);

  const previewMut = useMutation({
    mutationFn: () => deduplicateDataset(task, { partition, keep, preview_only: true }),
  });
  const saveMut = useMutation({
    mutationFn: () => deduplicateDataset(task, { partition, keep, output_partition: outputName.trim() || null }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
      qc.invalidateQueries({ queryKey: ["duplicates", task, partition] });
    },
  });
  const report = saveMut.data ?? previewMut.data ?? duplicatesQ.data;
  const analysis = duplicatesQ.data;

  return (
    <Card
      title="Duplicate rows"
      sub="Exact duplicate feature rows, their target consistency and what removing them would change. Removal is never silent: read the warnings first."
    >
      {duplicatesQ.isLoading && <Loading />}
      {duplicatesQ.isError && <ErrorNote error={duplicatesQ.error} />}
      {analysis && (
        <>
          <dl className="kv">
            <dt>Duplicate rows</dt>
            <dd>
              {fmtNum(analysis.duplicate_rows)} in {fmtNum(analysis.duplicate_groups)} group(s) ·{" "}
              {fmtNum(analysis.removable_rows)} removable
            </dd>
            <dt>Full-row duplicates</dt>
            <dd>{fmtNum(analysis.full_duplicate_rows)}</dd>
            <dt>Conflicting labels</dt>
            <dd>
              {fmtNum(analysis.conflicting_label_groups)} group(s), {fmtNum(analysis.conflicting_label_rows)} rows
              {analysis.conflicting_label_groups > 0 && <Badge tone="err">label contradiction</Badge>}
            </dd>
            <dt>Target before → after</dt>
            <dd>
              <TargetDistribution distribution={analysis.target_before} /> →{" "}
              <TargetDistribution distribution={analysis.target_after} />
            </dd>
            <dt>Rows before → after</dt>
            <dd>
              {fmtNum(analysis.rows_before)} → {fmtNum(analysis.rows_after)} (
              {fmtNum(analysis.rows_removed)} removed, {pct(analysis.rows_removed_pct)})
            </dd>
          </dl>

          {analysis.warnings.length > 0 && (
            <div className="warn-note" style={{ marginTop: 8 }}>
              <strong>Implications of removing duplicates</strong>
              <ul style={{ margin: "6px 0 0 18px" }}>
                {analysis.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </div>
          )}

          {Object.keys(analysis.numeric_shift).length > 0 && (
            <>
              <h4 style={{ marginTop: 14 }}>Numeric feature shift</h4>
              <Table<string>
                columns={["Feature", "Mean before → after", "Relative shift"]}
                rows={Object.keys(analysis.numeric_shift)}
                keyOf={(column) => column}
                render={(column) => {
                  const shift = analysis.numeric_shift[column];
                  return [
                    <td key="c" className="mono">{column}</td>,
                    <td key="m">{num(shift.before?.mean)} → {num(shift.after?.mean)}</td>,
                    <td key="d">
                      {shift.mean_delta_pct === null
                        ? "—"
                        : `${shift.mean_delta_pct > 0 ? "+" : ""}${(shift.mean_delta_pct * 100).toFixed(2)}%`}
                    </td>,
                  ];
                }}
              />
            </>
          )}

          <div className="row top" style={{ marginTop: 12 }}>
            <div className="field fixed" style={{ flex: "0 1 180px", minWidth: 140 }}>
              <label>Keep</label>
              <select value={keep} onChange={(e) => setKeep(e.target.value as "first" | "last")}>
                <option value="first">first occurrence</option>
                <option value="last">last occurrence</option>
              </select>
            </div>
            <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
              <label>Output partition</label>
              <input value={outputName} onChange={(e) => setOutputName(e.target.value)} spellCheck={false} />
              <span className="hint">Written as a new partition; the source is never modified.</span>
            </div>
          </div>
          {previewMut.isError && <ErrorNote error={previewMut.error} />}
          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          <div className="row">
            <button className="btn" onClick={() => previewMut.mutate()} disabled={previewMut.isPending || saveMut.isPending}>
              {previewMut.isPending ? "Computing…" : "Preview removal"}
            </button>
            <button
              className="btn primary"
              onClick={() => saveMut.mutate()}
              disabled={previewMut.isPending || saveMut.isPending || analysis.rows_removed === 0}
            >
              {saveMut.isPending ? "Saving…" : "Save deduplicated partition"}
            </button>
          </div>
          {saveMut.data?.written && (
            <div className="success-note">
              Wrote partition <code>{saveMut.data.output_partition}</code>
              {saveMut.data.dataset_fingerprint && (
                <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
              )}
            </div>
          )}
          {report && report !== analysis && (
            <>
              <h4 style={{ marginTop: 14 }}>{report.written ? "Saved result" : "Preview"}</h4>
              <dl className="kv">
                <dt>Rows</dt>
                <dd>{fmtNum(report.rows_before)} → {fmtNum(report.rows_after)}</dd>
                <dt>Removed</dt>
                <dd>{fmtNum(report.rows_removed)} ({pct(report.rows_removed_pct)})</dd>
              </dl>
            </>
          )}
        </>
      )}
    </Card>
  );
}

function ClipCard({
  task,
  partition,
  overview,
  qc,
}: {
  task: string;
  partition: string;
  overview: PreprocessingOverview;
  qc: ReturnType<typeof useQueryClient>;
}) {
  const [method, setMethod] = useState(overview.clip_methods[0]?.name ?? "iqr");
  const activeMethod = overview.clip_methods.find((entry) => entry.name === method);
  const [values, setValues] = useState<Record<string, string>>({});
  const numericColumns = useMemo(
    () =>
      overview.missing.columns.filter(
        (column) => !column.is_target && !column.ignored && column.numeric_summary !== null,
      ),
    [overview],
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [outputName, setOutputName] = useState(`${partition}_clipped`);

  useEffect(() => {
    setSelected(numericColumns.map((column) => column.column));
  }, [numericColumns]);

  useEffect(() => {
    setOutputName(`${partition}_clipped`);
    setValues({});
  }, [partition, method]);

  const params = useMemo(() => {
    const result: Record<string, number> = {};
    for (const param of activeMethod?.params ?? []) {
      const raw = values[param.name];
      const parsed = raw === undefined || raw === "" ? Number(param.default ?? 0) : Number(raw);
      result[param.name] = parsed;
    }
    return result;
  }, [activeMethod, values]);

  const previewMut = useMutation({
    mutationFn: () =>
      clipDataset(task, { partition, method, columns: selected, params, preview_only: true }),
  });
  const saveMut = useMutation({
    mutationFn: () =>
      clipDataset(task, {
        partition,
        method,
        columns: selected,
        params,
        output_partition: outputName.trim() || null,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
    },
  });
  const report: ClipReport | null = saveMut.data ?? previewMut.data ?? null;

  return (
    <Card
      title="Outlier clipping / winsorizing"
      sub="Cap extreme numeric values at robust bounds. Clipping changes the tails, so the preview reports how many values move and how the mean and standard deviation shift."
    >
      {numericColumns.length === 0 ? (
        <p className="muted">No numeric feature columns to clip.</p>
      ) : (
        <>
          <div className="row top">
            <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
              <label>Method</label>
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {overview.clip_methods.map((entry) => (
                  <option key={entry.name} value={entry.name}>{entry.label}</option>
                ))}
              </select>
            </div>
            {(activeMethod?.params ?? []).map((param) => (
              <div key={param.name} className="field fixed" style={{ flex: "0 1 130px", minWidth: 105 }}>
                <label>{param.name}</label>
                <input
                  type="number"
                  min={param.min}
                  max={param.max}
                  step="any"
                  value={values[param.name] ?? String(param.default ?? "")}
                  onChange={(e) => setValues((current) => ({ ...current, [param.name]: e.target.value }))}
                />
                {param.help && <span className="hint">{param.help}</span>}
              </div>
            ))}
            <div className="field fixed" style={{ flex: "0 1 190px", minWidth: 160 }}>
              <label>Output partition</label>
              <input value={outputName} onChange={(e) => setOutputName(e.target.value)} spellCheck={false} />
            </div>
          </div>

          <h4 style={{ marginTop: 12 }}>Numeric columns</h4>
          <Table<PreprocessingColumn>
            columns={["", "Column", "Median", "IQR range", "Min / max"]}
            rows={numericColumns}
            keyOf={(column) => column.column}
            render={(column) => [
              <td key="sel">
                <input
                  type="checkbox"
                  aria-label={`Clip ${column.column}`}
                  checked={selected.includes(column.column)}
                  onChange={() =>
                    setSelected((current) =>
                      current.includes(column.column)
                        ? current.filter((name) => name !== column.column)
                        : [...current, column.column],
                    )
                  }
                />
              </td>,
              <td key="c" className="mono">{column.column}</td>,
              <td key="m">{num(column.numeric_summary?.q50)}</td>,
              <td key="q">
                {num(column.numeric_summary?.q25)} … {num(column.numeric_summary?.q75)}
              </td>,
              <td key="r">{num(column.numeric_summary?.min)} / {num(column.numeric_summary?.max)}</td>,
            ]}
          />

          {previewMut.isError && <ErrorNote error={previewMut.error} />}
          {saveMut.isError && <ErrorNote error={saveMut.error} />}
          <div className="row">
            <button
              className="btn"
              onClick={() => previewMut.mutate()}
              disabled={selected.length === 0 || previewMut.isPending || saveMut.isPending}
            >
              {previewMut.isPending ? "Computing…" : "Preview clipping"}
            </button>
            <button
              className="btn primary"
              onClick={() => saveMut.mutate()}
              disabled={selected.length === 0 || previewMut.isPending || saveMut.isPending}
            >
              {saveMut.isPending ? "Saving…" : "Save clipped partition"}
            </button>
          </div>
          {saveMut.data?.written && (
            <div className="success-note">
              Wrote partition <code>{saveMut.data.output_partition}</code>
              {saveMut.data.dataset_fingerprint && (
                <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
              )}
            </div>
          )}
          {report && (
            <>
              <h4 style={{ marginTop: 14 }}>
                {report.written ? "Saved result" : "Preview"} · {report.method} · {fmtNum(report.cells_clipped)} values clipped
              </h4>
              <Table<string>
                columns={["Column", "Bounds", "Clipped low / high", "Share", "Mean before → after", "Std before → after"]}
                rows={report.columns}
                keyOf={(column) => column}
                render={(column) => {
                  const stats = report.columns_report[column];
                  return [
                    <td key="c" className="mono">{column}</td>,
                    <td key="b">{num(stats.lower)} … {num(stats.upper)}</td>,
                    <td key="n">{fmtNum(stats.clipped_low)} / {fmtNum(stats.clipped_high)}</td>,
                    <td key="s">{pct(stats.clipped_pct)}</td>,
                    <td key="m">{num(stats.before?.mean)} → {num(stats.after?.mean)}</td>,
                    <td key="d">{num(stats.before?.std)} → {num(stats.after?.std)}</td>,
                  ];
                }}
              />
            </>
          )}
        </>
      )}
    </Card>
  );
}

export default function PreprocessingPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const metaQ = useQuery({
    queryKey: ["dataset-meta", task],
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

  const [partition, setPartition] = useState("train");
  useEffect(() => {
    if (partitions.length && !partitions.includes(partition)) setPartition(partitions[0]);
  }, [partitions, partition]);

  const prepQ = useQuery<PreprocessingOverview>({
    queryKey: ["preprocessing", task, partition],
    queryFn: () => getPreprocessing(task, partition),
    enabled: !!task && !!partition && partitions.includes(partition),
  });
  const data = prepQ.data;

  const [method, setMethod] = useState("most_frequent");
  const [selected, setSelected] = useState<string[]>([]);
  const [constantValue, setConstantValue] = useState("");
  const [nNeighbors, setNNeighbors] = useState("5");
  const [maxIter, setMaxIter] = useState("10");
  const [outputName, setOutputName] = useState("train_imputed");
  const [dropTarget, setDropTarget] = useState(false);

  // Default selection: every active (non-target, non-ignored) column with
  // missing values. The user can narrow it; ignored columns are not offered.
  useEffect(() => {
    if (!data) return;
    setSelected(
      data.missing.columns
        .filter((column) => !column.is_target && !column.ignored && column.missing > 0)
        .map((column) => column.column),
    );
  }, [data]);

  useEffect(() => {
    setOutputName(`${partition}_imputed`);
    setDropTarget(false);
  }, [partition]);

  const selectedMethod = data?.methods.find((entry) => entry.name === method);
  const requiresNumeric = selectedMethod?.numeric_only ?? false;
  const numericSelected =
    !requiresNumeric ||
    selected.every((name) => {
      const column = data?.missing.columns.find((entry) => entry.column === name);
      return column ? column.numeric_summary !== null : false;
    });

  const params = useMemo(() => {
    if (method === "constant") return { constant_value: constantValue };
    if (method === "knn") return { n_neighbors: Number(nNeighbors) };
    if (method === "iterative") return { max_iter: Number(maxIter) };
    return {};
  }, [method, constantValue, nNeighbors, maxIter]);

  const body = (previewOnly: boolean) => ({
    partition,
    method,
    columns: selected,
    params,
    output_partition: outputName.trim() || null,
    preview_only: previewOnly,
    drop_missing_target: dropTarget,
  });

  const previewMut = useMutation({
    mutationFn: () => imputeDataset(task, body(true)),
  });
  const saveMut = useMutation({
    mutationFn: () => imputeDataset(task, body(false)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      qc.invalidateQueries({ queryKey: ["preprocessing", task] });
    },
  });
  const report = saveMut.data ?? previewMut.data ?? null;
  const canSubmit =
    !!data && selected.length > 0 && !!method && (method !== "constant" || constantValue.trim() !== "");

  const toggleColumn = (column: string) => {
    setSelected((current) =>
      current.includes(column) ? current.filter((name) => name !== column) : [...current, column],
    );
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Data Preprocessing</h1>
        <span className="desc">
          Missing-data profile and materialized imputation for task “{task || "…"}”. The source
          partition is never modified; imputed data is saved as a new partition.
        </span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}
      {task && metaQ.data === null && (
        <EmptyState
          title="No dataset registered for this task yet"
          hint="Register a CSV partition on the Datasets page first."
        />
      )}

      {task && partitions.length > 0 && (
        <Card title="Partition">
          <div className="row fixed">
            <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
              <label>Stored partition</label>
              <select value={partition} onChange={(e) => setPartition(e.target.value)}>
                {partitions.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </div>
            {data && (
              <span className="muted" style={{ alignSelf: "flex-end", marginBottom: 8 }}>
                target <code>{data.target}</code> · {fmtNum(data.rows)} rows · {fmtNum(data.columns.length)} columns
              </span>
            )}
          </div>
        </Card>
      )}

      {prepQ.isLoading && <Loading />}
      {prepQ.isError && <ErrorNote error={prepQ.error} />}

      {data && (
        <>
          <Card title="Missing data overview" sub="Per-column counts and the share of complete rows. Ignored columns are excluded from modeling but shown here.">
            <dl className="kv">
              <dt>Rows</dt>
              <dd>{fmtNum(data.missing.rows)}</dd>
              <dt>Missing cells</dt>
              <dd>
                {fmtNum(data.missing.missing_cells)} of {fmtNum(data.missing.total_cells)} ({pct(data.missing.missing_pct)})
              </dd>
              <dt>Complete rows</dt>
              <dd>
                {fmtNum(data.missing.complete_rows)} ({pct(data.missing.complete_pct)}) ·{" "}
                {fmtNum(data.missing.rows_with_any_missing)} rows ({pct(data.missing.rows_with_any_missing_pct)}) have a missing feature
              </dd>
              <dt>Target missing</dt>
              <dd>{fmtNum(data.missing.target_missing)}</dd>
            </dl>

            <h4 style={{ marginTop: 14 }}>Columns and implied types</h4>
            <Table<PreprocessingColumn>
              columns={["Column", "Detected type", "Implied type", "Missing", "Unique", "Distribution"]}
              rows={data.missing.columns}
              keyOf={(column) => column.column}
              render={(column) => [
                <td key="c" className="mono">
                  {column.column}{" "}
                  {column.is_target && <Badge tone="info">target</Badge>}{" "}
                  {column.ignored && <Badge tone="warn">ignored</Badge>}{" "}
                  {column.all_missing && <Badge tone="err">all missing</Badge>}
                  {!column.all_missing && column.constant && <Badge tone="warn">constant</Badge>}
                </td>,
                <td key="d" className="mono">{column.dtype}</td>,
                <td key="i">{column.deduced_type ?? "—"}</td>,
                <td key="m">
                  {fmtNum(column.missing)} ({pct(column.missing_pct)})
                </td>,
                <td key="u">{fmtNum(column.unique)}</td>,
                <td key="dist" className="muted">
                  {column.numeric_summary ? summaryLine(column.numeric_summary) : topLine(column.top_values)}
                </td>,
              ]}
            />
            {data.profile_features_truncated > 0 && (
              <p className="hint">
                {fmtNum(data.profile_features_truncated)} further feature(s) are not profiled here (limit 100);
                per-column missingness above still covers every column.
              </p>
            )}
          </Card>

          <DistributionCard data={data} />

          <TargetCard composition={data.target_composition} />

          <AssociationsCard data={data} />

          <CollinearityCard multicollinearity={data.multicollinearity} />

          <Card title="Data quality" sub="Computed on a bounded row sample; duplicates and constants are descriptive flags, not automatic exclusions.">
            {data.quality ? (
              <dl className="kv">
                <dt>Duplicate rows</dt>
                <dd>
                  {fmtNum(data.quality.duplicate_rows.count)} ({pct(data.quality.duplicate_rows.rate)})
                </dd>
                <dt>Constant features</dt>
                <dd className="mono">
                  {data.quality.constant_features.length ? data.quality.constant_features.join(", ") : "—"}
                </dd>
                <dt>All-missing features</dt>
                <dd className="mono">
                  {data.quality.all_missing_features.length ? data.quality.all_missing_features.join(", ") : "—"}
                </dd>
                <dt>Duplicate features</dt>
                <dd className="mono">
                  {data.quality.duplicate_features.length
                    ? data.quality.duplicate_features
                        .map((entry) => `${entry.feature} = ${entry.duplicate_of}`)
                        .join(", ")
                    : "—"}
                </dd>
                {data.target_profile && (
                  <>
                    <dt>Target distribution</dt>
                    <dd className="muted">
                      <ProfileDistributionLine
                        profile={data.target_profile as Record<string, unknown>}
                      />
                    </dd>
                  </>
                )}
              </dl>
            ) : (
              <p className="muted">No active feature column to profile (all features are ignored or the dataset has only the target).</p>
            )}
            {data.missingness_patterns && data.missingness_patterns.patterns.length > 0 && (
              <>
                <h4 style={{ marginTop: 14 }}>Missingness patterns</h4>
                <Table<(typeof data.missingness_patterns.patterns)[number]>
                  columns={["Missing columns", "Rows", "Share"]}
                  rows={data.missingness_patterns.patterns}
                  keyOf={(pattern, index) => `${index}-${pattern.missing_features.join(",")}`}
                  render={(pattern) => [
                    <td key="f" className="mono">{pattern.missing_features.join(", ")}</td>,
                    <td key="c">{fmtNum(pattern.count)}</td>,
                    <td key="p">{pct(pattern.proportion)}</td>,
                  ]}
                />
              </>
            )}
          </Card>

          <DuplicatesCard task={task} partition={partition} qc={qc} />

          <EncodingCard task={task} partition={partition} />

          <Card title="Dataset preview" sub={`First ${fmtNum(data.preview.length)} rows of the stored partition. “—” is a missing value.`}>
            <Table<Record<string, unknown>>
              columns={data.columns}
              rows={data.preview}
              keyOf={(_, index) => index}
              render={(row) =>
                data.columns.map((column) => (
                  <td key={column} className="mono">
                    {row[column] === null || row[column] === undefined ? "—" : String(row[column])}
                  </td>
                ))
              }
            />
          </Card>

          <Card
            title="Impute missing data"
            sub="Preview the result first; saving writes an imputed copy as a NEW partition and records its provenance. The target is never imputed."
          >
            <div className="warn-note">
              Materialized imputation changes the data. Experiments and cross-validation still use
              fold-local imputation on the original partition unless you deliberately select the
              derived partition later.
            </div>
            <div className="row top">
              <div className="field fixed" style={{ flex: "0 1 300px", minWidth: 240 }}>
                <label>Method</label>
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                  {data.methods.map((entry) => (
                    <option key={entry.name} value={entry.name}>{entry.label}</option>
                  ))}
                </select>
                {requiresNumeric && (
                  <span className="hint">This method accepts numeric columns only.</span>
                )}
              </div>
              {method === "constant" && (
                <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 160 }}>
                  <label>Constant value</label>
                  <input value={constantValue} onChange={(e) => setConstantValue(e.target.value)} spellCheck={false} />
                </div>
              )}
              {method === "knn" && (
                <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 110 }}>
                  <label>Neighbours</label>
                  <input type="number" min={1} max={50} value={nNeighbors} onChange={(e) => setNNeighbors(e.target.value)} />
                </div>
              )}
              {method === "iterative" && (
                <div className="field fixed" style={{ flex: "0 1 140px", minWidth: 110 }}>
                  <label>Max iterations</label>
                  <input type="number" min={1} max={100} value={maxIter} onChange={(e) => setMaxIter(e.target.value)} />
                </div>
              )}
              <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
                <label>Output partition</label>
                <input
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  spellCheck={false}
                  disabled={saveMut.isPending}
                />
                <span className="hint">Must be a new partition name; the source is never overwritten.</span>
              </div>
            </div>

            <label className="checkbox-row" style={{ marginTop: 8 }}>
              <input type="checkbox" checked={dropTarget} onChange={(e) => setDropTarget(e.target.checked)} />
              Drop rows whose target value is missing
            </label>

            <h4 style={{ marginTop: 14 }}>Columns to impute</h4>
            {data.missing.columns.filter((column) => !column.is_target && !column.ignored).length === 0 ? (
              <p className="muted">No active feature columns to impute — ignored columns are not imputed.</p>
            ) : (
              <Table<PreprocessingColumn>
                columns={["", "Column", "Missing", "Type"]}
                rows={data.missing.columns.filter((column) => !column.is_target && !column.ignored)}
                keyOf={(column) => column.column}
                render={(column) => [
                  <td key="sel">
                    <input
                      type="checkbox"
                      aria-label={`Impute ${column.column}`}
                      checked={selected.includes(column.column)}
                      onChange={() => toggleColumn(column.column)}
                    />
                  </td>,
                  <td key="c" className="mono">{column.column}</td>,
                  <td key="m">{fmtNum(column.missing)} ({pct(column.missing_pct)})</td>,
                  <td key="t">{column.numeric_summary ? "numeric" : "categorical/text"}</td>,
                ]}
              />
            )}

            {!numericSelected && (
              <p className="warn-note">
                The selected method is numeric-only but the selection includes categorical/text columns.
                Deselect those or choose most-frequent/constant.
              </p>
            )}
            {previewMut.isError && <ErrorNote error={previewMut.error} />}
            {saveMut.isError && <ErrorNote error={saveMut.error} />}
            <div className="row">
              <button
                className="btn"
                disabled={!canSubmit || previewMut.isPending || saveMut.isPending}
                onClick={() => previewMut.mutate()}
              >
                {previewMut.isPending ? "Computing…" : "Preview imputation"}
              </button>
              <button
                className="btn primary"
                disabled={!canSubmit || saveMut.isPending || previewMut.isPending}
                onClick={() => saveMut.mutate()}
              >
                {saveMut.isPending ? "Saving…" : "Save as new partition"}
              </button>
            </div>
            {saveMut.data && (
              <div className="success-note">
                Wrote partition <code>{saveMut.data.output_partition}</code>
                {saveMut.data.dataset_fingerprint && (
                  <> · fingerprint <code>{saveMut.data.dataset_fingerprint.slice(0, 16)}…</code></>
                )}
              </div>
            )}
            {report && (
              <>
                <h4 style={{ marginTop: 16 }}>
                  {report.written ? "Saved result" : "Preview"} · {report.method}
                </h4>
                <ReportView report={report} />
              </>
            )}
          </Card>

          <ClipCard task={task} partition={partition} overview={data} qc={qc} />
        </>
      )}
    </div>
  );
}
