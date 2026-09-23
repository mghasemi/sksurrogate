/**
 * Typed client for the SKSurrogate control-plane API.
 *
 * Requests default to a *same-origin* base ("") so the Vite dev server's
 * ``/api`` proxy (see vite.config.ts) forwards them to the backend. That keeps
 * the UI working wherever it is served from — http://localhost:5174, the
 * container's or host's LAN IP, or a forwarded/tunnel URL — and sidesteps both
 * CORS and mixed-content failures, because the browser only ever talks to its
 * own origin.
 *
 * Point at an absolute API URL instead by setting VITE_API_BASE_URL, or with a
 * runtime override saved from the Settings page (see setApiBase).
 */

/** Same-origin by default: the dev server proxies /api to the backend. */
const DEFAULT_BASE = "";

/** Explicitly configured base, or "" for same-origin. */
function configuredBase(): string {
  const env = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return (localStorage.getItem("sksurrogate_api_base") || env || DEFAULT_BASE).replace(/\/+$/, "");
}

/** Base used to build request URLs ("" means same-origin). */
function base(): string {
  return configuredBase();
}

/** The configured API base; an empty string means "this page's own origin". */
export function getApiBase(): string {
  return configuredBase();
}

/**
 * Absolute API base (never empty), for URLs that need a scheme — WebSocket
 * handshakes and read-only display. Falls back to the page's own origin, which
 * is what the dev-server proxy answers on.
 */
export function getEffectiveApiBase(): string {
  return configuredBase() || window.location.origin;
}

/** Save an absolute API base; an empty value clears the override (back to same-origin). */
export function setApiBase(url: string): void {
  const trimmed = url.replace(/\/+$/, "");
  if (trimmed) localStorage.setItem("sksurrogate_api_base", trimmed);
  else localStorage.removeItem("sksurrogate_api_base");
}

/** Shared API key, sent as an X-API-Key header (and WS query param) when the server requires one. */
export function getApiKey(): string {
  return localStorage.getItem("sksurrogate_api_key") || "";
}

export function setApiKey(key: string): void {
  if (key.trim()) localStorage.setItem("sksurrogate_api_key", key.trim());
  else localStorage.removeItem("sksurrogate_api_key");
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * Retry predicate for queries where a 404 means "nothing registered yet"
 * rather than a real failure — retry everything except not-found responses.
 * TanStack Query v5 invokes this as (failureCount, error), so the error is
 * the SECOND argument; treating the first arg as the error would make every
 * 404 retry forever and leave queries stuck in isLoading.
 */
export const retryUnlessNotFound = (_failureCount: number, err: unknown): boolean =>
  !(err instanceof ApiError && err.status === 404);

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers = authHeaders();
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(`${base()}${path}`, {
    method,
    headers: Object.keys(headers).length ? headers : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
      else if (data.detail !== undefined) detail = JSON.stringify(data.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, `${method} ${path} -> ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {});

/* ------------------------------------------------------------------ */
/* Shared shapes                                                       */
/* ------------------------------------------------------------------ */

export interface JobRecord {
  job_id: string;
  task_name: string;
  kind: "sensitivity" | "experiment" | "retraining" | string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface BundleSummary {
  task_name: string;
  model_version: string;
  created_at: string;
  schema: Record<string, { dtype: string }>;
  metrics: Record<string, number>;
  dependencies: unknown[];
  dataset_fingerprint: string | null;
  owner: string | null;
  run_id: string | null;
  audit_events: Array<Record<string, unknown>>;
}

export interface PartitionInfo {
  source_uri?: string | null;
  ingestion_timestamp: string;
  fingerprint: string;
  rows: number;
  columns: string[];
  target?: string;
  feature_count?: number;
}

/* ------------------------------------------------------------------ */
/* Health                                                              */
/* ------------------------------------------------------------------ */

export const health = () => get<{ status: string }>("/api/health");

/* ------------------------------------------------------------------ */
/* Tasks                                                               */
/* ------------------------------------------------------------------ */

/** Known task names discovered across the API storage layout. */
export const listTasks = () => get<{ tasks: string[] }>("/api/tasks");

/* ------------------------------------------------------------------ */
/* Datasets                                                            */
/* ------------------------------------------------------------------ */

export interface RegisterDatasetResponse {
  task_name: string;
  partition: string;
  target: string;
  rows: number;
  columns: string[];
  dataset_fingerprint: string | null;
  dataset_schema: Record<string, unknown> | null;
  deduced_types: Record<string, string>;
  /** Name-based PII / sensitive-column scan of the uploaded frame (Phase 2.4). */
  sensitive_scan?: SensitiveScanReport;
}

export interface DatasetMetadata {
  task_name: string;
  target_name: string | null;
  dataset_fingerprint: string | null;
  dataset_columns: string[] | null;
  dataset_schema: Record<string, unknown> | null;
  dataset_feature_count: number | null;
  partitions: Record<string, PartitionInfo>;
}

export interface DatasetPreview {
  task_name: string;
  partition: string;
  rows: Array<Record<string, unknown>>;
}

export async function registerDataset(
  taskName: string,
  target: string,
  partition: string,
  file: File,
): Promise<RegisterDatasetResponse> {
  const form = new FormData();
  form.append("target", target);
  form.append("partition", partition);
  form.append("file", file);
  const res = await fetch(`${base()}/api/datasets/${encodeURIComponent(taskName)}/register`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data.detail) detail = data.detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, `POST /api/datasets/${taskName}/register -> ${res.status}: ${detail}`);
  }
  return (await res.json()) as RegisterDatasetResponse;
}

export const getDatasetMetadata = (task: string) =>
  get<DatasetMetadata>(`/api/datasets/${encodeURIComponent(task)}`);

export const previewDataset = (task: string, partition: string, limit = 20) =>
  get<DatasetPreview>(
    `/api/datasets/${encodeURIComponent(task)}/${encodeURIComponent(partition)}/preview?limit=${limit}`,
  );

/** A scikit-learn cross-validation splitter spec (see SKSurrogate.mltrace.cv_to_spec). */
export type CVSpec = { type: string; [param: string]: unknown } | number;

/** One editable constructor parameter of a CV splitter. */
export interface CVParamDef {
  name: string;
  kind: "int" | "float" | "bool";
  default: unknown;
}

export interface CVOptionsResponse {
  task_name: string;
  options: string[];
  /** Constructor parameters per splitter, for rendering editable fields. */
  param_defs?: Record<string, CVParamDef[]>;
  current: CVSpec;
  stored: boolean;
}

export const getDatasetCV = (task: string) =>
  get<CVOptionsResponse>(`/api/datasets/${encodeURIComponent(task)}/cv`);

export const setDatasetCV = (task: string, spec: CVSpec, params?: Record<string, unknown>) =>
  request<{ task_name: string; cv: CVSpec }>("PUT", `/api/datasets/${encodeURIComponent(task)}/cv`, {
    spec,
    ...(params ? { params } : {}),
  });

/** Descriptive statistics of the registered target column (pandas describe()). */
export interface TargetStats {
  task_name: string;
  /** Partition the stats were computed from (train preferred). */
  partition: string;
  target: string;
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  "25%": number | null;
  "50%": number | null;
  "75%": number | null;
  max: number | null;
}

export const getTargetStats = (task: string) =>
  get<TargetStats>(`/api/datasets/${encodeURIComponent(task)}/target-stats`);

/* ------------------------------------------------------------------ */
/* Bundles                                                             */
/* ------------------------------------------------------------------ */

export interface TrainBaselineRequest {
  estimator: string;
  train_partition?: string;
  validation_partition?: string | null;
  owner?: string | null;
  run_id?: string | null;
}

export interface TrainBaselineResponse {
  task_name: string;
  model_version: string;
  estimator: string;
  metrics: Record<string, number>;
  schema: Record<string, { dtype: string }>;
  bundle_path: string;
}

export const BASELINE_ESTIMATORS = [
  "linear_regression",
  "logistic_regression",
  "random_forest_classifier",
  "random_forest_regressor",
] as const;

export const trainBaseline = (task: string, body: TrainBaselineRequest) =>
  post<TrainBaselineResponse>(`/api/bundles/${encodeURIComponent(task)}/train-baseline`, body);

export interface BundleList {
  task_name: string;
  bundles: string[];
}

export const listBundles = (task: string) => get<BundleList>(`/api/bundles/${encodeURIComponent(task)}`);

export const getBundle = (task: string, modelVersion: string) =>
  get<BundleSummary>(`/api/bundles/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}`);

/* ------------------------------------------------------------------ */
/* Quality gates                                                       */
/* ------------------------------------------------------------------ */

export interface QualityGateRequest {
  metric_thresholds?: Record<string, number>;
  max_bundle_size_bytes?: number | null;
}

export interface QualityCheck {
  name: string;
  passed: boolean;
  error?: string;
  metric?: string;
  minimum?: number;
  maximum?: number;
  value?: number | null;
}

export interface QualityGateReport {
  task_name: string;
  model_version: string;
  passed: boolean;
  checks: QualityCheck[];
  failures: QualityCheck[];
}

export const checkQuality = (task: string, modelVersion: string, body: QualityGateRequest) =>
  post<QualityGateReport>(
    `/api/quality-gates/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/check`,
    body,
  );

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

export const REGISTRY_STATES = ["candidate", "validated", "staging", "production", "archived"] as const;
export type RegistryState = (typeof REGISTRY_STATES)[number];

export interface PromoteRequest {
  model_version: string;
  state: string;
  approvers: string[];
  quality_report?: Record<string, unknown> | null;
  required_approvals?: number;
}

export const registerBundle = (task: string, modelVersion: string) =>
  post<{ task_name: string; model_version: string; state: string }>(
    `/api/registry/${encodeURIComponent(task)}/register`,
    { model_version: modelVersion },
  );

export const promoteBundle = (task: string, body: PromoteRequest) =>
  post<Record<string, unknown>>(`/api/registry/${encodeURIComponent(task)}/promote`, body);

export interface RollbackResult {
  status: string;
  task_name: string;
  state: string;
  model_version: string | null;
}

export const rollbackBundle = (task: string, state: string, modelVersion?: string) =>
  post<RollbackResult>(`/api/registry/${encodeURIComponent(task)}/rollback`, {
    state,
    model_version: modelVersion ?? null,
  });

export interface RegistryHistoryEntry {
  event_type: string;
  action: "promote" | "rollback";
  from?: string | null;
  to: string;
  model_version?: string;
  state?: string;
  timestamp: string;
}

export interface RegistrySummary {
  task_name: string;
  aliases: Record<string, string>;
  versions: Record<string, string>;
}

export const registrySummary = (task: string) =>
  get<RegistrySummary>(`/api/registry/${encodeURIComponent(task)}/summary`);

export const registryHistory = (task: string) =>
  get<{ task_name: string; history: RegistryHistoryEntry[] }>(`/api/registry/${encodeURIComponent(task)}/history`);

export interface AuditEvent {
  event_type: string;
  action?: string;
  model_version?: string;
  timestamp: string;
  details?: Record<string, unknown>;
  [key: string]: unknown;
}

export const registryAudit = (task: string) =>
  get<{ task_name: string; audit: AuditEvent[] }>(`/api/registry/${encodeURIComponent(task)}/audit`);

export interface RegisteredBundleView extends BundleSummary {
  alias: string;
}

export const loadRegisteredBundle = (task: string, alias = "production") =>
  get<RegisteredBundleView>(`/api/registry/${encodeURIComponent(task)}/load?alias=${encodeURIComponent(alias)}`);

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */

export interface PredictRequest {
  model_version?: string | null;
  alias?: string | null;
  partition?: string | null;
  rows?: Array<Record<string, unknown>> | null;
  request_id?: string | null;
}

export interface InferenceMetrics {
  latency_ms: number;
  rows: number;
  [key: string]: number;
}

export interface PredictResponse {
  task_name: string;
  model_version: string;
  request_id: string;
  predictions: Array<number | string>;
  metrics: InferenceMetrics;
  /** Columns present in the input but not in the bundle schema (e.g. the target label). */
  ignored_columns: string[];
}

export const predict = (task: string, body: PredictRequest) =>
  post<PredictResponse>(`/api/inference/${encodeURIComponent(task)}/predict`, body);

export interface PredictBatchResponse {
  task_name: string;
  model_version: string;
  request_id: string;
  rows: number;
  metrics: InferenceMetrics;
  output_path: string;
  /** Columns present in the input but not in the bundle schema (e.g. the target label). */
  ignored_columns: string[];
}

export const predictBatch = (task: string, body: PredictRequest) =>
  post<PredictBatchResponse>(`/api/inference/${encodeURIComponent(task)}/predict-batch`, body);

/* ------------------------------------------------------------------ */
/* Monitoring                                                          */
/* ------------------------------------------------------------------ */

export interface DriftRequest {
  reference_partition: string;
  current_partition: string;
  model_version?: string | null;
  psi_threshold?: number;
  categorical_threshold?: number;
  missingness_threshold?: number;
  range_threshold?: number;
}

export interface DriftAlert {
  type: "missingness" | "numeric_drift" | "categorical_drift" | "range_drift";
  column: string;
  value: number;
}

export interface DriftReport {
  model_version: string | null;
  schema_changes: {
    missing_columns: string[];
    extra_columns: string[];
    reordered: boolean;
    dtype_changes: Record<string, { reference: string; current: string }>;
  };
  data_quality: {
    reference_rows: number;
    current_rows: number;
    missingness: Record<string, { reference: number; current: number; delta: number }>;
  };
  drift: {
    numeric: Record<string, { method: string; value: number | null; bins?: number[] }>;
    categorical: Record<
      string,
      { metric: string; value: number; reference_distribution: Record<string, number>; current_distribution: Record<string, number> }
    >;
    range: Record<
      string,
      { reference_min: number | null; reference_max: number | null; current_min: number | null; current_max: number | null; out_of_range_rate: number }
    >;
  };
  thresholds: { psi: number; categorical: number; missingness: number; range: number };
  alerts: DriftAlert[];
}

export const checkDrift = (task: string, body: DriftRequest) =>
  post<DriftReport>(`/api/monitoring/${encodeURIComponent(task)}/drift`, body);

export interface FairnessRequest {
  y_true: Array<number | string>;
  y_pred: Array<number | string>;
  groups: Array<number | string>;
  positive_label?: number;
}

export interface GroupStats {
  count: number;
  selection_rate: number;
  true_positive_rate: number;
  accuracy: number;
}

export interface FairnessReport {
  groups: Record<string, GroupStats>;
  fairness: { demographic_parity_gap: number; equal_opportunity_gap: number };
}

export const checkFairness = (task: string, body: FairnessRequest) =>
  post<FairnessReport>(`/api/monitoring/${encodeURIComponent(task)}/fairness`, body);

export interface DelayedLabelRequest {
  predictions: Array<number | string>;
  labels: Array<number | string>;
  model_version?: string | null;
  task_type?: "classification" | "regression";
}

export const checkDelayedLabels = (task: string, body: DelayedLabelRequest) =>
  post<{ model_version: string | null; rows: number; metrics: Record<string, number> }>(
    `/api/monitoring/${encodeURIComponent(task)}/delayed-label-performance`,
    body,
  );

export interface MonitorSummary {
  model_version: string;
  requests: number;
  rows: number;
  error_rate: number;
  mean_latency_ms: number;
  throughput_rows_per_second: number;
}

export const monitorSummary = (task: string, modelVersion: string) =>
  get<MonitorSummary>(`/api/monitoring/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/summary`);

/** Where a prediction series comes from — exactly one field must be set. */
export interface PredictionSource {
  /** Stored dataset partition; its registered target column is used as the series by default. */
  partition?: string | null;
  /** Persisted inference artifact id (from predict-batch). */
  request_id?: string | null;
  /** Inline values for ad-hoc checks (numbers, or labels for groups). */
  values?: Array<number | string> | null;
  /** Column override: any stored column of the partition/artifact (e.g. a demographic attribute for groups). */
  column?: string | null;
}

export interface PredictionDriftRequest {
  reference_source: PredictionSource;
  current_source: PredictionSource;
  threshold?: number;
  bins?: number;
}

/** Same shape as DriftReport (single "prediction" column) plus the marker field. */
export type PredictionDriftReport = DriftReport & { prediction_column: string };

export const checkPredictionDrift = (task: string, modelVersion: string, body: PredictionDriftRequest) =>
  post<PredictionDriftReport>(
    `/api/monitoring/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/prediction-drift`,
    body,
  );

export type SubgroupMetric = "accuracy" | "precision" | "recall" | "f1" | "loss";

export interface SubgroupPerformanceRequest {
  y_true_source: PredictionSource;
  y_pred_source: PredictionSource;
  groups_source?: PredictionSource | null;
  inline_groups?: Array<string | number> | null;
  metric?: SubgroupMetric;
  positive_label?: number;
}

export interface SubgroupPerformanceReport {
  groups: Record<string, { [metric in SubgroupMetric]?: number } & { count: number }>;
  metric: SubgroupMetric;
  fairness_gap: number;
}

export const checkSubgroupPerformance = (task: string, modelVersion: string, body: SubgroupPerformanceRequest) =>
  post<SubgroupPerformanceReport>(
    `/api/monitoring/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/subgroup-performance`,
    body,
  );

export interface SensitiveScanRequest {
  partition?: string | null;
  columns?: string[] | null;
  sensitive_features?: string[] | null;
}

export interface SensitiveScanReport {
  pii_columns: string[];
  sensitive_columns: string[];
  warnings: string[];
}

export const scanSensitiveFeatures = (task: string, body: SensitiveScanRequest) =>
  post<SensitiveScanReport>(`/api/monitoring/${encodeURIComponent(task)}/sensitive-scan`, body);

/* ------------------------------------------------------------------ */
/* Sensitivity                                                         */
/* ------------------------------------------------------------------ */

export interface SensitivityRequest {
  method?: "sobol" | "morris" | "delta-mmnt";
  n_features_to_select?: number;
  train_partition?: string;
}

export const runSensitivity = (task: string, body: SensitivityRequest) =>
  post<{ job_id: string; status: string }>(`/api/sensitivity/${encodeURIComponent(task)}/run`, body);

/** Result payload of a completed sensitivity job (see api/routers/sensitivity.py). */
export interface SensitivityJobResult {
  method: string;
  n_features_to_select: number;
  feature_columns: string[];
  top_feature_indices: number[];
  top_feature_names: string[];
  /** One weight per entry of `feature_columns`; null where the value was not finite. */
  weights: Array<number | null>;
}

/** Narrow an unknown job result to a sensitivity result, or return null. */
export function asSensitivityResult(result: unknown): SensitivityJobResult | null {
  if (!result || typeof result !== "object") return null;
  const r = result as Partial<SensitivityJobResult>;
  if (!Array.isArray(r.feature_columns) || !Array.isArray(r.weights) || typeof r.method !== "string") return null;
  return {
    method: r.method,
    n_features_to_select: r.n_features_to_select ?? 0,
    feature_columns: r.feature_columns,
    top_feature_indices: r.top_feature_indices ?? [],
    top_feature_names: r.top_feature_names ?? [],
    weights: r.weights,
  };
}

export interface CorrelationThresholdResult {
  threshold: number;
  feature_columns: string[];
  kept_feature_names: string[];
  dropped_feature_names: string[];
}

export const runCorrelationThreshold = (task: string, threshold = 0.7, trainPartition = "train") =>
  post<CorrelationThresholdResult>(
    `/api/sensitivity/${encodeURIComponent(task)}/correlation-threshold?threshold=${threshold}&train_partition=${encodeURIComponent(trainPartition)}`,
  );

/** Square feature-by-feature matrix as consumed by the heatmap renderer (mirrors mltrack.heatmap). */
export interface HeatmapData {
  labels: string[];
  columns: string[];
  values: Array<Array<number | null>>;
}

export interface CorrelationMatrixResult extends HeatmapData {
  task_name: string;
  partition: string;
  kind: "correlation";
}

export const getCorrelationMatrix = (task: string, trainPartition = "train", maxFeatures = 60) =>
  get<CorrelationMatrixResult>(
    `/api/sensitivity/${encodeURIComponent(task)}/correlation-matrix?train_partition=${encodeURIComponent(trainPartition)}&max_features=${maxFeatures}`,
  );

export interface WeightsHeatmapResult extends HeatmapData {
  task_name: string;
  kind: "weights";
  available: boolean;
  /** "stored" = read from the mltrack weights table; "computed" = pearson/variance on demand. */
  source: "stored" | "computed";
}

export const getWeightsHeatmap = (task: string, trainPartition = "train", maxFeatures = 60) =>
  get<WeightsHeatmapResult>(
    `/api/sensitivity/${encodeURIComponent(task)}/weights-heatmap?train_partition=${encodeURIComponent(trainPartition)}&max_features=${maxFeatures}`,
  );

/** Consensus top features across the stored weightings (mltrack.TopFeatures). */
export interface TopFeaturesResponse {
  task_name: string;
  /** False when no weights table exists yet — render a disabled card with `hint`. */
  available: boolean;
  hint?: string;
  num?: number;
  /** Number of weighting columns the consensus was computed over. */
  weightings?: number;
  /** [feature, appearances] pairs, most frequent first. Empty when unavailable. */
  features: Array<[string, number]>;
}

export const getTopFeatures = (task: string, num = 10) =>
  get<TopFeaturesResponse>(
    `/api/sensitivity/${encodeURIComponent(task)}/top-features?num=${Math.max(1, Math.floor(num))}`,
  );

/* ------------------------------------------------------------------ */
/* Evaluation — learning curves                                        */
/* ------------------------------------------------------------------ */

/** One curated metric offered as a learning-curve tab (see api/training.py). */
export interface LearningCurveMetric {
  /** Scikit-learn scorer name passed to the API as `scoring`. */
  value: string;
  label: string;
  /** False for error metrics such as RMSE — the curve then decreases with more data. */
  higher_is_better: boolean;
}

export interface LearningCurveMetricsResponse {
  task_name: string;
  /** "Classification" | "Regression" | null when no bundle could be inspected. */
  family: string | null;
  metrics: LearningCurveMetric[];
  default: string | null;
}

export interface LearningCurveSeries {
  model_version: string;
  /** Absolute number of training samples at each point of the curve. */
  train_sizes: number[];
  /** The same sizes as a fraction of the dataset (0–1). */
  train_sizes_fraction: number[];
  /** Null where a fold could not be scored (rendered as a gap). */
  train_scores_mean: Array<number | null>;
  train_scores_std: Array<number | null>;
  validation_scores_mean: Array<number | null>;
  validation_scores_std: Array<number | null>;
}

export interface LearningCurvesResponse {
  task_name: string;
  scoring: string;
  scoring_label: string;
  train_partition: string;
  n_points: number;
  series: LearningCurveSeries[];
  /** Bundles that could not be scored for this metric. */
  errors: Array<{ model_version: string; detail: string }>;
}

/** Relevant learning-curve metrics for the task, inferred from its bundles' problem family. */
export const getLearningCurveMetrics = (task: string, modelVersion?: string) =>
  get<LearningCurveMetricsResponse>(
    `/api/evaluation/${encodeURIComponent(task)}/metrics${modelVersion ? `?model_version=${encodeURIComponent(modelVersion)}` : ""}`,
  );

/** Cross-validated learning curves for every (or the given) bundle under one metric. */
export function getLearningCurves(
  task: string,
  opts: { scoring: string; trainPartition?: string; nPoints?: number; modelVersions?: string[] },
): Promise<LearningCurvesResponse> {
  const params = new URLSearchParams({ scoring: opts.scoring });
  if (opts.trainPartition) params.set("train_partition", opts.trainPartition);
  if (opts.nPoints) params.set("n_points", String(opts.nPoints));
  if (opts.modelVersions?.length) params.set("model_versions", opts.modelVersions.join(","));
  return get<LearningCurvesResponse>(`/api/evaluation/${encodeURIComponent(task)}/learning-curves?${params}`);
}

/* ------------------------------------------------------------------ */
/* Evaluation — nested CV (background job)                             */
/* ------------------------------------------------------------------ */

export interface NestedCVRequest {
  /** Exactly one of model_version / alias must be set. */
  model_version?: string | null;
  alias?: string | null;
  inner_cv?: number;
  outer_cv?: number;
  train_partition?: string;
  validation_partition?: string;
  scoring?: string | null;
  repeats?: number;
  confidence_level?: number;
}

/** Submit a nested-CV evaluation job; poll the returned id for the result. */
export const runNestedCV = (task: string, body: NestedCVRequest) =>
  post<{ job_id: string; status: string }>(`/api/evaluation/${encodeURIComponent(task)}/nested-cv`, body);

/** Result payload of a completed nested-CV job (see api/routers/evaluation.py). */
export interface NestedCVJobResult {
  task_name: string;
  model_version: string;
  outer_scores: number[];
  inner_scores: number[];
  mean_outer_score: number | null;
  fold_metrics: Array<{
    outer_fold: number;
    inner_scores: number[];
    outer_predictions: unknown[];
    outer_truth: unknown[];
    outer_score: number | null;
    inner_score_mean: number | null;
  }>;
  repeat_count: number;
  repeated_outer_scores: number[];
  confidence_interval: { confidence_level: number; lower: number | null; upper: number | null };
  final_validation_score?: number | null;
}

/** Narrow an unknown job result to a nested-CV result, or return null. */
export function asNestedCVResult(result: unknown): NestedCVJobResult | null {
  if (!result || typeof result !== "object") return null;
  const r = result as Partial<NestedCVJobResult>;
  if (!Array.isArray(r.outer_scores) || !r.confidence_interval) return null;
  return r as NestedCVJobResult;
}

/* ------------------------------------------------------------------ */
/* Evaluation — diagnostic curves                                      */
/* ------------------------------------------------------------------ */

export interface BundleCurvesResponse {
  task_name: string;
  model_version: string;
  bins: number;
  n_test_samples: number;
  classes: string[];
  roc?: { fpr: Array<number | null>; tpr: Array<number | null>; auc: number | null };
  calibration?: {
    mean_predicted_value: Array<number | null>;
    fraction_of_positives: Array<number | null>;
    histogram_counts: number[];
    histogram_edges: Array<number | null>;
  };
  cumulative_gain?: {
    percentages: Array<number | null>;
    gains_class0: Array<number | null>;
    gains_class1: Array<number | null>;
    class0: string;
    class1: string;
  };
  lift?: {
    percentages: Array<number | null>;
    lifts_class0: Array<number | null>;
    lifts_class1: Array<number | null>;
    class0: string;
    class1: string;
  };
  /** Per-curve failures (e.g. gain/lift on a multi-class target) — not fatal. */
  errors: Array<{ curve: string; detail: string }>;
}

/** Diagnostic curves for one classification bundle (422 for regression bundles). */
export const getBundleCurves = (task: string, modelVersion: string, bins = 10) =>
  get<BundleCurvesResponse>(
    `/api/evaluation/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/curves?bins=${Math.max(2, Math.min(50, bins))}`,
  );

/* ------------------------------------------------------------------ */
/* Experiments                                                         */
/* ------------------------------------------------------------------ */

export interface ParamSpec {
  type: "real" | "integer" | "categorical";
  low?: number;
  high?: number;
  items?: unknown[];
}

export interface RunExperimentRequest {
  config: Record<string, Record<string, ParamSpec>>;
  length?: number;
  max_generation?: number;
  num_parents?: number;
  train_partition?: string;
  scoring?: string;
  random_state?: number | null;
  owner?: string | null;
  run_id?: string | null;
}

export const runExperiment = (task: string, body: RunExperimentRequest) =>
  post<{ job_id: string; status: string }>(`/api/experiments/${encodeURIComponent(task)}/run`, body);

/** A small, safe default search space (subset of the library's default_config). */
export const DEFAULT_EXPERIMENT_CONFIG: Record<string, Record<string, ParamSpec>> = {
  "sklearn.linear_model.LogisticRegression": {
    C: { type: "real", low: 1e-6, high: 10 },
    penalty: { type: "categorical", items: ["l2"] },
  },
  "sklearn.tree.DecisionTreeClassifier": {
    criterion: { type: "categorical", items: ["gini", "entropy"] },
    min_samples_split: { type: "integer", low: 2, high: 10 },
    min_samples_leaf: { type: "integer", low: 1, high: 10 },
  },
  "sklearn.ensemble.RandomForestClassifier": {
    n_estimators: { type: "integer", low: 10, high: 200 },
    criterion: { type: "categorical", items: ["gini", "entropy"] },
    min_samples_split: { type: "integer", low: 2, high: 10 },
  },
};

/* ------------------------------------------------------------------ */
/* Scoring (AML/EOA objective)                                         */
/* ------------------------------------------------------------------ */

export interface ScoringOption {
  /** Scikit-learn scorer name passed straight to AML as `scoring`. */
  value: string;
  /** Human-readable label (e.g. "Mean squared error (negated)"). */
  label: string;
}

export interface ScoringGroup {
  label: string;
  options: ScoringOption[];
}

export interface ScoringOptionsResponse {
  groups: ScoringGroup[];
  default: string;
}

/**
 * The standard scikit-learn scorers that can drive the AML/EOA search. The
 * selected value becomes the objective the search maximizes; `neg_*` metrics
 * are already sign-flipped by scikit-learn, so "higher score" always means a
 * better model regardless of whether the raw metric is an error or a score.
 */
export const getScoringOptions = () => get<ScoringOptionsResponse>("/api/experiments/scoring-options");

/* ------------------------------------------------------------------ */
/* Jobs                                                                */
/* ------------------------------------------------------------------ */

export const listJobs = (task?: string) =>
  get<{ jobs: JobRecord[] }>(`/api/jobs${task ? `?task_name=${encodeURIComponent(task)}` : ""}`);

export const getJob = (jobId: string) => get<JobRecord>(`/api/jobs/${encodeURIComponent(jobId)}`);

/** Subscribe to a job's WebSocket progress stream. Returns an unsubscribe fn. */
export function subscribeToJob(
  jobId: string,
  onStatus: (record: JobRecord) => void,
): () => void {
  // WebSocket URLs need an explicit scheme, so resolve the (possibly empty)
  // same-origin base to an absolute one: http->ws, https->wss.
  const url = getEffectiveApiBase().replace(/^http/, "ws");
  // Browsers cannot set custom headers on a WebSocket handshake, so the key is
  // passed as a query parameter (the server accepts it there when auth is on).
  const key = getApiKey();
  const ws = new WebSocket(
    `${url}/api/jobs/${encodeURIComponent(jobId)}/stream${key ? `?api_key=${encodeURIComponent(key)}` : ""}`,
  );
  ws.onmessage = (event) => {
    try {
      onStatus(JSON.parse(event.data as string) as JobRecord);
    } catch {
      /* ignore malformed frames */
    }
  };
  return () => ws.close();
}

/* ------------------------------------------------------------------ */
/* Retraining                                                          */
/* ------------------------------------------------------------------ */

export interface BaselineTrainerConfig {
  kind: "baseline";
  estimator?: string;
  train_partition?: string;
  validation_partition?: string | null;
}

export interface ExperimentTrainerConfig {
  kind: "experiment";
  config: Record<string, Record<string, ParamSpec>>;
  length?: number;
  max_generation?: number;
  num_parents?: number;
  train_partition?: string;
  scoring?: string;
  random_state?: number | null;
}

export type TrainerConfig = BaselineTrainerConfig | ExperimentTrainerConfig;

export interface RunRetrainingRequest {
  trainer: TrainerConfig;
  trigger?: "always" | "scheduled" | "on_data";
  due?: boolean;
  promotion_state?: string | null;
  context?: Record<string, unknown>;
  owner?: string | null;
  run_id?: string | null;
}

export const runRetraining = (task: string, body: RunRetrainingRequest) =>
  post<{ job_id: string; status: string }>(`/api/retraining/${encodeURIComponent(task)}/run`, body);

/* ------------------------------------------------------------------ */
/* Lineage                                                             */
/* ------------------------------------------------------------------ */

export interface LineageDataset {
  dataset_fingerprint: string | null;
  target_name: string | null;
  partitions: Record<string, PartitionInfo>;
}

export interface LineageRegistry {
  state: string | null;
  history: AuditEvent[];
}

export interface LineageMonitoring {
  requests: number;
  rows: number;
  error_rate: number;
}

export interface LineageRecord {
  task_name: string;
  model_version: string;
  bundle: BundleSummary;
  dataset: LineageDataset | null;
  registry: LineageRegistry | null;
  monitoring: LineageMonitoring | null;
}

export const getLineage = (task: string, modelVersion: string) =>
  get<LineageRecord>(`/api/lineage/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}`);
