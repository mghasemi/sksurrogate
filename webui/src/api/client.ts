/**
 * Typed client for the SKSurrogate control-plane API.
 *
 * The base URL defaults to http://localhost:8013 (the uvicorn port used in
 * docs/HANDOFF.md) and can be overridden with VITE_API_BASE_URL or a runtime
 * override stored in localStorage (see setApiBase).
 */

const DEFAULT_BASE = "http://localhost:8013";

function base(): string {
  const env = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return (localStorage.getItem("sksurrogate_api_base") || env || DEFAULT_BASE).replace(/\/+$/, "");
}

export function getApiBase(): string {
  return base();
}

export function setApiBase(url: string): void {
  localStorage.setItem("sksurrogate_api_base", url.replace(/\/+$/, ""));
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
 */
export const retryUnlessNotFound = (err: unknown): boolean =>
  !(err instanceof ApiError && err.status === 404);

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${base()}${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
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
  const url = base().replace(/^http/, "ws");
  const ws = new WebSocket(`${url}/api/jobs/${encodeURIComponent(jobId)}/stream`);
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
