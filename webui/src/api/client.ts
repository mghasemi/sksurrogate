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
    // The browser session is an HttpOnly cookie; same-origin calls attach it by
    // default, and "include" also covers an explicitly configured API base.
    credentials: "include",
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
  backend?: "local" | "dask";
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

export const health = () => get<{ status: string; version?: string }>("/api/health");

/* ------------------------------------------------------------------ */
/* Tasks                                                               */
/* ------------------------------------------------------------------ */

/** Known task names discovered across the API storage layout. */
export const listTasks = () => get<{ tasks: string[] }>("/api/tasks");

/**
 * Probe whether the server enforces the shared API key. `/api/health` is never
 * gated, so a cheap protected endpoint (`GET /api/tasks`) stands in for it:
 * 401 means a key is required, any other success means open. Connection errors
 * are rethrown — we do not guess the auth state of an unreachable server (the
 * Settings page's health card already surfaces that failure).
 */
export async function probeAuthMode(): Promise<"open" | "required"> {
  try {
    await listTasks();
    return "open";
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return "required";
    throw err;
  }
}

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
  dataset_deduced_types?: Record<string, string>;
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
  dataset_deduced_types?: Record<string, string> | null;
  dataset_feature_count: number | null;
  partitions: Record<string, PartitionInfo>;
}

export interface DatasetPreview {
  task_name: string;
  partition: string;
  rows: Array<Record<string, unknown>>;
}

/** Phase 5.1: schema preview of an uploaded CSV, without persisting anything. */
export interface DatasetInspectResponse {
  task_name: string;
  rows: number;
  columns: string[];
  deduced_types: Record<string, string>;
  target_candidates: string[];
  sensitive_scan?: SensitiveScanReport;
}

/** Phase 5.2: pre-flight validation result for an upload or stored partition. */
export interface DatasetValidationResponse {
  task_name: string;
  source: string;
  valid: boolean;
}

/** Phase 5.3: one generated synthetic sample file. */
export interface SyntheticFileEntry {
  name: string;
  path: string;
  rows: number;
}

export interface GenerateSyntheticResponse {
  task_name: string;
  path: string;
  partition: string;
  rows: number;
  columns: string[];
  preview: Array<Record<string, unknown>>;
}

export interface SyntheticListResponse {
  task_name: string;
  files: SyntheticFileEntry[];
  preview: { file: string; columns: string[]; rows: Array<Record<string, unknown>> } | null;
}

async function uploadDataset(
  path: string,
  form: FormData,
): Promise<unknown> {
  const res = await fetch(`${base()}${path}`, { method: "POST", headers: authHeaders(), body: form });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: string };
      if (data.detail) detail = data.detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, `POST ${path} -> ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function inspectDataset(
  taskName: string,
  file: File,
): Promise<DatasetInspectResponse> {
  const form = new FormData();
  form.append("file", file);
  return (await uploadDataset(
    `/api/datasets/${encodeURIComponent(taskName)}/inspect`,
    form,
  )) as DatasetInspectResponse;
}

export async function validateDataset(
  taskName: string,
  body: { file?: File; partition?: string },
): Promise<DatasetValidationResponse> {
  const form = new FormData();
  if (body.file) form.append("file", body.file);
  if (body.partition) form.append("partition", body.partition);
  return (await uploadDataset(
    `/api/datasets/${encodeURIComponent(taskName)}/validate`,
    form,
  )) as DatasetValidationResponse;
}

export async function registerDataset(
  taskName: string,
  target: string,
  partition: string,
  file: File,
  options?: { type_overrides?: Record<string, string>; binarize_label?: boolean },
): Promise<RegisterDatasetResponse> {
  const form = new FormData();
  form.append("target", target);
  form.append("partition", partition);
  form.append("file", file);
  if (options?.type_overrides && Object.keys(options.type_overrides).length) {
    form.append("type_overrides", JSON.stringify(options.type_overrides));
  }
  if (options?.binarize_label) form.append("binarize_label", "true");
  return (await uploadDataset(
    `/api/datasets/${encodeURIComponent(taskName)}/register`,
    form,
  )) as RegisterDatasetResponse;
}

export const getDatasetMetadata = (task: string) =>
  get<DatasetMetadata>(`/api/datasets/${encodeURIComponent(task)}`);

export const previewDataset = (task: string, partition: string, limit = 20) =>
  get<DatasetPreview>(
    `/api/datasets/${encodeURIComponent(task)}/${encodeURIComponent(partition)}/preview?limit=${limit}`,
  );

/** Phase 5.3: generate a synthetic sample from a registered partition. */
export const generateSynthetic = (task: string, body: {
  num?: number;
  partition?: string | null;
  distribution_type?: "marginal" | "joint";
  default_rv?: string;
  type_overrides?: Record<string, string> | null;
}) => post<GenerateSyntheticResponse>(
  `/api/datasets/${encodeURIComponent(task)}/synthetic`,
  { num: 100, distribution_type: "marginal", default_rv: "uniform", ...body },
);

export const listSynthetic = (task: string, limit = 50) =>
  get<SyntheticListResponse>(
    `/api/datasets/${encodeURIComponent(task)}/synthetic?limit=${limit}`,
  );

export async function downloadSynthetic(task: string, filename: string): Promise<void> {
  const path = `/api/datasets/${encodeURIComponent(task)}/synthetic/${encodeURIComponent(filename)}`;
  const res = await fetch(`${base()}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, `GET ${path} -> ${res.status}: ${detail}`);
  }
  const objectUrl = URL.createObjectURL(await res.blob());
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

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

/* --- Phase 3.1 model preservation --------------------------------- */

export interface PreservedSnapshot {
  pickle_id: number;
  model_id: number;
  init_date: string;
}

export const preserveBundle = (task: string, modelVersion: string) =>
  post<PreservedSnapshot & { task_name: string; model_version: string }>(
    `/api/bundles/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/preserve`,
  );

export const listPreservedModels = (task: string) =>
  get<{ task_name: string; snapshots: PreservedSnapshot[] }>(
    `/api/bundles/${encodeURIComponent(task)}/preserved`,
  );

export interface RecoverModelResponse {
  task_name: string;
  pickle_id: number;
  model_type: string;
  smoke_score: number | null;
  model_version: string;
}

export const recoverModel = (task: string, pickleId: number) =>
  post<RecoverModelResponse>(`/api/bundles/${encodeURIComponent(task)}/recover`, {
    pickle_id: pickleId,
  });

/* --- Phase 3.2 best model ----------------------------------------- */

export interface BestBundle {
  task_name: string;
  /** The metric that won the ranking. */
  metric: string;
  /** The candidate metrics that were tried, in order. */
  metrics_tried: string[];
  model_version: string;
  value: number | null;
}

/**
 * Metric names to rank by, in order of preference. A task's bundles come from
 * different producers — baselines record `score`, AML experiments record
 * `train_score` — so the endpoint picks the first metric that ranks anything.
 */
export const BEST_METRIC_PREFERENCE = ["score", "train_score"] as const;

/** The bundle with the highest value of `metric`, or of the first of several that ranks anything. */
export const getBestBundle = (task: string, metric: string = BEST_METRIC_PREFERENCE.join(",")) =>
  get<BestBundle>(
    `/api/bundles/${encodeURIComponent(task)}/best?metric=${encodeURIComponent(metric)}`,
  );

/** Best bundle across {@link BEST_METRIC_PREFERENCE}, or `null` when nothing is rankable yet. */
export async function findBestBundle(
  task: string,
  metrics: readonly string[] = BEST_METRIC_PREFERENCE,
): Promise<BestBundle | null> {
  try {
    return await getBestBundle(task, metrics.join(","));
  } catch (err) {
    // 404 = "no bundle carries any of these metrics yet" — a normal empty state.
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

/* --- Phase 3.3 export & download ---------------------------------- */

/**
 * Absolute URL that streams the bundle as an MLflow-layout zip.
 *
 * Anchor downloads cannot carry the `X-API-Key` header, so the key is appended
 * as a query parameter — the same escape hatch the WebSocket job stream uses.
 */
export function exportMlflowUrl(task: string, modelVersion: string): string {
  const key = getApiKey();
  const base = getEffectiveApiBase();
  return `${base}/api/bundles/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/export-mlflow${key ? `?api_key=${encodeURIComponent(key)}` : ""}`;
}

/** Absolute URL that streams the raw `.bundle` file (API key as query param, see above). */
export function bundleDownloadUrl(task: string, modelVersion: string): string {
  const key = getApiKey();
  const base = getEffectiveApiBase();
  return `${base}/api/bundles/${encodeURIComponent(task)}/${encodeURIComponent(modelVersion)}/download${key ? `?api_key=${encodeURIComponent(key)}` : ""}`;
}

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

/* --- Phase 3.4 artifact lineage & cleanup -------------------------- */

export interface RegistryArtifact {
  kind: "dataset" | "prediction";
  path: string;
  exists: boolean;
  size_bytes: number;
  registered_at?: string;
  metadata: Record<string, unknown>;
  /** Datasets only. */
  dataset_fingerprint?: string;
  source_path?: string;
  /** Predictions only. */
  model_version?: string;
  prediction_id?: string;
}

export interface RegistryArtifacts {
  task_name: string;
  datasets: RegistryArtifact[];
  predictions: RegistryArtifact[];
  dataset_count: number;
  prediction_count: number;
  total_bytes: number;
}

export const getRegistryArtifacts = (task: string) =>
  get<RegistryArtifacts>(`/api/registry/${encodeURIComponent(task)}/artifacts`);

/** Selectors for a cleanup; at least one must be set (the server returns 422 otherwise). */
export interface DeleteArtifactsRequest {
  model_version?: string | null;
  dataset_fingerprint?: string | null;
}

export interface DeleteArtifactsResponse {
  task_name: string;
  model_version: string | null;
  dataset_fingerprint: string | null;
  deleted: { datasets: number; predictions: number };
}

export const deleteArtifacts = (task: string, body: DeleteArtifactsRequest) => {
  const params = new URLSearchParams();
  if (body.model_version) params.set("model_version", body.model_version);
  if (body.dataset_fingerprint) params.set("dataset_fingerprint", body.dataset_fingerprint);
  const query = params.toString();
  return request<DeleteArtifactsResponse>(
    "DELETE",
    `/api/registry/${encodeURIComponent(task)}/artifacts${query ? `?${query}` : ""}`,
  );
};

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */

export interface PredictRequest {
  model_version?: string | null;
  alias?: string | null;
  partition?: string | null;
  rows?: Array<Record<string, unknown>> | null;
  request_id?: string | null;
  /** Phase 5.2: run validate_prediction_data before predicting (400 on mismatch). */
  preflight?: boolean;
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

export interface MonitoringAlert {
  timestamp: string;
  task: string;
  version: string | null;
  type: string;
  severity: "warning" | "error";
  detail: string;
}

export const getMonitoringAlerts = (task?: string, limit = 20) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (task) params.set("task", task);
  return get<{ alerts: MonitoringAlert[] }>(`/api/monitoring/alerts?${params.toString()}`);
};

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
  /** Which SALib estimator produced the weights (delta-mmnt only). */
  estimator?: string | null;
  /** Column this run was persisted under in the task's ``weights`` table. */
  weights_column?: string;
  /** False when the run's weights could not be stored (analysis still valid). */
  weights_stored?: boolean;
  storage_error?: string | null;
  evaluation_protocol?: {
    train_partition: string;
    train_fingerprint: string | null;
    row_count: number;
    feature_columns: string[];
    /** Non-numeric columns excluded from the analysis. */
    excluded_columns?: Array<{ column: string; dtype: string }>;
    rows_dropped?: number;
  };
  interpretation?: {
    category: "input_sensitivity";
    measure: string;
    train_partition: string;
    causal_effect: false;
    limitations: string[];
  };
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
    interpretation: r.interpretation,
    estimator: r.estimator ?? null,
    weights_column: r.weights_column,
    weights_stored: r.weights_stored,
    storage_error: r.storage_error ?? null,
    evaluation_protocol: r.evaluation_protocol,
  };
}

export interface CorrelationThresholdResult {
  threshold: number;
  feature_columns: string[];
  kept_feature_names: string[];
  dropped_feature_names: string[];
  /** Non-numeric columns: Pearson correlation is undefined for them. */
  excluded_feature_names?: string[];
  rows_dropped?: number;
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
  interpretation: {
    category: "descriptive_association";
    measure: "Pearson correlation";
    causal_effect: false;
  };
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
  /** Names of those weighting columns (e.g. ``sobol``, ``pearson``). */
  weight_columns?: string[];
  /** [feature, appearances] pairs, most frequent first. Empty when unavailable. */
  features: Array<[string, number]>;
}

/** Recompute and store the dataset-level weightings (Pearson, variance). */
export const computeBaseWeights = (task: string, trainPartition = "train") =>
  post<{
    task_name: string;
    partition: string;
    weights_columns: string[];
    feature_count: number;
    row_count: number;
    excluded_columns: Array<{ column: string; dtype: string }>;
    rows_dropped: number;
  }>(
    `/api/sensitivity/${encodeURIComponent(task)}/base-weights?train_partition=${encodeURIComponent(trainPartition)}`,
  );

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
  /** The metric requires an explicit quantile level. */
  requires_alpha?: boolean;
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
  pinball_alpha: number | null;
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
  opts: {
    scoring: string;
    pinballAlpha?: number;
    trainPartition?: string;
    nPoints?: number;
    modelVersions?: string[];
  },
): Promise<LearningCurvesResponse> {
  const params = new URLSearchParams({ scoring: opts.scoring });
  if (opts.pinballAlpha !== undefined) params.set("pinball_alpha", String(opts.pinballAlpha));
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
  /** Conditional rule: {other_param: [allowed values]} — the param is dropped when the rule does not match. */
  depends_on?: Record<string, unknown[]>;
}

/** A search-space entry: a range spec, or a scalar under the reserved "stacking" directive. */
export type ParamValue = ParamSpec | boolean | number | null;

export type SearchSpace = Record<string, Record<string, ParamValue>>;

/**
 * Reserved search-space key configuring the automatic ``StackingEstimator``
 * wrappers (out-of-fold stacking of intermediate estimators). It is a
 * directive, not a pipeline component, so it never becomes a step.
 */
export const STACKING_COMPONENT = {
  stacking: {
    res: true,
    probs: true,
    decision: true,
    cv: 5,
    n_jobs: null as number | null,
  },
};

export interface RunExperimentRequest {
  /** Omit when `preset` is given. */
  config?: SearchSpace;
  /** Quick-start preset (bounded search space); mutually exclusive with `config`. */
  preset?: "classification" | "regression";
  /** `eoa` (evolutionary, default) or `surrogate` (surrogate search only). */
  strategy?: "eoa" | "surrogate";
  length?: number;
  max_generations?: number;
  num_parents?: number;
  train_partition?: string;
  scoring?: string;
  random_state?: number | null;
  owner?: string | null;
  run_id?: string | null;
  /** Use a single lightweight surrogate regressor instead of the default KRR+GPR pair. */
  surrogate_mode?: boolean;
  /** Per-surrogate iteration budget (surrogate mode only). */
  surrogate_itrs?: number | null;
  /** Disallowed parameter combinations, e.g. [["penalty", "l1"]] or [["max_depth", 1]]. */
  forbidden?: Array<[string, unknown]> | null;
  backend?: "local" | "dask";
}

export interface ExperimentTopPipeline {
  pipeline: string[];
  /** In the scorer's own units; null for candidates that could not be evaluated. */
  score: number | null;
}

/** One non-dominated candidate of the search (score maximized, duration minimized). */
export interface ParetoPoint {
  pipeline: string[];
  score: number | null;
  duration: number | null;
}

export interface ExperimentResult {
  model_version: string;
  bundle_path: string;
  train_score: number | null;
  evaluation_history: Array<{ pipeline: string[]; score: number | null; duration?: number }>;
  top_pipelines: ExperimentTopPipeline[];
  /** Null when the search produced no candidates with both a score and a duration. */
  pareto: ParetoPoint[] | null;
}

export const runExperiment = (task: string, body: RunExperimentRequest) =>
  post<{ job_id: string; status: string }>(`/api/experiments/${encodeURIComponent(task)}/run`, body);

/** Optimize one explicit pipeline structure (no search over structures). */
export interface OptimizePipelineRequest {
  seq: string[];
  config?: SearchSpace | null;
  train_partition?: string;
  scoring?: string;
  random_state?: number | null;
  owner?: string | null;
  run_id?: string | null;
}

export interface OptimizePipelineResult {
  model_version: string;
  bundle_path: string;
  train_score: number | null;
  pipeline: string[];
}

export const optimizePipeline = (task: string, body: OptimizePipelineRequest) =>
  post<{ job_id: string; status: string }>(`/api/experiments/${encodeURIComponent(task)}/optimize-pipeline`, body);

/** A small, safe default search space (subset of the library's default_config). */
export const DEFAULT_EXPERIMENT_CONFIG: SearchSpace = {
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
  backends: Array<"local" | "dask">;
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

export const resumeJob = (jobId: string) =>
  post<{ job_id: string; status: string }>(`/api/jobs/${encodeURIComponent(jobId)}/resume`);

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
/* Logs                                                                */
/* ------------------------------------------------------------------ */

/** Severity vocabulary, matching the standard library's ordering. */
export const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] as const;
export type LogLevel = (typeof LOG_LEVELS)[number];

export interface LogEntry {
  seq: number;
  ts: string;
  level: LogLevel;
  logger: string;
  message: string;
  job_id: string;
  task_name?: string | null;
  kind?: string | null;
  /** Zero-based trial index, present on per-trial telemetry entries. */
  trial?: number | null;
  /** Sampling phase of the trial: init | explore | surrogate_exploit. */
  phase?: string | null;
  status?: string | null;
  /** Full traceback, present on error entries raised from a job. */
  detail?: string;
}

export interface LogCounts {
  DEBUG: number;
  INFO: number;
  WARNING: number;
  ERROR: number;
  CRITICAL: number;
}

export interface LogQueryResponse {
  entries: LogEntry[];
  counts: LogCounts;
  total: number;
  levels?: LogLevel[];
  job_id?: string;
  /** Present when the query was scoped to a research project. */
  project_id?: string;
  /** Number of project-linked jobs the query scanned. */
  job_count?: number;
}

export interface LogQuery {
  job_id?: string;
  /** Minimum severity; `WARNING` means WARNING and above. */
  level?: LogLevel;
  /** Explicit subset of severities (wins over `level`). */
  levels?: LogLevel[];
  kind?: string;
  /** Restrict to one dataset/task. */
  task_name?: string;
  /** Restrict to the jobs linked as artifacts of this research project. */
  project_id?: string;
  q?: string;
  limit?: number;
  since_seq?: number;
}

function logQueryString(query: LogQuery): string {
  const params = new URLSearchParams();
  if (query.job_id) params.set("job_id", query.job_id);
  if (query.level) params.set("level", query.level);
  if (query.levels && query.levels.length) params.set("levels", query.levels.join(","));
  if (query.kind) params.set("kind", query.kind);
  if (query.task_name) params.set("task_name", query.task_name);
  if (query.project_id) params.set("project_id", query.project_id);
  if (query.q) params.set("q", query.q);
  if (query.limit) params.set("limit", String(query.limit));
  if (query.since_seq !== undefined) params.set("since_seq", String(query.since_seq));
  const text = params.toString();
  return text ? `?${text}` : "";
}

/** Browse captured log entries across jobs. */
export function listLogs(query: LogQuery = {}): Promise<LogQueryResponse> {
  return get<LogQueryResponse>(`/api/logs${logQueryString(query)}`);
}

/** Tail one job's captured entries. */
export function jobLogs(jobId: string, query: LogQuery = {}): Promise<LogQueryResponse> {
  return get<LogQueryResponse>(`/api/logs/${encodeURIComponent(jobId)}${logQueryString(query)}`);
}

/**
 * Subscribe to a job's log stream. Frames look like
 * `{entries: LogEntry[], since_seq: number, status: "running"}`; `done` fires
 * once the job reached a terminal state and no further entries arrive.
 */
export function subscribeToJobLogs(
  jobId: string,
  options: { level?: LogLevel; levels?: LogLevel[]; since_seq?: number },
  onBatch: (frame: { entries: LogEntry[]; since_seq: number; status?: string }) => void,
  onDone?: () => void,
): () => void {
  const wsBase = getEffectiveApiBase().replace(/^http/, "ws");
  const params = new URLSearchParams();
  if (options.level) params.set("level", options.level);
  if (options.levels && options.levels.length) params.set("levels", options.levels.join(","));
  if (options.since_seq) params.set("since_seq", String(options.since_seq));
  const key = getApiKey();
  if (key) params.set("api_key", key);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const ws = new WebSocket(
    `${wsBase}/api/logs/${encodeURIComponent(jobId)}/stream${suffix}`,
  );
  ws.onmessage = (event) => {
    try {
      const frame = JSON.parse(event.data as string) as {
        entries?: LogEntry[];
        since_seq?: number;
        status?: string;
        error?: string;
      };
      if (frame.error) return;
      onBatch({
        entries: frame.entries ?? [],
        since_seq: frame.since_seq ?? options.since_seq ?? 0,
        status: frame.status,
      });
    } catch {
      /* ignore malformed frames */
    }
  };
  ws.onclose = () => onDone?.();
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
  config: SearchSpace;
  length?: number;
  max_generations?: number;
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
  backend?: "local" | "dask";
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

/* ------------------------------------------------------------------ */
/* Quick-start presets                                                 */
/* ------------------------------------------------------------------ */

export interface ExperimentPreset {
  value: "classification" | "regression";
  label: string;
  description: string;
  default_scoring: string;
}

export const getPresets = () => get<{ presets: ExperimentPreset[] }>("/api/experiments/presets");

/* ------------------------------------------------------------------ */
/* Explanations and prediction intervals                               */
/* ------------------------------------------------------------------ */

export interface CurveStability {
  feature: string;
  grid: Array<number | null>;
  mean: Array<number | null>;
  sample_std: Array<number | null>;
  min: Array<number | null>;
  max: Array<number | null>;
  sample_means: Array<number | null>;
}

export interface PdpStability {
  sample_count?: number;
  sample_size?: number;
  sampling?: "random_without_replacement";
  grid_source?: "primary_curve_grid";
  response?: { family: string; output: string; causal_effect: false };
  curves?: CurveStability[];
  skipped_reason?: string;
  note?: string;
}

export interface ExplainResult {
  model_version: string;
  partition: string;
  rows: number;
  reference_sample: {
    partition: string;
    source_rows: number;
    used_rows: number;
    sampling: "all_rows" | "random_without_replacement";
  };
  settings: {
    max_rows: number;
    n_repeats: number;
    top_k: number;
    stability_samples: number;
    random_state: number;
  };
  importance: Array<{ feature: string; mean: number | null; std: number | null }>;
  reference_sample_stability: {
    sample_count: number;
    sample_size: number;
    permutation_repeats_per_sample: number;
    sampling: "random_without_replacement";
    summary: Array<{
      feature: string;
      mean_importance: number | null;
      sample_std: number | null;
      top_rank_frequency: number;
    }>;
  } | null;
  partial_dependence: Array<{ feature: string; grid: Array<number | null>; average: Array<number | null> }>;
  partial_dependence_stability: PdpStability | null;
  errors: Array<{ feature: string; detail: string }>;
  interpretation: {
    importance: {
      category: "model_attribution";
      method: string;
      score: string;
      causal_effect: false;
    };
    partial_dependence: {
      category: "fitted_model_response_profile";
      reference_partition: string;
      causal_effect: false;
    };
    limitations: string[];
  };
}

export const explainBundle = (
  task: string,
  version: string,
  body: {
    partition?: string;
    max_rows?: number;
    n_repeats?: number;
    top_k?: number;
    stability_samples?: number;
  } = {},
) =>
  post<ExplainResult>(
    `/api/evaluation/${encodeURIComponent(task)}/${encodeURIComponent(version)}/explain`,
    body,
  );

export interface AleCurve {
  feature: string;
  kind: "numeric";
  grid: Array<number | null>;
  accumulated: Array<number | null>;
  counts: number[];
  excluded_rows: number;
  min_bin_fraction: number;
  stability?: {
    sample_count: number;
    sample_size: number;
    sampling: "random_without_replacement";
    mean: Array<number | null>;
    sample_std: Array<number | null>;
    min: Array<number | null>;
    max: Array<number | null>;
    note: string;
  };
}

export interface AleResult {
  model_version: string;
  partition: string;
  response: {
    family: "Regression" | "Classification";
    output: string;
    positive_class?: number | string;
    causal_effect: false;
  };
  reference_sample: {
    partition: string;
    source_rows: number;
    used_rows: number;
    sampling: "all_rows" | "random_without_replacement";
  };
  feature_selection: "explicit" | "permutation_importance_top_k";
  settings: {
    max_rows: number;
    n_bins: number;
    top_k: number;
    features: string[] | null;
    stability_samples: number;
    random_state: number;
    min_bin_fraction: number;
  };
  curves: AleCurve[];
  skipped: Array<{ feature: string; reason: string; detail: string }>;
  errors: Array<{ feature: string; detail: string }>;
  interpretation: {
    accumulated_local_effects: {
      category: "fitted_model_response_profile";
      method: string;
      reference_partition: string;
      response_output: string;
      causal_effect: false;
    };
    limitations: string[];
  };
}

export const computeAle = (
  task: string,
  version: string,
  body: {
    partition?: string;
    max_rows?: number;
    n_bins?: number;
    top_k?: number;
    features?: string[];
    stability_samples?: number;
    random_state?: number;
  } = {},
) =>
  post<AleResult>(
    `/api/evaluation/${encodeURIComponent(task)}/${encodeURIComponent(version)}/ale`,
    body,
  );

export interface RowContribution {
  feature: string;
  mean: number | null;
  sample_std: number | null;
  min: number | null;
  max: number | null;
}

export interface RowExplainResult {
  model_version: string;
  partition: string;
  row_index: number;
  row_count: number;
  response: {
    family: "Regression" | "Classification";
    output: string;
    positive_class?: number | string;
    causal_effect: false;
  };
  reference_sample: {
    partition: string;
    source_rows: number;
    used_rows: number;
    sampling: "all_rows" | "random_without_replacement" | "explicit_row";
  };
  baseline: {
    mode: "reference_mean" | "row";
    partition: string;
    row_index: number | null;
    prediction: number | null;
    values: Record<string, number | string | null>;
    value_sources: Record<string, string>;
  };
  prediction: number | null;
  explained_prediction: number | null;
  explanation_scope: "full_row" | "subset_hybrid";
  explained_features: string[];
  settings: { orders: number; reference_max_rows: number; random_state: number };
  orders: { count: number; distinct: boolean; note: string };
  contributions: RowContribution[];
  reconstruction: {
    baseline_prediction: number | null;
    explained_prediction: number | null;
    sum_contributions: number | null;
    target_change: number | null;
    max_abs_residual: number | null;
    tolerance: number | null;
    exact_by_construction: true;
  };
  interpretation: {
    category: "model_attribution";
    method: "ordered_occlusion_chain";
    causal_effect: false;
    baseline_mode: string;
    limitations: string[];
  };
}

export const explainRow = (
  task: string,
  version: string,
  body: {
    partition?: string;
    row_index?: number;
    features?: string[];
    baseline?: "reference_mean" | "row";
    baseline_partition?: string;
    baseline_row_index?: number;
    orders?: number;
    reference_max_rows?: number;
    random_state?: number;
  } = {},
) =>
  post<RowExplainResult>(
    `/api/evaluation/${encodeURIComponent(task)}/${encodeURIComponent(version)}/explain/row`,
    body,
  );

export interface IntervalResult {
  model_version: string;
  method: string;
  coverage_assumption: string;
  alpha: number;
  nominal_coverage: number;
  half_width: number | null;
  calibration_size: number;
  training_partition: string;
  fit_rows: number;
  calibration_source_partition: string;
  holdout_partition: string;
  random_state: number;
  holdout_rows: number;
  empirical_coverage: number | null;
  coverage_interval: {
    method: string;
    confidence_level: number;
    lower: number;
    upper: number;
  };
  mean_interval_width: number | null;
  median_interval_width: number | null;
  finite_interval_count: number;
  unbounded_interval_count: number;
  mean_interval_score: number | null;
  interval_score_method: string;
  diagnostic_assumption: string;
  group_by: string | null;
  subgroup_coverage: {
    group_limit: number;
    missing_group_count: number;
    other_group_count: number;
    groups: Array<{
      group: string;
      row_count: number;
      sample_size: number;
      covered_count: number;
      empirical_coverage: number;
      coverage_interval: { method: string; confidence_level: number; lower: number; upper: number };
    }>;
    interpretation: string;
  } | null;
  points: Array<{ actual: number | null; prediction: number | null; lower: number | null; upper: number | null }>;
}

export const getIntervals = (
  task: string,
  version: string,
  body: {
    alpha?: number;
    train_partition?: string;
    holdout_partition?: string | null;
    group_by?: string;
    max_groups?: number;
  } = {},
) =>
  post<IntervalResult>(
    `/api/evaluation/${encodeURIComponent(task)}/${encodeURIComponent(version)}/intervals`,
    body,
  );

/* ------------------------------------------------------------------ */
/* Experiment design                                                   */
/* ------------------------------------------------------------------ */

export interface DesignVariable {
  name: string;
  low: number;
  high: number;
}

export const latinHypercube = (body: { variables: DesignVariable[]; n_samples: number; random_state?: number }) =>
  post<{ columns: string[]; points: number[][] }>("/api/design/latin-hypercube", body);

export const suggestNextPoint = (body: {
  variables: DesignVariable[];
  samples: number[][];
  values: number[];
  n_candidates?: number;
  beta?: number;
}) => post<{ columns: string[]; result: Record<string, unknown> }>("/api/design/suggest-next", body);

/* ------------------------------------------------------------------ */
/* Research projects                                                   */
/* ------------------------------------------------------------------ */

export interface ProjectArtifact {
  kind: "dataset_partition" | "model_bundle" | "job" | "reference";
  task_name?: string | null;
  artifact_id?: string | null;
  url?: string | null;
  title?: string | null;
  doi?: string | null;
  label?: string | null;
  snapshot?: {
    schema: string;
    captured_at: string;
    fingerprint: string;
    identity: { kind: string; task_name?: string; artifact_id?: string; url?: string };
    details: Record<string, unknown>;
  };
}

export interface ProjectResearchRecord {
  record_id: string;
  kind: "hypothesis" | "analysis" | "finding" | "conclusion";
  title: string;
  text: string;
  research_question?: string;
  status?: "proposed" | "under_investigation" | "supported" | "contradicted" | "inconclusive";
  status_history?: Array<{
    status: "proposed" | "under_investigation" | "supported" | "contradicted" | "inconclusive";
    changed_at: string;
  }>;
  evidence: Array<{
    relation: "supports" | "contradicts";
    artifact: ProjectArtifact;
  }>;
  created_at: string;
}

export interface SynthesisSource {
  id: string;
  kind: string;
  label: string;
}

export interface SynthesisFact extends SynthesisSource {
  data: Record<string, unknown>;
}

export interface ProjectNote {
  note_id: string;
  text: string;
  created_at: string;
  source?: "human" | "generated";
  model?: string | null;
  citations?: string[];
  sources?: SynthesisSource[];
  computed_facts?: SynthesisFact[];
  synthesis_job_id?: string | null;
}

export interface Project {
  project_id: string;
  name: string;
  description: string;
  research_questions: string[];
  artifacts: ProjectArtifact[];
  records: ProjectResearchRecord[];
  notes: ProjectNote[];
  created_at: string;
  updated_at: string;
}

export interface ProjectSummary {
  project_id: string;
  artifacts: Array<
    ProjectArtifact & {
      exists: boolean;
      status: string | null;
      snapshot_status: "current" | "changed" | "missing" | "unverified";
    }
  >;
  stages: Array<{ stage: string; label: string; done: boolean }>;
}

export interface SynthesisStatus {
  enabled: boolean;
  provider: {
    configured: boolean;
    model: string | null;
    host: string | null;
    timeout_seconds: number;
    max_tokens: number;
    protocol: string;
  };
  instructions: Record<string, string>;
}

export interface SynthesisResult {
  project_id: string;
  task_name: string;
  model_version: string | null;
  source_job_id: string | null;
  draft: { text: string; citations: string[]; word_count: number };
  sources: SynthesisSource[];
  computed_facts: SynthesisFact[];
  provider: SynthesisStatus["provider"];
  interpretation: {
    category: "generated_interpretation";
    generated_by: string;
    causal_effect: false;
    limitations: string[];
  };
}

export const getSynthesisStatus = () => get<SynthesisStatus>("/api/synthesis/status");

export const submitSynthesis = (
  projectId: string,
  body: {
    task_name: string;
    model_version?: string | null;
    job_id?: string | null;
    partition?: string;
    include_explanations?: boolean;
    include_drift?: boolean;
    instruction?: string | null;
  },
) =>
  post<{ job_id: string; status: string; provider: SynthesisStatus["provider"] }>(
    `/api/synthesis/projects/${encodeURIComponent(projectId)}`,
    body,
  );

export const addGeneratedProjectNote = (
  projectId: string,
  body: {
    text: string;
    model: string;
    citations: string[];
    sources: SynthesisSource[];
    computed_facts: SynthesisFact[];
    synthesis_job_id?: string | null;
  },
) =>
  post<Project>(`/api/projects/${encodeURIComponent(projectId)}/notes`, {
    ...body,
    source: "generated",
  });

export const listProjects = () => get<{ projects: Project[] }>("/api/projects");
export const getProject = (id: string) => get<Project>(`/api/projects/${encodeURIComponent(id)}`);
export const getProjectSummary = (id: string) => get<ProjectSummary>(`/api/projects/${encodeURIComponent(id)}/summary`);
export const addProjectArtifact = (id: string, artifact: ProjectArtifact) =>
  post<Project>(`/api/projects/${encodeURIComponent(id)}/artifacts`, artifact);
export const createProject = (body: {
  name: string;
  description?: string;
  research_questions?: string[];
  artifacts?: ProjectArtifact[];
}) => post<Project>("/api/projects", body);

export const updateProject = (
  id: string,
  body: { name: string; description?: string; research_questions?: string[]; artifacts?: ProjectArtifact[] },
) => request<Project>("PUT", `/api/projects/${encodeURIComponent(id)}`, body);

export const addProjectNote = (id: string, text: string) =>
  post<Project>(`/api/projects/${encodeURIComponent(id)}/notes`, { text });

export const addProjectRecord = (
  id: string,
  body: Omit<ProjectResearchRecord, "record_id" | "created_at">,
) => post<Project>(`/api/projects/${encodeURIComponent(id)}/records`, body);

export const addProjectRecordEvidence = (
  projectId: string,
  recordId: string,
  evidence: ProjectResearchRecord["evidence"][number],
) =>
  post<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/records/${encodeURIComponent(recordId)}/evidence`,
    evidence,
  );

export const updateProjectHypothesisStatus = (
  projectId: string,
  recordId: string,
  status: NonNullable<ProjectResearchRecord["status"]>,
) =>
  post<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/records/${encodeURIComponent(recordId)}/status`,
    { status },
  );

/* ------------------------------------------------------------------ */
/* Identity, ownership and sharing                                     */
/* ------------------------------------------------------------------ */

export interface AuthStatus {
  auth_enforced: boolean;
  login_available: boolean;
  bootstrap_required: boolean;
  session_ttl_seconds: number;
}

export interface AuthUser {
  user_id: number;
  username: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

/** How the current request authenticated; `anonymous` means auth is not enforced. */
export interface Principal {
  user_id: number | null;
  username: string;
  is_admin: boolean;
  source: "session" | "api_key" | "static" | "anonymous";
  user: AuthUser | null;
  auth_enforced: boolean;
}

export type AclRole = "viewer" | "editor" | "owner";

export interface ApiKeyRecord {
  id: number;
  user_id: number;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface ShareEntry {
  user_id: number;
  username: string | null;
  is_active: boolean;
  role: AclRole;
  granted_by: number | null;
  created_at: string;
}

export interface SharesPayload {
  kind: string;
  resource_id: string;
  shares: ShareEntry[];
  role?: AclRole;
}

const enc = encodeURIComponent;

export const authStatus = () => get<AuthStatus>("/api/auth/status");
export const currentPrincipal = () => get<Principal>("/api/auth/me");

export const login = (username: string, password: string) =>
  post<{ user: AuthUser; expires_in: number }>("/api/auth/login", { username, password });

export const logout = () => post<{ ok: boolean }>("/api/auth/logout");

export const changePassword = (currentPassword: string, newPassword: string) =>
  post<{ ok: boolean }>("/api/auth/password", {
    current_password: currentPassword,
    new_password: newPassword,
  });

/** The caller's API keys. The secret itself is only ever returned by createApiKey. */
export const listApiKeys = async (): Promise<ApiKeyRecord[]> =>
  (await get<{ keys: ApiKeyRecord[] }>("/api/auth/keys")).keys;

export const createApiKey = (name: string) =>
  post<{ key: string; record: ApiKeyRecord }>("/api/auth/keys", { name });

export const revokeApiKey = (keyId: number) =>
  request<{ revoked: ApiKeyRecord }>("DELETE", `/api/auth/keys/${keyId}`);

export const listUsers = async (): Promise<AuthUser[]> =>
  (await get<{ users: AuthUser[] }>("/api/auth/users")).users;

export const createUser = (username: string, password: string, isAdmin = false) =>
  post<{ user: AuthUser }>("/api/auth/users", { username, password, is_admin: isAdmin });

export const updateUser = (
  userId: number,
  patch: { is_active?: boolean; is_admin?: boolean; password?: string },
) => request<{ user: AuthUser }>("PATCH", `/api/auth/users/${userId}`, patch);

/** Tasks the caller can see, with their role — empty when auth is not enforced. */
export const myTasks = () =>
  get<{ tasks: Array<{ task_name: string; role: AclRole }>; auth_enforced: boolean }>(
    "/api/access/tasks",
  );

export const myProjects = () =>
  get<{ projects: Array<{ project_id: string; role: AclRole }>; auth_enforced: boolean }>(
    "/api/access/projects",
  );

export const taskShares = (taskName: string) =>
  get<SharesPayload>(`/api/access/tasks/${enc(taskName)}`);

export const shareTask = (taskName: string, username: string, role: AclRole) =>
  post<SharesPayload>(`/api/access/tasks/${enc(taskName)}/shares`, { username, role });

export const unshareTask = (taskName: string, userId: number) =>
  request<SharesPayload>(
    "DELETE",
    `/api/access/tasks/${enc(taskName)}/shares/${userId}`,
  );

export const projectShares = (projectId: string) =>
  get<SharesPayload>(`/api/access/projects/${enc(projectId)}`);

export const shareProject = (projectId: string, username: string, role: AclRole) =>
  post<SharesPayload>(`/api/access/projects/${enc(projectId)}/shares`, { username, role });

export const unshareProject = (projectId: string, userId: number) =>
  request<SharesPayload>(
    "DELETE",
    `/api/access/projects/${enc(projectId)}/shares/${userId}`,
  );

/** Admin: give one account ownership of every unowned task/project. */
export const adoptOrphans = (kind: "task" | "project", userId?: number) =>
  post<{ kind: string; username: string; adopted: string[]; count: number }>(
    "/api/access/adopt",
    { kind, user_id: userId },
  );
