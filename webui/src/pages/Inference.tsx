import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Download } from "lucide-react";

import {
  REGISTRY_STATES,
  findBestBundle,
  forecast,
  forecastServed,
  getDatasetMetadata,
  getEffectiveApiBase,
  getServing,
  listBundles,
  predict,
  predictBatch,
  predictionDownloadUrl,
  predictServed,
  retryUnlessNotFound,
  startServing,
  stopServing,
} from "../api/client";
import type {
  ForecastRequest,
  ForecastResponse,
  PredictRequest,
  PredictResponse,
  PredictBatchResponse,
  ServedForecastRequest,
  ServedPredictRequest,
  ServedPredictResponse,
} from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, JsonView, Stat, fmtNum } from "../components/ui";
import { useBundleLabel } from "../lib/hooks";
import { useTask } from "../lib/task-context";

export default function InferencePage() {
  const { task } = useTask();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialVersion = searchParams.get("model_version") ?? "";

  const bundlesQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });
  const datasetQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
    retry: false,
  });

  const versions = useMemo(() => bundlesQ.data?.bundles ?? [], [bundlesQ.data]);
  const partitions = useMemo(
    () => Object.keys(datasetQ.data?.partitions ?? {}),
    [datasetQ.data],
  );

  useEffect(() => {
    if (!task || !initialVersion) return;
    const params = new URLSearchParams(searchParams);
    params.delete("model_version");
    setSearchParams(params, { replace: true });
  }, [initialVersion, searchParams, setSearchParams, task]);

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Inference</h1>
        <span className="desc">Live predict console and batch runs for task “{task || "…"}” — every call is logged to Monitoring.</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {task && (
        <>
          <ServingCard task={task} versions={versions} partitions={partitions} />
          <PredictConsole key={`predict-${task}`} task={task} versions={versions} partitions={partitions} initialVersion={initialVersion} />
          <ForecastConsole key={`forecast-${task}`} task={task} versions={versions} initialVersion={initialVersion} />
          <BatchRun key={`batch-${task}`} task={task} versions={versions} partitions={partitions} initialVersion={initialVersion} />
        </>
      )}
    </div>
  );
}

function useModelInputs(initialVersion: string) {
  const [mode, setMode] = useState<"version" | "alias">("version");
  const [modelVersion, setModelVersion] = useState(initialVersion);
  const [alias, setAlias] = useState<string>("production");
  return { mode, setMode, modelVersion, setModelVersion, alias, setAlias };
}

function ModelPicker({
  task,
  versions,
  inputs,
}: {
  task: string;
  versions: string[];
  inputs: ReturnType<typeof useModelInputs>;
}) {
  const { mode, setMode, modelVersion, setModelVersion, alias, setAlias } = inputs;
  // Phase 3.2: suggest the bundle with the best recorded score, but never
  // preselect it — inference on production data stays an explicit choice.
  const bestQ = useQuery({
    queryKey: ["bundle-best", task],
    queryFn: () => findBestBundle(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });
  const bestVersion = bestQ.data?.model_version ?? null;
  const bestIsAvailable = !!bestVersion && versions.includes(bestVersion);

  const labelOf = useBundleLabel(task);

  return (
    <div className="row top">
      <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
        <label>Resolve by</label>
        <select value={mode} onChange={(e) => setMode(e.target.value as "version" | "alias")}>
          <option value="version">Model version</option>
          <option value="alias">Registry alias</option>
        </select>
      </div>
      {mode === "version" ? (
        <div className="field fixed" style={{ flex: "0 1 320px", minWidth: 240 }}>
          <label>Model version</label>
          <select value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>
            <option value="">— select —</option>
            {versions.map((v) => (
              <option key={v} value={v}>{labelOf(v)}</option>
            ))}
          </select>
          {bestIsAvailable && (
            <div className="chips" style={{ marginTop: 6 }}>
              <button
                type="button"
                className="chip chip-best"
                title={`Highest ${bestQ.data?.metric ?? "score"} metric: ${bestQ.data?.value ?? "—"}`}
                onClick={() => setModelVersion(bestVersion!)}
              >
                ★ best: {labelOf(bestVersion)}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="field fixed" style={{ flex: "0 1 200px", minWidth: 150 }}>
          <label>Alias</label>
          <select value={alias} onChange={(e) => setAlias(e.target.value)}>
            {REGISTRY_STATES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

type InputSourceKind = "partition" | "rows";

function InputSource({
  partitions,
  source,
  setSource,
  partition,
  setPartition,
  rowsJson,
  setRowsJson,
}: {
  partitions: string[];
  source: InputSourceKind;
  setSource: (v: InputSourceKind) => void;
  partition: string;
  setPartition: (v: string) => void;
  rowsJson: string;
  setRowsJson: (v: string) => void;
}) {
  return (
    <>
      <div className="row">
        <div className="field fixed" style={{ flex: "0 1 180px", minWidth: 140 }}>
          <label>Input source</label>
          <select value={source} onChange={(e) => setSource(e.target.value as InputSourceKind)}>
            <option value="partition">Stored partition</option>
            <option value="rows">Inline rows (JSON)</option>
          </select>
        </div>
        {source === "partition" && (
          <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
            <label>Partition</label>
            <select value={partition} onChange={(e) => setPartition(e.target.value)}>
              <option value="">— select —</option>
              {partitions.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      {source === "rows" && (
        <textarea
          className="mono"
          rows={6}
          value={rowsJson}
          onChange={(e) => setRowsJson(e.target.value)}
          placeholder='[{"feature_a": 1.0, "feature_b": 2}, …]'
          spellCheck={false}
        />
      )}
    </>
  );
}

/** Non-schema columns the API dropped from a stored partition (e.g. the target label). */
function IgnoredColumns({ columns }: { columns?: string[] }) {
  if (!columns || columns.length === 0) return null;
  return (
    <div className="hint" style={{ marginTop: 10 }}>
      Ignored (not part of the bundle schema):{" "}
      <span className="mono">{columns.join(", ")}</span>
    </div>
  );
}

/** Last path component of a server-side artifact path — the file to download. */
function artifactFilename(outputPath: string): string {
  return outputPath.replace(/\\/g, "/").split("/").pop() || "predictions.csv";
}

function buildSource(
  source: InputSourceKind,
  partition: string,
  rowsJson: string,
): { body: PredictRequest; error?: string } {
  const body: PredictRequest = {};
  if (source === "partition") {
    if (!partition) return { body, error: "Choose a stored partition." };
    body.partition = partition;
  } else {
    try {
      const parsed = JSON.parse(rowsJson);
      if (!Array.isArray(parsed)) throw new Error("rows must be a JSON array of objects");
      body.rows = parsed;
    } catch (e) {
      return { body, error: `Invalid rows JSON: ${e instanceof Error ? e.message : String(e)}` };
    }
  }
  return { body };
}

function buildBody(
  inputs: ReturnType<typeof useModelInputs>,
  source: InputSourceKind,
  partition: string,
  rowsJson: string,
): { body: PredictRequest; error?: string } {
  const { body, error } = buildSource(source, partition, rowsJson);
  if (inputs.mode === "version") body.model_version = inputs.modelVersion || null;
  else body.alias = inputs.alias;
  return { body, error };
}

function PredictConsole({ task, versions, partitions, initialVersion }: { task: string; versions: string[]; partitions: string[]; initialVersion: string }) {
  const inputs = useModelInputs(initialVersion);
  const labelOf = useBundleLabel(task);
  const [source, setSource] = useState<InputSourceKind>("partition");
  const [partition, setPartition] = useState("");
  const [rowsJson, setRowsJson] = useState("");
  const [preflight, setPreflight] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const mut = useMutation({ mutationFn: (body: PredictRequest) => predict(task, body), onSuccess: setResult });

  return (
    <Card
      title="Predict console"
      sub="Single request → JSON predictions. Stored partitions are projected onto the bundle's features (the target label is ignored); inline rows are validated exactly as sent."
    >
      <ModelPicker task={task} versions={versions} inputs={inputs} />
      {inputs.mode === "version" && inputs.modelVersion && !result && (
        <LineageRail task={task} modelVersion={inputs.modelVersion} />
      )}
      <InputSource
        partitions={partitions}
        source={source}
        setSource={setSource}
        partition={partition}
        setPartition={setPartition}
        rowsJson={rowsJson}
        setRowsJson={setRowsJson}
      />
      <label className="checkbox-row" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={preflight} onChange={(e) => setPreflight(e.target.checked)} />
        Validate before run
      </label>

      {inputError && <div className="error-note">{inputError}</div>}
      {mut.isError && <ErrorNote error={mut.error} />}

      <button
        className="btn primary"
        disabled={!task || mut.isPending}
        onClick={() => {
          const { body, error } = buildBody(inputs, source, partition, rowsJson);
          setInputError(error ?? null);
          if (error) return;
          body.preflight = preflight;
          mut.mutate(body);
        }}
      >
        {mut.isPending ? "Predicting…" : "Run prediction"}
      </button>

      {result && (
        <div style={{ marginTop: 14 }}>
          <LineageRail task={task} modelVersion={result.model_version} />
          <div className="grid cols-3">
            <Stat label="Model version" value={labelOf(result.model_version)} />
            <Stat label="Rows" value={fmtNum(result.metrics.rows)} />
            <Stat label="Latency" value={`${fmtNum(result.metrics.latency_ms)} ms`} />
          </div>
          <IgnoredColumns columns={result.ignored_columns} />
          <h3 style={{ marginTop: 14 }}>Predictions</h3>
          <JsonView data={result.predictions} />
        </div>
      )}
    </Card>
  );
}

/** Parse pasted CSV (header row + data rows) into history records; numbers are coerced. */
function parseHistoryCsv(text: string): { rows: Array<Record<string, unknown>>; error?: string } {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return { rows: [], error: "Paste a header row plus at least one observation row." };
  }
  const header = lines[0].split(",").map((cell) => cell.trim());
  const rows = lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row: Record<string, unknown> = {};
    header.forEach((name, index) => {
      const raw = (cells[index] ?? "").trim();
      // Numeric coercion only when the cell looks numeric — epoch timestamps and
      // ISO dates both survive as their original text.
      row[name] = raw !== "" && /^-?\d/.test(raw) && Number.isFinite(Number(raw)) ? Number(raw) : raw;
    });
    return row;
  });
  return { rows };
}

function downloadCsv(
  filename: string,
  header: string[],
  rows: Array<Array<string | number | null>>,
): void {
  const escape = (value: string | number | null) => {
    const text = value == null ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const text = [header.join(","), ...rows.map((row) => row.map(escape).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Forecast console for temporal bundles (plan phase P7.4).
 *
 * The multi-step counterpart of the predict console: the payload is *history*
 * rows (one series), not feature rows — the bundle's stored design validates
 * it. Works against an exact model/alias pick or the pinned serving model (the
 * deployment path, which answers 409 while nothing is pinned). Results can be
 * downloaded as the classic time/prediction/lower/upper CSV.
 */
function ForecastConsole({
  task,
  versions,
  initialVersion,
}: {
  task: string;
  versions: string[];
  initialVersion: string;
}) {
  const inputs = useModelInputs(initialVersion);
  const labelOf = useBundleLabel(task);
  const [historyCsv, setHistoryCsv] = useState("");
  const [horizon, setHorizon] = useState(1);
  const [useServed, setUseServed] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<ForecastResponse | null>(null);

  // Default to the newest bundle: a forecast with no model id is a 400, and a
  // blank picker reads like a broken card rather than "pick a model".
  useEffect(() => {
    if (inputs.mode === "version" && !inputs.modelVersion && versions.length > 0) {
      inputs.setModelVersion(versions[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputs.mode, inputs.modelVersion, versions]);

  const mut = useMutation({
    mutationFn: (vars: { body: ForecastRequest | ServedForecastRequest; served: boolean }) =>
      vars.served ? forecastServed(task, vars.body) : forecast(task, vars.body),
    onSuccess: setResult,
  });

  const submit = () => {
    const { rows, error } = parseHistoryCsv(historyCsv);
    setInputError(error ?? null);
    if (error) return;
    const body: ForecastRequest = {
      history: rows,
      horizon: Math.max(1, horizon),
    };
    if (!useServed) {
      if (inputs.mode === "version") body.model_version = inputs.modelVersion || null;
      else body.alias = inputs.alias;
    }
    mut.mutate({ body: useServed ? { history: rows, horizon: body.horizon } : body, served: useServed });
  };

  const rows =
    result?.times.map((time, index) => ({
      time,
      prediction: result.predictions[index] ?? null,
      lower: result.lower_bounds?.[index] ?? null,
      upper: result.upper_bounds?.[index] ?? null,
    })) ?? [];

  return (
    <Card
      title="Forecast console (temporal bundles)"
      sub="Multi-step path: send one series of past observations as CSV (header + rows) and a horizon. The bundle's stored temporal design validates the history — a too-short or out-of-order series is refused with the reason, never silently padded."
    >
      <ModelPicker task={task} versions={versions} inputs={inputs} />
      <label className="checkbox-row">
        <input type="checkbox" checked={useServed} onChange={(e) => setUseServed(e.target.checked)} />
        Use the pinned serving model (ignore the picker above; 409 while nothing is pinned)
      </label>

      <div className="row top" style={{ marginTop: 8 }}>
        <div className="field" style={{ flex: "1 1 480px" }}>
          <label>History CSV</label>
          <textarea
            className="mono"
            rows={7}
            spellCheck={false}
            placeholder={"time,value\n2026-01-01,10.5\n2026-01-02,10.9\n2026-01-03,11.2"}
            value={historyCsv}
            onChange={(e) => setHistoryCsv(e.target.value)}
          />
        </div>
        <div className="field fixed" style={{ flex: "0 1 130px", minWidth: 100 }}>
          <label>Horizon (steps)</label>
          <input
            type="number"
            min={1}
            step={1}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </div>
      </div>

      {inputError && <div className="error-note">{inputError}</div>}
      {mut.isError && <ErrorNote error={mut.error} />}

      <button className="btn primary" disabled={!task || mut.isPending} onClick={submit}>
        {mut.isPending ? "Forecasting…" : "Run forecast"}
      </button>

      {result && (
        <div style={{ marginTop: 14 }}>
          <div className="grid cols-3">
            <Stat label="Model version" value={labelOf(result.model_version)} />
            <Stat
              label="Strategy"
              value={result.strategy}
              sub={useServed && result.requested ? `served via ${result.resolve_mode}: ${result.requested}` : "direct or recursive per bundle"}
            />
            <Stat label="Horizon" value={fmtNum(result.horizon)} sub={`${fmtNum(result.metrics.latency_ms)} ms`} />
          </div>
          {!result.lower_bounds && (
            <p className="muted">{result.interval_note ?? "No interval calibration on this bundle."}</p>
          )}
          <h3 style={{ marginTop: 14 }}>Forecast</h3>
          <table className="tbl">
            <thead>
              <tr>
                <th>Step</th>
                <th>Time</th>
                <th>Prediction</th>
                <th>Lower</th>
                <th>Upper</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${row.time}-${index}`}>
                  <td>{index + 1}</td>
                  <td className="mono">{row.time}</td>
                  <td>{fmtNum(row.prediction)}</td>
                  <td>{fmtNum(row.lower)}</td>
                  <td>{fmtNum(row.upper)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            className="btn"
            style={{ marginTop: 10 }}
            onClick={() =>
              downloadCsv(
                `forecast-${task}-${result.model_version}.csv`,
                ["time", "prediction", "lower", "upper"],
                rows.map((row) => [row.time, row.prediction, row.lower, row.upper]),
              )
            }
          >
            <Download size={14} /> Download forecast CSV
          </button>
        </div>
      )}
    </Card>
  );
}

function BatchRun({ task, versions, partitions, initialVersion }: { task: string; versions: string[]; partitions: string[]; initialVersion: string }) {
  const inputs = useModelInputs(initialVersion);
  const labelOf = useBundleLabel(task);
  const [source, setSource] = useState<InputSourceKind>("partition");
  const [partition, setPartition] = useState("");
  const [rowsJson, setRowsJson] = useState("");
  const [preflight, setPreflight] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictBatchResponse | null>(null);

  const mut = useMutation({ mutationFn: (body: PredictRequest) => predictBatch(task, body), onSuccess: setResult });

  return (
    <Card title="Batch run" sub="Persists predictions as a CSV artifact under the task's predictions folder.">
      <ModelPicker task={task} versions={versions} inputs={inputs} />
      <InputSource
        partitions={partitions}
        source={source}
        setSource={setSource}
        partition={partition}
        setPartition={setPartition}
        rowsJson={rowsJson}
        setRowsJson={setRowsJson}
      />
      <label className="checkbox-row" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={preflight} onChange={(e) => setPreflight(e.target.checked)} />
        Validate before run
      </label>

      {inputError && <div className="error-note">{inputError}</div>}
      {mut.isError && <ErrorNote error={mut.error} />}

      <button
        className="btn"
        disabled={!task || mut.isPending}
        onClick={() => {
          const { body, error } = buildBody(inputs, source, partition, rowsJson);
          setInputError(error ?? null);
          if (error) return;
          body.preflight = preflight;
          mut.mutate(body);
        }}
      >
        {mut.isPending ? "Running…" : "Run batch"}
      </button>

      {result && (
        <>
          <LineageRail task={task} modelVersion={result.model_version} />
          <div className="success-note" style={{ marginTop: 12 }}>
            Wrote <code>{fmtNum(result.rows)}</code> rows for model <code>{labelOf(result.model_version)}</code> →{" "}
            <span className="mono">{result.output_path}</span>
          </div>
          <div className="row top" style={{ marginTop: 10, gap: 8 }}>
            <a
              className="btn"
              href={predictionDownloadUrl(task, result.model_version, artifactFilename(result.output_path))}
              download
              title={`Download ${result.output_path}`}
            >
              <Download size={15} /> Download CSV
            </a>
          </div>
          <IgnoredColumns columns={result.ignored_columns} />
        </>
      )}
    </Card>
  );
}

/**
 * Serving card: the deployment declaration for this task.
 *
 * Every other card on this page names a model per request. This one *declares*
 * the model the task serves — the state `/api/serving/{task}/predict` scores
 * through and the top-bar indicator reports — and lets the operator exercise
 * that endpoint. The two resolution modes are a real choice, so the card says
 * which one is in force: a version pin is frozen against later promotions,
 * while an alias pin follows the registry (and the resolved version is shown,
 * not just the alias name).
 */
function ServingCard({ task, versions, partitions }: { task: string; versions: string[]; partitions: string[] }) {
  const qc = useQueryClient();
  const pinQ = useQuery({ queryKey: ["serving", task], queryFn: () => getServing(task), enabled: !!task });
  const inputs = useModelInputs("");
  const [note, setNote] = useState("");
  const [source, setSource] = useState<InputSourceKind>("partition");
  const [partition, setPartition] = useState("");
  const [rowsJson, setRowsJson] = useState("");
  const [preflight, setPreflight] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<ServedPredictResponse | null>(null);

  // One key prefix for the pin: ["serving", "active"] (top bar) and
  // ["serving", task] (this card) both refresh from a single invalidation.
  const refresh = () => qc.invalidateQueries({ queryKey: ["serving"] });

  const startMut = useMutation({
    mutationFn: () =>
      startServing(task, {
        model_version: inputs.mode === "version" ? inputs.modelVersion || null : null,
        alias: inputs.mode === "alias" ? inputs.alias : null,
        note: note.trim() || null,
      }),
    onSuccess: () => {
      setResult(null);
      refresh();
    },
  });
  const stopMut = useMutation({
    mutationFn: () => stopServing(task),
    onSuccess: () => {
      setResult(null);
      refresh();
    },
  });
  const predictMut = useMutation({
    mutationFn: (body: ServedPredictRequest) => predictServed(task, body),
    onSuccess: setResult,
  });

  const pin = pinQ.data;
  const active = !!pin?.active;
  const labelOf = useBundleLabel(task);
  const needsModel = inputs.mode === "version" ? !inputs.modelVersion : !inputs.alias;
  const busy = startMut.isPending || stopMut.isPending || predictMut.isPending;
  const history = [...(pin?.history ?? [])].reverse();
  // The reachable endpoint a deployment client points at (same-origin base, so it
  // honors both the dev-server proxy and the container's served origin).
  const endpointUrl = `${getEffectiveApiBase()}/api/serving/${encodeURIComponent(task)}/predict`;

  return (
    <Card
      title="Serving"
      sub="Declare the model this task serves. The serving endpoint answers with it without being told which model to use, and the top bar shows the declaration."
    >
      <div className="row top" style={{ alignItems: "center", marginBottom: 4 }}>
        <Badge tone={pin?.resolve_error ? "warn" : active ? "ok" : "muted"}>
          {active ? `Serving ${pin?.model_version ? labelOf(pin.model_version) : "…"}` : "Not serving"}
        </Badge>
        {active && pin && (
          <span className="hint">
            {pin.resolve_mode === "alias"
              ? `follows alias “${pin.requested}” — a promotion or rollback changes the served model`
              : `frozen at model version “${labelOf(pin.model_version)}” — later promotions do not change it`}
          </span>
        )}
      </div>

      {pinQ.isError && <ErrorNote error={pinQ.error} />}
      {pin?.resolve_error && (
        <div className="error-note">The pinned model no longer resolves: {pin.resolve_error}</div>
      )}

      {active && pin && (
        <dl className="kv" style={{ marginTop: 10 }}>
          <dt>Model version</dt>
          <dd className="mono">{labelOf(pin.model_version)}</dd>
          <dt>Endpoint</dt>
          <dd>
            <code className="mono">{endpointUrl}</code>
          </dd>
          <dt>Resolve mode</dt>
          <dd>
            {pin.resolve_mode} · <span className="mono">{pin.requested}</span>
          </dd>
          <dt>Serving since</dt>
          <dd>{pin.started_at ? new Date(pin.started_at).toLocaleString() : "—"}</dd>
          <dt>Started by</dt>
          <dd>{pin.started_by ?? "—"}</dd>
          {pin.note && (
            <>
              <dt>Note</dt>
              <dd>{pin.note}</dd>
            </>
          )}
        </dl>
      )}

      <ModelPicker task={task} versions={versions} inputs={inputs} />
      <div className="row top">
        <div className="field">
          <label>Note (optional)</label>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="e.g. canary rollout"
            spellCheck={false}
          />
        </div>
      </div>
      <div className="row top" style={{ gap: 8 }}>
        <button
          className="btn primary"
          disabled={!task || needsModel || busy}
          onClick={() => startMut.mutate()}
        >
          {active ? "Replace served model" : "Serve this model"}
        </button>
        <button className="btn" disabled={!task || !active || busy} onClick={() => stopMut.mutate()}>
          Stop serving
        </button>
      </div>
      {needsModel && <div className="hint" style={{ marginTop: 8 }}>Pick a model version or alias to serve.</div>}
      {startMut.isError && <ErrorNote error={startMut.error} />}
      {stopMut.isError && <ErrorNote error={stopMut.error} />}

      <h3 style={{ marginTop: 18 }}>Score through the served model</h3>
      <div className="hint">
        Sends no model id — the pin decides, so this is the payload a deployment client uses.{" "}
        {active ? "" : "Start serving to enable it."}
      </div>
      <InputSource
        partitions={partitions}
        source={source}
        setSource={setSource}
        partition={partition}
        setPartition={setPartition}
        rowsJson={rowsJson}
        setRowsJson={setRowsJson}
      />
      <label className="checkbox-row" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={preflight} onChange={(e) => setPreflight(e.target.checked)} />
        Validate before run
      </label>

      {inputError && <div className="error-note">{inputError}</div>}
      {predictMut.isError && <ErrorNote error={predictMut.error} />}

      <button
        className="btn"
        disabled={!task || !active || busy}
        onClick={() => {
          const { body, error } = buildSource(source, partition, rowsJson);
          setInputError(error ?? null);
          if (error) return;
          body.preflight = preflight;
          predictMut.mutate(body);
        }}
      >
        {predictMut.isPending ? "Predicting…" : "Run on served model"}
      </button>

      {result && (
        <div style={{ marginTop: 14 }}>
          <div className="success-note">
            Answered by <code>{labelOf(result.model_version)}</code> (
            {result.resolve_mode === "alias" ? `alias “${result.requested}”` : "pinned version"}) ·{" "}
            {fmtNum(result.metrics.rows)} row(s) · {fmtNum(result.metrics.latency_ms)} ms
          </div>
          <IgnoredColumns columns={result.ignored_columns} />
          <h3 style={{ marginTop: 14 }}>Predictions</h3>
          <JsonView data={result.predictions} />
        </div>
      )}

      {history.length > 0 && (
        <>
          <h3 style={{ marginTop: 18 }}>Pin history</h3>
          <ul className="pin-history">
            {history.slice(0, 8).map((event, index) => (
              <li key={`${event.at}-${index}`}>
                <Badge tone={event.event === "stop" ? "warn" : "ok"}>{event.event}</Badge>
                <span className="mono">{labelOf(event.model_version)}</span>
                <span className="muted">
                  {event.resolve_mode === "alias" ? `alias ${event.requested}` : `version ${event.requested}`} ·{" "}
                  {event.actor ?? "—"} · {new Date(event.at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}
