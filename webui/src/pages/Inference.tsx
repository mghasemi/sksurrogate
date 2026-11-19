import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Download } from "lucide-react";

import {
  REGISTRY_STATES,
  findBestBundle,
  getDatasetMetadata,
  getEffectiveApiBase,
  getServing,
  listBundles,
  predict,
  predictBatch,
  predictionDownloadUrl,
  predictServed,
  registrySummary,
  retryUnlessNotFound,
  startServing,
  stopServing,
} from "../api/client";
import type {
  PredictRequest,
  PredictResponse,
  PredictBatchResponse,
  ServedPredictRequest,
  ServedPredictResponse,
} from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, JsonView, Stat, fmtNum } from "../components/ui";
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

/**
 * Alias-aware display label for a model version: ``"{alias(es)} - {version}"``
 * when a named registry alias has been set, otherwise just the version. The
 * Registry page owns the ``["registry-summary", task]`` cache entry, so calling
 * this from the several cards on the Inference page costs one request total.
 */
function useBundleLabel(task: string): (version: string | null | undefined) => string {
  const aliasQ = useQuery({
    queryKey: ["registry-summary", task],
    queryFn: () => registrySummary(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });
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
  return useMemo(
    () => (version) => {
      if (!version) return "—";
      const aliases = aliasByVersion.get(version);
      return aliases && aliases.length > 0 ? `${aliases.join(", ")} - ${version}` : version;
    },
    [aliasByVersion],
  );
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
                className="chip"
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
