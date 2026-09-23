import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  REGISTRY_STATES,
  getDatasetMetadata,
  listBundles,
  predict,
  predictBatch,
} from "../api/client";
import type { PredictRequest, PredictResponse, PredictBatchResponse } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Card, ErrorNote, JsonView, Stat, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function InferencePage() {
  const { task } = useTask();

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

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Inference</h1>
        <span className="desc">Live predict console and batch runs for task “{task || "…"}” — every call is logged to Monitoring.</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      {task && (
        <>
          <PredictConsole task={task} versions={versions} partitions={partitions} />
          <BatchRun task={task} versions={versions} partitions={partitions} />
        </>
      )}
    </div>
  );
}

function useModelInputs() {
  const [mode, setMode] = useState<"version" | "alias">("version");
  const [modelVersion, setModelVersion] = useState("");
  const [alias, setAlias] = useState<string>("production");
  return { mode, setMode, modelVersion, setModelVersion, alias, setAlias };
}

function ModelPicker({
  versions,
  inputs,
}: {
  versions: string[];
  inputs: ReturnType<typeof useModelInputs>;
}) {
  const { mode, setMode, modelVersion, setModelVersion, alias, setAlias } = inputs;
  return (
    <div className="row">
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
              <option key={v}>{v}</option>
            ))}
          </select>
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

function buildBody(
  inputs: ReturnType<typeof useModelInputs>,
  source: InputSourceKind,
  partition: string,
  rowsJson: string,
): { body: PredictRequest; error?: string } {
  const body: PredictRequest = {};
  if (inputs.mode === "version") body.model_version = inputs.modelVersion || null;
  else body.alias = inputs.alias;

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

function PredictConsole({ task, versions, partitions }: { task: string; versions: string[]; partitions: string[] }) {
  const inputs = useModelInputs();
  const [source, setSource] = useState<InputSourceKind>("partition");
  const [partition, setPartition] = useState("");
  const [rowsJson, setRowsJson] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const mut = useMutation({ mutationFn: (body: PredictRequest) => predict(task, body), onSuccess: setResult });

  return (
    <Card
      title="Predict console"
      sub="Single request → JSON predictions. Stored partitions are projected onto the bundle's features (the target label is ignored); inline rows are validated exactly as sent."
    >
      <ModelPicker versions={versions} inputs={inputs} />
      <InputSource
        partitions={partitions}
        source={source}
        setSource={setSource}
        partition={partition}
        setPartition={setPartition}
        rowsJson={rowsJson}
        setRowsJson={setRowsJson}
      />

      {inputError && <div className="error-note">{inputError}</div>}
      {mut.isError && <ErrorNote error={mut.error} />}

      <button
        className="btn primary"
        disabled={!task || mut.isPending}
        onClick={() => {
          const { body, error } = buildBody(inputs, source, partition, rowsJson);
          setInputError(error ?? null);
          if (error) return;
          mut.mutate(body);
        }}
      >
        {mut.isPending ? "Predicting…" : "Run prediction"}
      </button>

      {result && (
        <div style={{ marginTop: 14 }}>
          <LineageRail task={task} modelVersion={result.model_version} />
          <div className="grid cols-3">
            <Stat label="Model version" value={result.model_version} />
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

function BatchRun({ task, versions, partitions }: { task: string; versions: string[]; partitions: string[] }) {
  const inputs = useModelInputs();
  const [source, setSource] = useState<InputSourceKind>("partition");
  const [partition, setPartition] = useState("");
  const [rowsJson, setRowsJson] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictBatchResponse | null>(null);

  const mut = useMutation({ mutationFn: (body: PredictRequest) => predictBatch(task, body), onSuccess: setResult });

  return (
    <Card title="Batch run" sub="Persists predictions as a CSV artifact under the task's predictions folder.">
      <ModelPicker versions={versions} inputs={inputs} />
      <InputSource
        partitions={partitions}
        source={source}
        setSource={setSource}
        partition={partition}
        setPartition={setPartition}
        rowsJson={rowsJson}
        setRowsJson={setRowsJson}
      />

      {inputError && <div className="error-note">{inputError}</div>}
      {mut.isError && <ErrorNote error={mut.error} />}

      <button
        className="btn"
        disabled={!task || mut.isPending}
        onClick={() => {
          const { body, error } = buildBody(inputs, source, partition, rowsJson);
          setInputError(error ?? null);
          if (error) return;
          mut.mutate(body);
        }}
      >
        {mut.isPending ? "Running…" : "Run batch"}
      </button>

      {result && (
        <>
          <LineageRail task={task} modelVersion={result.model_version} />
          <div className="success-note" style={{ marginTop: 12 }}>
            Wrote <code>{fmtNum(result.rows)}</code> rows for model <code>{result.model_version}</code> →{" "}
            <span className="mono">{result.output_path}</span>
          </div>
          <IgnoredColumns columns={result.ignored_columns} />
        </>
      )}
    </Card>
  );
}
