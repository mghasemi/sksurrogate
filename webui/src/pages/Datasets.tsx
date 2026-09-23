import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";

import { getDatasetMetadata, previewDataset, registerDataset } from "../api/client";
import type { DatasetPreview, RegisterDatasetResponse } from "../api/client";
import { Card, ErrorNote, Loading, Table, Badge, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function DatasetsPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const [target, setTarget] = useState("");
  const [partition, setPartition] = useState("train");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const metaQ = useQuery({
    queryKey: ["dataset-meta", task],
    queryFn: () => getDatasetMetadata(task),
    enabled: !!task,
  });

  const [previewPartition, setPreviewPartition] = useState<string | null>(null);
  const previewQ = useQuery<DatasetPreview>({
    queryKey: ["dataset-preview", task, previewPartition],
    queryFn: () => previewDataset(task, previewPartition!),
    enabled: !!task && !!previewPartition,
  });

  const registerMut = useMutation({
    mutationFn: (vars: { target: string; partition: string; file: File }) =>
      registerDataset(task, vars.target, vars.partition, vars.file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-meta", task] });
      setFile(null);
      if (fileRef.current) fileRef.current.value = "";
    },
  });

  const partitions = metaQ.data ? Object.keys(metaQ.data.partitions ?? {}) : [];
  const previewRows = previewQ.data?.rows ?? [];
  const previewCols = previewRows.length ? Object.keys(previewRows[0]) : [];

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Datasets</h1>
        <span className="desc">Register CSV partitions, inspect schema and fingerprint for task “{task || "…"}”</span>
      </div>

      {!task && (
        <div className="error-note">Pick a task name in the top bar to work with datasets.</div>
      )}

      <Card title="Register a partition" sub="Server infers dtypes and stores the CSV under var/sksurrogate-api/datasets/{task}/">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 180 }}>
            <label>Target column</label>
            <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. target" spellCheck={false} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 160px", minWidth: 120 }}>
            <label>Partition</label>
            <input value={partition} onChange={(e) => setPartition(e.target.value)} placeholder="train / validation / test" spellCheck={false} />
          </div>
        </div>

        <div
          className={`file-drop${dragOver ? " drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) setFile(f);
          }}
          onClick={() => fileRef.current?.click()}
        >
          <UploadCloud size={22} style={{ marginBottom: 6 }} />
          {file ? (
            <span>
              <strong>{file.name}</strong> ({fmtNum(file.size)} bytes) — click to replace
            </span>
          ) : (
            <span>Drop a CSV here or click to browse</span>
          )}
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.tsv,text/csv"
            style={{ display: "none" }}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        {registerMut.isError && <ErrorNote error={registerMut.error} />}
        {registerMut.isSuccess && (
          <div className="success-note">
            Registered {fmtNum((registerMut.data as RegisterDatasetResponse).rows)} rows → fingerprint{" "}
            <code>{(registerMut.data as RegisterDatasetResponse).dataset_fingerprint ?? "n/a"}</code>
          </div>
        )}

        <button
          className="btn primary"
          disabled={!task || !target || !partition || !file || registerMut.isPending}
          onClick={() => file && registerMut.mutate({ target, partition, file })}
        >
          {registerMut.isPending ? "Registering…" : "Register dataset"}
        </button>
      </Card>

      <Card title="Registered metadata" sub={metaQ.data?.dataset_fingerprint ?? undefined}>
        {metaQ.isLoading && task && <Loading />}
        {metaQ.isError && <ErrorNote error={metaQ.error} />}
        {metaQ.data && (
          <>
            <dl className="kv">
              <dt>Target</dt>
              <dd>{metaQ.data.target_name ?? "—"}</dd>
              <dt>Fingerprint</dt>
              <dd className="mono">{metaQ.data.dataset_fingerprint ?? "—"}</dd>
              <dt>Feature count</dt>
              <dd>{fmtNum(metaQ.data.dataset_feature_count)}</dd>
            </dl>

            {partitions.length > 0 && (
              <>
                <h3 style={{ marginTop: 16 }}>Partitions</h3>
                <Table<string>
                  columns={["Partition", "Rows", "Ingested at", "Fingerprint"]}
                  rows={partitions}
                  keyOf={(p) => p}
                  render={(p) => {
                    const info = metaQ.data!.partitions[p];
                    return [
                      <td key="p" className="mono">{p}</td>,
                      <td key="r">{fmtNum(info.rows)}</td>,
                      <td key="i" className="mono">{info.ingestion_timestamp}</td>,
                      <td key="f"><Badge tone="info">{(info.fingerprint ?? "").slice(0, 12)}…</Badge></td>,
                    ];
                  }}
                />

                {previewPartition && (
                  <>
                    <h3 style={{ marginTop: 16 }}>Preview — {previewPartition}</h3>
                    {previewQ.isLoading && <Loading />}
                    {previewQ.isError && <ErrorNote error={previewQ.error} />}
                    {previewQ.data && previewRows.length > 0 && (
                      <Table<Record<string, unknown>>
                        columns={previewCols}
                        rows={previewRows}
                        keyOf={(_, i) => i}
                        render={(r) =>
                          previewCols.map((c) => (
                            <td key={c} className="mono">
                              {String(r[c] ?? "")}
                            </td>
                          ))
                        }
                      />
                    )}
                  </>
                )}

                <div style={{ marginTop: 12 }}>
                  {partitions.map((p) => (
                    <button
                      key={p}
                      className={`btn${previewPartition === p ? " primary" : ""}`}
                      style={{ marginRight: 8 }}
                      onClick={() => setPreviewPartition(previewPartition === p ? null : p)}
                    >
                      Preview {p}
                    </button>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
