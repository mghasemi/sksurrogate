import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { BASELINE_ESTIMATORS, getBundle, listBundles, trainBaseline } from "../api/client";
import type { BundleSummary, TrainBaselineResponse } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Card, ErrorNote, JsonView, Loading, Table, fmtNum } from "../components/ui";
import { useTask } from "../lib/task-context";

export default function BundlesPage() {
  const { task } = useTask();
  const qc = useQueryClient();

  const [estimator, setEstimator] = useState<string>(BASELINE_ESTIMATORS[0]);
  const [trainPartition, setTrainPartition] = useState("train");
  const [validationPartition, setValidationPartition] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const listQ = useQuery({ queryKey: ["bundles-list", task], queryFn: () => listBundles(task), enabled: !!task });

  const detailQ = useQuery<BundleSummary>({
    queryKey: ["bundle-detail", task, selected],
    queryFn: () => getBundle(task, selected!),
    enabled: !!task && !!selected,
  });

  const trainMut = useMutation({
    mutationFn: () =>
      trainBaseline(task, {
        estimator,
        train_partition: trainPartition,
        validation_partition: validationPartition || null,
      }),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["bundles-list", task] });
      setSelected(res.model_version);
    },
  });

  const bundles = listQ.data?.bundles ?? [];

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Model Bundles</h1>
        <span className="desc">Train baselines and inspect schema, metrics & audit trail for task “{task || "…"}”</span>
      </div>

      {!task && <div className="error-note">Pick a task name in the top bar first.</div>}

      <Card title="Train a baseline" sub="Fits one estimator on a registered partition and saves a .bundle under var/sksurrogate-api/bundles/{task}/">
        <div className="row">
          <div className="field fixed" style={{ flex: "0 1 240px", minWidth: 180 }}>
            <label>Estimator</label>
            <select value={estimator} onChange={(e) => setEstimator(e.target.value)}>
              {BASELINE_ESTIMATORS.map((e) => (
                <option key={e}>{e}</option>
              ))}
            </select>
          </div>
          <div className="field fixed" style={{ flex: "0 1 150px", minWidth: 120 }}>
            <label>Train partition</label>
            <input value={trainPartition} onChange={(e) => setTrainPartition(e.target.value)} spellCheck={false} />
          </div>
          <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 130 }}>
            <label>Validation partition (optional)</label>
            <input value={validationPartition} onChange={(e) => setValidationPartition(e.target.value)} placeholder="leave empty to skip" spellCheck={false} />
          </div>
        </div>

        {trainMut.isError && <ErrorNote error={trainMut.error} />}
        {trainMut.isSuccess && (
          <div className="success-note">
            Trained <code>{(trainMut.data as TrainBaselineResponse).model_version}</code> —{" "}
            {Object.entries((trainMut.data as TrainBaselineResponse).metrics)
              .map(([k, v]) => `${k}=${fmtNum(v)}`)
              .join(", ")}
          </div>
        )}

        <button className="btn primary" disabled={!task || trainMut.isPending} onClick={() => trainMut.mutate()}>
          {trainMut.isPending ? "Training…" : "Train baseline"}
        </button>
      </Card>

      <Card title={`Bundles (${fmtNum(bundles.length)})`}>
        {listQ.isLoading && task && <Loading />}
        {listQ.isError && <ErrorNote error={listQ.error} />}
        {bundles.length === 0 && !listQ.isLoading && (
          <p className="muted">No bundles yet — train a baseline above or run an experiment.</p>
        )}
        {bundles.length > 0 && (
          <Table<string>
            columns={["Model version", "Selected"]}
            rows={[...bundles].reverse()}
            keyOf={(v) => v}
            render={(v) => [
              <td key="v" className="mono">{v}</td>,
              <td key="s">
                <button className={`btn${selected === v ? " primary" : ""}`} onClick={() => setSelected(v)}>
                  Inspect
                </button>
              </td>,
            ]}
          />
        )}
      </Card>

      {selected && (
        <>
          <LineageRail task={task} modelVersion={selected} />
          <Card title={`Bundle ${selected}`} sub={detailQ.data?.created_at}>
            {detailQ.isLoading && <Loading />}
            {detailQ.isError && <ErrorNote error={detailQ.error} />}
          {detailQ.data && (
            <>
              <div className="grid cols-2">
                <div>
                  <h3 style={{ marginTop: 0 }}>Metrics</h3>
                  <dl className="kv">
                    {Object.entries(detailQ.data.metrics).map(([k, v]) => (
                      <MetricRow key={k} k={k} v={v} />
                    ))}
                  </dl>

                  <h3 style={{ marginTop: 16 }}>Schema</h3>
                  <div className="chips">
                    {Object.entries(detailQ.data.schema).map(([col, s]) => (
                      <span key={col} className="chip" title={s.dtype}>
                        {col}: {s.dtype}
                      </span>
                    ))}
                  </div>

                  <h3 style={{ marginTop: 16 }}>Lineage</h3>
                  <dl className="kv">
                    <dt>Dataset fingerprint</dt>
                    <dd className="mono">{detailQ.data.dataset_fingerprint ?? "—"}</dd>
                    <dt>Run id</dt>
                    <dd className="mono">{detailQ.data.run_id ?? "—"}</dd>
                    <dt>Owner</dt>
                    <dd>{detailQ.data.owner ?? "—"}</dd>
                  </dl>
                </div>

                <div>
                  <h3 style={{ marginTop: 0 }}>Dependencies</h3>
                  <JsonView data={detailQ.data.dependencies} />
                  <h3 style={{ marginTop: 16 }}>Audit events</h3>
                  <JsonView data={detailQ.data.audit_events} />
                </div>
              </div>
            </>
          )}
          </Card>
        </>
      )}
    </div>
  );
}

function MetricRow({ k, v }: { k: string; v: number }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{fmtNum(v)}</dd>
    </>
  );
}
