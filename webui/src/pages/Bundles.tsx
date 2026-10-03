import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Package } from "lucide-react";

import {
  BASELINE_ESTIMATORS,
  bundleDownloadUrl,
  exportMlflowUrl,
  findBestBundle,
  getBundle,
  listBundles,
  listPreservedModels,
  preserveBundle,
  recoverModel,
  retryUnlessNotFound,
  trainBaseline,
} from "../api/client";
import type { BundleSummary, PreservedSnapshot, TrainBaselineResponse } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, ErrorNote, JsonView, Loading, Table, fmtNum } from "../components/ui";
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

  // A 404 here means "no bundle carries that metric yet" — not an error worth showing.
  const bestQ = useQuery({
    queryKey: ["bundle-best", task],
    queryFn: () => findBestBundle(task),
    enabled: !!task,
    retry: retryUnlessNotFound,
  });
  const bestVersion = bestQ.data?.model_version ?? null;

  const preservedQ = useQuery({
    queryKey: ["bundle-preserved", task],
    queryFn: () => listPreservedModels(task),
    enabled: !!task,
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
      qc.invalidateQueries({ queryKey: ["bundle-best", task] });
      setSelected(res.model_version);
    },
  });

  const preserveMut = useMutation({
    mutationFn: (version: string) => preserveBundle(task, version),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["bundle-preserved", task] }),
  });

  const recoverMut = useMutation({
    mutationFn: (pickleId: number) => recoverModel(task, pickleId),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["bundles-list", task] });
      qc.invalidateQueries({ queryKey: ["bundle-preserved", task] });
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

      <Card
        title={`Bundles (${fmtNum(bundles.length)})`}
        sub={
          bestQ.data
            ? `Best ${bestQ.data.metric} = ${fmtNum(bestQ.data.value)} on ${bestQ.data.model_version}`
            : "No bundle carries a rankable score metric yet"
        }
        actions={
          <button
            className="btn"
            disabled={!bestVersion}
            title={bestVersion ? `Select ${bestVersion}` : "No bundle carries a rankable score metric"}
            onClick={() => bestVersion && setSelected(bestVersion)}
          >
            ★ Jump to best
          </button>
        }
      >
        {listQ.isLoading && task && <Loading />}
        {listQ.isError && <ErrorNote error={listQ.error} />}
        {bundles.length === 0 && !listQ.isLoading && (
          <p className="muted">No bundles yet — train a baseline above or run an experiment.</p>
        )}
        {bundles.length > 0 && (
          <Table<string>
            columns={["Model version", "", "Selected"]}
            rows={[...bundles].reverse()}
            keyOf={(v) => v}
            render={(v) => [
              <td key="v" className="mono">{v}</td>,
              <td key="b">{v === bestVersion && <Badge tone="ok">★ best</Badge>}</td>,
              <td key="s">
                <button className={`btn${selected === v ? " primary" : ""}`} onClick={() => setSelected(v)}>
                  Inspect
                </button>
              </td>,
            ]}
          />
        )}
      </Card>

      <PreservedCard
        task={task}
        snapshots={preservedQ.data?.snapshots ?? []}
        loading={preservedQ.isLoading}
        error={preservedQ.error}
        recovering={recoverMut.isPending}
        recovered={recoverMut.data}
        onRecover={(pickleId) => {
          if (window.confirm(`Recover snapshot #${pickleId} as a new bundle version?`)) recoverMut.mutate(pickleId);
        }}
        recoverError={recoverMut.isError ? recoverMut.error : null}
      />

      {selected && (
        <>
          <LineageRail task={task} modelVersion={selected} />
          <Card
            title={
              <>
                Bundle {selected} {selected === bestVersion && <Badge tone="ok">★ best</Badge>}
              </>
            }
            sub={detailQ.data?.created_at}
            actions={
              <>
                <button
                  className="btn"
                  disabled={preserveMut.isPending}
                  onClick={() => preserveMut.mutate(selected)}
                >
                  <Package size={15} /> {preserveMut.isPending ? "Preserving…" : "Preserve snapshot"}
                </button>
                <a className="btn" href={exportMlflowUrl(task, selected)}>
                  <Download size={15} /> Export MLflow
                </a>
                <a className="btn" href={bundleDownloadUrl(task, selected)}>
                  <Download size={15} /> Download bundle
                </a>
              </>
            }
          >
            {detailQ.isLoading && <Loading />}
            {detailQ.isError && <ErrorNote error={detailQ.error} />}
            {preserveMut.isError && <ErrorNote error={preserveMut.error} />}
            {preserveMut.isSuccess && (
              <div className="success-note">
                Preserved snapshot <code>#{preserveMut.data.pickle_id}</code> (model {preserveMut.data.model_id}).
              </div>
            )}
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

/**
 * Preserved (pickled) snapshots for the task, from mltrace's `saved` table.
 *
 * Recovery re-materializes the fitted estimator server-side and re-saves it as
 * a brand-new bundle version, so the list stays append-only — recovering never
 * overwrites the snapshot it came from.
 */
function PreservedCard({
  task,
  snapshots,
  loading,
  error,
  recovering,
  recovered,
  recoverError,
  onRecover,
}: {
  task: string;
  snapshots: PreservedSnapshot[];
  loading: boolean;
  error: unknown;
  recovering: boolean;
  recovered?: { pickle_id: number; model_type: string; smoke_score: number | null; model_version: string };
  recoverError: unknown;
  onRecover: (pickleId: number) => void;
}) {
  return (
    <Card
      title={`Preserved snapshots (${fmtNum(snapshots.length)})`}
      sub="Point-in-time pickles of fitted estimators kept inside the task's mltrace database."
    >
      {loading && <Loading />}
      {error ? <ErrorNote error={error} /> : null}
      {!task && <p className="muted">Pick a task to see its preserved snapshots.</p>}
      {task && !loading && snapshots.length === 0 && (
        <p className="muted">Nothing preserved yet — use “Preserve snapshot” on a bundle above.</p>
      )}
      {recoverError ? <ErrorNote error={recoverError} /> : null}
      {recovered && (
        <div className="success-note">
          Recovered #{recovered.pickle_id} ({recovered.model_type}) as{" "}
          <code>{recovered.model_version}</code>
          {recovered.smoke_score !== null && <> — train score {fmtNum(recovered.smoke_score)}</>}
        </div>
      )}
      {snapshots.length > 0 && (
        <Table<PreservedSnapshot>
          columns={["Pickle id", "Mlmodel id", "Preserved at", ""]}
          rows={snapshots}
          keyOf={(s) => s.pickle_id}
          render={(s) => [
            <td key="p" className="mono">#{s.pickle_id}</td>,
            <td key="m" className="mono">{s.model_id}</td>,
            <td key="t" className="mono">{s.init_date}</td>,
            <td key="a">
              <button className="btn" disabled={recovering} onClick={() => onRecover(s.pickle_id)}>
                {recovering ? "Recovering…" : "Recover as new version"}
              </button>
            </td>,
          ]}
        />
      )}
    </Card>
  );
}
