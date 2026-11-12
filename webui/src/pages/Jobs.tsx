import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";

import { listJobs, resumeJob, stopJob } from "../api/client";
import type { JobRecord } from "../api/client";
import { Card, ErrorNote, Loading, StatusBadge, Table } from "../components/ui";
import { JobTracker } from "../components/JobTracker";

const KINDS = ["sensitivity", "nested-cv", "experiment", "optimize_pipeline", "retraining", "project-synthesis"] as const;
const STATUSES = ["queued", "running", "paused", "interrupted", "completed", "failed", "cancelled"] as const;

export default function JobsPage() {
  const qc = useQueryClient();
  const [searchParams] = useSearchParams();
  const [kindFilter, setKindFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [selectedId, setSelectedId] = useState<string | null>(
    () => searchParams.get("job_id"),
  );
  const resumeMut = useMutation({
    mutationFn: resumeJob,
    onSuccess: (res) => {
      setSelectedId(res.job_id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const stopMut = useMutation({
    mutationFn: ({ jobId, mode }: { jobId: string; mode: "pause" | "cancel" }) => stopJob(jobId, mode),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: () => listJobs(),
    refetchInterval: 3000,
  });

  const jobs = useMemo(() => {
    const all = [...(jobsQ.data?.jobs ?? [])];
    all.sort((a, b) => b.created_at.localeCompare(a.created_at));
    return all.filter(
      (j) =>
        (kindFilter === "all" || j.kind === kindFilter) &&
        (statusFilter === "all" || j.status === statusFilter),
    );
  }, [jobsQ.data, kindFilter, statusFilter]);

  const counts = useMemo(() => {
    const all = jobsQ.data?.jobs ?? [];
    return {
      running: all.filter((j) => j.status === "running" || j.status === "queued").length,
      paused: all.filter((j) => j.status === "paused" || j.status === "interrupted").length,
      completed: all.filter((j) => j.status === "completed").length,
      failed: all.filter((j) => j.status === "failed").length,
    };
  }, [jobsQ.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Jobs</h1>
        <span className="desc">All background jobs across tasks — sensitivity, nested-CV, experiments, pipeline optimization and retraining. Refreshes every 3 s.</span>
      </div>

      {jobsQ.isError && <ErrorNote error={jobsQ.error} />}
      {resumeMut.isError && <ErrorNote error={resumeMut.error} />}
      {stopMut.isError && <ErrorNote error={stopMut.error} />}
      {jobsQ.isLoading && <Loading />}

      <Card title="Filters">
        <div className="row fixed">
          <div className="field fixed" style={{ flex: "0 1 180px", minWidth: 140 }}>
            <label>Kind</label>
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
              <option value="all">All kinds</option>
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <div className="field fixed" style={{ flex: "0 1 180px", minWidth: 140 }}>
            <label>Status</label>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="all">All statuses</option>
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <span className="muted" style={{ alignSelf: "flex-end", marginBottom: 8 }}>
            {counts.running} active · {counts.paused} paused/interrupted · {counts.completed} completed ·{" "}
            {counts.failed} failed
          </span>
        </div>
      </Card>

      <Card title={`Jobs (${jobs.length})`} sub="Click a row to follow it live.">
        {jobs.length === 0 ? (
          <p className="muted">No jobs match the current filters.</p>
        ) : (
          <Table<JobRecord>
            columns={["Name", "Task", "Kind", "Status", "Backend", "Created", "Updated", ""]}
            rows={jobs}
            keyOf={(j) => j.job_id}
            onRowClick={(j) => setSelectedId(j.job_id)}
            render={(j) => [
              <td key="n">
                {j.display_name ?? j.label ?? j.job_id.slice(0, 8) + "…"}
                {!j.label && <span className="muted" style={{ display: "block", fontSize: 11.5 }}>auto name — click the row to rename</span>}
              </td>,
              <td key="t">{j.task_name}</td>,
              <td key="k">
                <span className="badge info">{j.kind}</span>
              </td>,
              <td key="s">
                <StatusBadge status={j.status} />
              </td>,
              <td key="b">{j.backend ?? "local"}</td>,
              <td key="c" className="mono">{new Date(j.created_at).toLocaleString()}</td>,
              <td key="u" className="mono">{new Date(j.updated_at).toLocaleString()}</td>,
              <td key="a">
                <Link
                  className="btn"
                  to={`/logs?job=${j.job_id}`}
                  onClick={(event) => event.stopPropagation()}
                >
                  Logs
                </Link>
                {j.status === "running" && j.resumable && (
                  <button
                    className="btn"
                    disabled={stopMut.isPending}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (window.confirm("Pause this job and keep its checkpoint for a later resume?")) {
                        stopMut.mutate({ jobId: j.job_id, mode: "pause" });
                      }
                    }}
                  >
                    Pause
                  </button>
                )}
                {(j.status === "running" || j.status === "queued") && (
                  <button
                    className="btn danger"
                    disabled={stopMut.isPending}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (window.confirm("Stop this job? The run is abandoned.")) {
                        stopMut.mutate({ jobId: j.job_id, mode: "cancel" });
                      }
                    }}
                  >
                    Stop
                  </button>
                )}
                {j.resumable &&
                  j.status !== "running" &&
                  j.status !== "queued" &&
                  j.status !== "completed" && (
                    <button
                      className="btn"
                      disabled={resumeMut.isPending}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (window.confirm("Resume this job from its checkpoint?")) {
                          resumeMut.mutate(j.job_id);
                        }
                      }}
                    >
                      {resumeMut.isPending && resumeMut.variables === j.job_id ? "Resuming…" : "Resume"}
                    </button>
                  )}
              </td>,
            ]}
          />
        )}
      </Card>

      {selectedId && (
        <>
          <JobTracker jobId={selectedId} onResume={setSelectedId} />
          <p className="muted">
            The tracker subscribes to the job’s WebSocket stream and falls back to polling; the full result JSON is
            expandable once the job reaches a terminal state. A job is only shown as active while this API process
            actually owns it — records left behind by a crash are marked interrupted.
          </p>
        </>
      )}
    </div>
  );
}
