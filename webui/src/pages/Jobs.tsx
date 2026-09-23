import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { listJobs } from "../api/client";
import type { JobRecord } from "../api/client";
import { Card, ErrorNote, Loading, StatusBadge, Table } from "../components/ui";
import { JobTracker } from "../components/JobTracker";

const KINDS = ["sensitivity", "experiment", "retraining"] as const;
const STATUSES = ["queued", "running", "completed", "failed"] as const;

export default function JobsPage() {
  const [kindFilter, setKindFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

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
      completed: all.filter((j) => j.status === "completed").length,
      failed: all.filter((j) => j.status === "failed").length,
    };
  }, [jobsQ.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Jobs</h1>
        <span className="desc">All background jobs across tasks — sensitivity, experiments and retraining. Refreshes every 3 s.</span>
      </div>

      {jobsQ.isError && <ErrorNote error={jobsQ.error} />}
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
            {counts.running} active · {counts.completed} completed · {counts.failed} failed
          </span>
        </div>
      </Card>

      <Card title={`Jobs (${jobs.length})`} sub="Click a row to follow it live.">
        {jobs.length === 0 ? (
          <p className="muted">No jobs match the current filters.</p>
        ) : (
          <Table<JobRecord>
            columns={["Task", "Kind", "Status", "Created", "Updated"]}
            rows={jobs}
            keyOf={(j) => j.job_id}
            onRowClick={(j) => setSelectedId(j.job_id)}
            render={(j) => [
              <td key="t">{j.task_name}</td>,
              <td key="k">
                <span className="badge info">{j.kind}</span>
              </td>,
              <td key="s">
                <StatusBadge status={j.status} />
              </td>,
              <td key="c" className="mono">{new Date(j.created_at).toLocaleString()}</td>,
              <td key="u" className="mono">{new Date(j.updated_at).toLocaleString()}</td>,
            ]}
          />
        )}
      </Card>

      {selectedId && (
        <>
          <JobTracker jobId={selectedId} />
          <p className="muted">
            The tracker subscribes to the job’s WebSocket stream and falls back to polling; the full result JSON is
            expandable once the job reaches a terminal state.
          </p>
        </>
      )}
    </div>
  );
}
