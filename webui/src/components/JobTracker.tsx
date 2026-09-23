import { useEffect, useRef, useState } from "react";
import { useLiveJob } from "../lib/hooks";
import { StatusBadge, JsonView, ErrorNote } from "./ui";

/**
 * Live tracker for a background job (sensitivity / experiment / retraining).
 * Subscribes to the WebSocket stream and polls as fallback.
 *
 * ``onResult`` is invoked once per job, as soon as the job completes with a
 * result payload — handy for rendering a visualization next to the raw JSON.
 */
export function JobTracker({
  jobId,
  onResult,
}: {
  jobId: string | null;
  onResult?: (result: Record<string, unknown>) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const job = useLiveJob(jobId);
  const delivered = useRef<string | null>(null);

  useEffect(() => {
    if (job?.status === "completed" && job.result && delivered.current !== job.job_id) {
      delivered.current = job.job_id;
      onResult?.(job.result);
    }
  }, [job, onResult]);

  if (!jobId) return null;

  const terminal = job && (job.status === "completed" || job.status === "failed");

  return (
    <div className="card">
      <div className="card-head">
        <h3>Job {jobId.slice(0, 8)}…</h3>
        {job ? <StatusBadge status={job.status} /> : <span className="muted">connecting…</span>}
        <span className="spacer" />
        {terminal && job.result !== null && (
          <button className="btn" onClick={() => setExpanded((e) => !e)}>
            {expanded ? "Hide result" : "Show result"}
          </button>
        )}
      </div>

      {job?.error && <ErrorNote error={new Error(job.error)} />}

      {terminal && job.result !== null && expanded && (
        <JsonView data={job.result} />
      )}

      {!terminal && !job?.error && (
        <p className="muted" style={{ marginTop: 8 }}>
          Waiting for the worker pool to finish this task…
        </p>
      )}
    </div>
  );
}
