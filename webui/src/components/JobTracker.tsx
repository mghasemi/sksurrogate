import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";

import { fmtElapsed, useElapsed, useJobActivity, useLiveJob } from "../lib/hooks";
import type { JobActivity } from "../lib/hooks";
import { StatusBadge, JsonView, ErrorNote, fmtNum } from "./ui";
import { resumeJob, renameJob, stopJob } from "../api/client";
import type { JobRecord, JobStatus } from "../api/client";

/** Compact one-line rendering of a trial's hyperparameters. */
function paramsLine(params: Record<string, unknown> | null | undefined): string {
  if (!params) return "";
  const entries = Object.entries(params);
  if (!entries.length) return "";
  return entries
    .slice(0, 4)
    .map(([name, value]) => `${name}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`)
    .join("  ");
}

function clock(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Statuses after which no worker owns the job any more. */
const TERMINAL_STATUSES: readonly JobStatus[] = [
  "completed",
  "failed",
  "paused",
  "cancelled",
  "interrupted",
];

/** Statuses a resume can start from (the API also requires a checkpoint). */
const RESUMABLE_STATUSES: readonly JobStatus[] = ["paused", "interrupted", "failed"];

/** A live job silent for longer than this is probably stuck. */
const HEARTBEAT_STALE_SECONDS = 30;

/**
 * Live tracker for a background job (analysis / experiment / retraining).
 *
 * Subscribes to the job's WebSocket status stream (with polling fallback) and
 * tails its captured log entries, so a running search shows what it is doing —
 * trial index, sampling phase, objective value and the parameters evaluated.
 *
 * Running jobs can be paused (checkpoint kept) or stopped; paused, interrupted
 * or failed-but-checkpointed jobs offer Resume, which submits a new job and
 * hands its id to ``onResume`` (or opens it on the Jobs page).
 */
export function JobTracker({
  jobId,
  onResult,
  onStatus,
  onResume,
  expectedTrials,
  label,
}: {
  jobId: string | null;
  onResult?: (result: Record<string, unknown>) => void;
  onStatus?: (job: JobRecord) => void;
  /** Called with the id of the job a Resume created. */
  onResume?: (newJobId: string) => void;
  /** Expected number of search trials, when known — drives the progress bar. */
  expectedTrials?: number;
  /** Optional heading; defaults to the job id prefix. */
  label?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draftLabel, setDraftLabel] = useState("");
  const [renamedLabel, setRenamedLabel] = useState<string | null>(null);
  const job = useLiveJob(jobId);
  const delivered = useRef<string | null>(null);
  const statusSink = useRef(onStatus);
  const navigate = useNavigate();

  useEffect(() => {
    statusSink.current = onStatus;
  }, [onStatus]);

  const terminal = Boolean(job && TERMINAL_STATUSES.includes(job.status));
  const active = Boolean(jobId) && !terminal;
  const activity = useJobActivity(jobId, active);
  const elapsed = useElapsed(job?.created_at, terminal ? job?.updated_at : null);
  const heartbeatAge = useElapsed(job?.heartbeat_at, null);

  const stopMut = useMutation({
    mutationFn: (mode: "pause" | "cancel") => stopJob(jobId as string, mode),
  });
  const resumeMut = useMutation({
    mutationFn: () => resumeJob(jobId as string),
    onSuccess: (created) => {
      if (onResume) onResume(created.job_id);
      else navigate(`/jobs?job_id=${encodeURIComponent(created.job_id)}`);
    },
  });
  const renameMut = useMutation({
    mutationFn: (value: string | null) => renameJob(jobId as string, value),
    onSuccess: (record) => {
      setRenamedLabel(record.display_name ?? null);
      setRenaming(false);
    },
  });

  const commitRename = () => {
    const value = draftLabel.trim();
    renameMut.mutate(value ? value : null);
  };

  useEffect(() => {
    if (job?.status === "completed" && job.result && delivered.current !== job.job_id) {
      delivered.current = job.job_id;
      onResult?.(job.result);
    }
  }, [job, onResult]);

  useEffect(() => {
    if (job) statusSink.current?.(job);
  }, [job]);

  if (!jobId) return null;

  const progress =
    expectedTrials && expectedTrials > 0
      ? Math.min(100, Math.round((activity.completedTrials / expectedTrials) * 100))
      : null;
  const stopRequested = Boolean(job?.stop_requested_at) && !job?.stop;
  const canPause = active && Boolean(job?.resumable) && !stopRequested;
  const canResume =
    !active && Boolean(job?.resumable) && Boolean(job && RESUMABLE_STATUSES.includes(job.status));
  const stalled = active && heartbeatAge !== null && heartbeatAge > HEARTBEAT_STALE_SECONDS;

  const requestPause = () => {
    if (window.confirm("Pause this job? Its checkpoint is kept so it can be resumed later.")) {
      stopMut.mutate("pause");
    }
  };
  const requestStop = () => {
    if (window.confirm("Stop this job? The run is abandoned (any checkpoint stays on disk).")) {
      stopMut.mutate("cancel");
    }
  };

  const displayName =
    label ?? renamedLabel ?? job?.display_name ?? job?.label ?? `Job ${jobId.slice(0, 8)}…`;

  return (
    <div className={`card job-card${active ? " active" : ""}`}>
      <div className="card-head">
        {active && <span className={`pulse${job?.status === "queued" ? " queued" : ""}`} />}
        {renaming ? (
          <input
            autoFocus
            value={draftLabel}
            onChange={(e) => setDraftLabel(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              else if (e.key === "Escape") setRenaming(false);
            }}
            placeholder="name this job, press Enter"
            spellCheck={false}
            style={{ maxWidth: 280 }}
          />
        ) : (
          <h3 title={`job id ${jobId}`}>{displayName}</h3>
        )}
        <span className="muted mono" style={{ fontSize: 11.5 }}>
          {jobId.slice(0, 8)}…
        </span>
        {job ? <StatusBadge status={job.status} /> : <span className="muted">connecting…</span>}
        {stopRequested && <span className="badge warn">stopping…</span>}
        <span className="muted mono" style={{ fontSize: 11.5 }}>
          {fmtElapsed(elapsed)}
        </span>
        <span className="spacer" />
        {renaming ? (
          <>
            <button className="btn primary" onClick={commitRename} disabled={renameMut.isPending}>
              Save name
            </button>
            <button className="btn" onClick={() => setRenaming(false)}>
              Cancel
            </button>
          </>
        ) : (
          <button
            className="btn"
            onClick={() => {
              setDraftLabel(job?.label ?? "");
              setRenaming(true);
            }}
          >
            Rename
          </button>
        )}
        {canPause && (
          <button className="btn" onClick={requestPause} disabled={stopMut.isPending}>
            Pause
          </button>
        )}
        {active && (
          <button className="btn danger" onClick={requestStop} disabled={stopMut.isPending}>
            Stop
          </button>
        )}
        {canResume && (
          <button className="btn primary" onClick={() => resumeMut.mutate()} disabled={resumeMut.isPending}>
            {resumeMut.isPending ? "Resuming…" : "Resume"}
          </button>
        )}
        <Link className="btn" to={`/logs?job=${jobId}`}>
          Logs
        </Link>
        {terminal && job && job.result !== null && (
          <button className="btn" onClick={() => setExpanded((e) => !e)}>
            {expanded ? "Hide result" : "Show result"}
          </button>
        )}
      </div>

      {active && <ActivitySummary activity={activity} progress={progress} expectedTrials={expectedTrials} />}

      {stopMut.isError && <ErrorNote error={stopMut.error} />}
      {resumeMut.isError && <ErrorNote error={resumeMut.error} />}
      {renameMut.isError && <ErrorNote error={renameMut.error} />}

      {stalled && (
        <p className="warn-note" style={{ marginTop: 8 }}>
          No worker heartbeat for {fmtElapsed(heartbeatAge)} — the job may be stalled even though it
          is still marked running.
        </p>
      )}

      {job?.status === "paused" && (
        <p className="muted" style={{ marginTop: 8 }}>
          Paused after {(activity.completedTrials || activity.trialsSeen) || 0} trial(s). Resume
          submits a new job that continues from the checkpoint saved with this one.
        </p>
      )}

      {job?.error && <ErrorNote error={new Error(job.error)} />}

      {terminal && job && job.result !== null && expanded && <JsonView data={job.result} />}

      {active && <ActivityFeed activity={activity} />}

      {active && !job?.error && activity.entries.length === 0 && (
        <p className="muted" style={{ marginTop: 8 }}>
          {job?.status === "queued"
            ? "Queued — waiting for a free worker slot."
            : "Waiting for the worker pool to report its first trial…"}
        </p>
      )}
    </div>
  );
}

/** Trial counter, best objective and (when known) a budget progress bar. */
function ActivitySummary({
  activity,
  progress,
  expectedTrials,
}: {
  activity: JobActivity;
  progress: number | null;
  expectedTrials?: number;
}) {
  const latest = activity.trialEvents[activity.trialEvents.length - 1];
  return (
    <div className="job-summary">
      <div className="job-summary-line">
        {activity.trialsSeen > 0 ? (
          <span>
            <strong>{activity.completedTrials || activity.trialsSeen}</strong>
            {progress !== null ? <> of ~{expectedTrials}</> : null} trials
            {latest?.phase ? <> · phase <code>{latest.phase}</code></> : null}
            {activity.bestScore !== null ? <> · best <strong>{fmtNum(activity.bestScore)}</strong></> : null}
          </span>
        ) : (
          <span className="muted">Starting the search — no trial reported yet.</span>
        )}
        {activity.warnings > 0 && <span className="badge warn">{activity.warnings} warnings</span>}
        {activity.errors > 0 && <span className="badge err">{activity.errors} errors</span>}
      </div>
      {progress !== null ? (
        <div
          className="progress-track"
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          title={`${activity.completedTrials} of ~${expectedTrials} trials`}
        >
          <div className="progress-fill" style={{ width: `${Math.max(progress, 2)}%` }} />
        </div>
      ) : (
        // No honest denominator (the search does not know its own trial count),
        // so move a stripe instead of inventing a percentage.
        <div className="progress-track indeterminate" role="progressbar" aria-label="search in progress">
          <div className="progress-fill" />
        </div>
      )}
    </div>
  );
}

/** The last few captured entries: a running commentary of what is being analysed. */
function ActivityFeed({ activity }: { activity: JobActivity }) {
  const latest = useMemo(() => activity.entries.slice(-6).reverse(), [activity.entries]);
  if (!latest.length) return null;
  return (
    <div className="feed">
      {latest.map((entry) => {
        const message = entry.message.replace(/\s+params=.*$/, "");
        const params = paramsLine(entry.params);
        return (
          <div key={`${entry.seq}-${entry.ts}`} className={`feed-row ${entry.level.toLowerCase()}`}>
            <span className="feed-ts mono">{clock(entry.ts)}</span>
            <span className="feed-msg">
              {message}
              {params && <span className="feed-params mono">{params}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Inline spinner used by pages that want a "submitting" cue next to a button. */
export function Spinner({ label }: { label: string }) {
  return (
    <span className="loading" style={{ padding: 0 }}>
      <span className="pulse" />
      {label}
    </span>
  );
}
