import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { listActiveJobs } from "../api/client";

/** Human labels for the job kinds the control plane submits. */
const KIND_LABELS: Record<string, string> = {
  experiment: "experiment",
  "nested-cv": "nested CV",
  sensitivity: "feature analysis",
  optimize_pipeline: "pipeline search",
  retraining: "retraining",
  "project-synthesis": "synthesis",
};

function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind;
}

/**
 * Always-on activity badge for the top bar.
 *
 * Polls the lightweight `/api/jobs/active` endpoint (summaries only — never the
 * full job records with their results) and shows a pulsing pill while anything
 * is queued or running. Clicking it opens the Jobs page. The document title is
 * prefixed as well, so a background tab still advertises the activity.
 */
export function GlobalJobIndicator() {
  const query = useQuery({
    queryKey: ["jobs", "active"],
    queryFn: () => listActiveJobs(),
    refetchInterval: 4000,
    // The badge exists for background tabs; without this React Query pauses the
    // interval while the page is hidden and a finished job keeps its pill.
    refetchIntervalInBackground: true,
    staleTime: 2000,
  });
  const baseTitle = useRef<string>(typeof document === "undefined" ? "" : document.title);

  const jobs = query.data?.jobs ?? [];
  const count = jobs.length;
  const running = jobs.filter((job) => job.status === "running").length;
  const queued = count - running;

  useEffect(() => {
    if (typeof document === "undefined") return;
    const base = baseTitle.current;
    document.title = count > 0 ? `⏳ ${count} job${count === 1 ? "" : "s"} · ${base}` : base;
  }, [count]);

  if (count === 0) return null;

  const primary = jobs.find((job) => job.status === "running") ?? jobs[0];
  const text =
    count === 1
      ? `${kindLabel(primary.kind)} ${primary.status === "running" ? "running" : "queued"}`
      : `${count} jobs active`;
  const detail = jobs
    .map((job) => `${job.task_name} · ${kindLabel(job.kind)} · ${job.status}`)
    .join("\n");

  return (
    <Link className="topbar-jobs" to="/jobs" title={detail} aria-label={`${count} active jobs — open the Jobs page`}>
      <span className={`pulse${running === 0 ? " queued" : ""}`} />
      <span className="topbar-jobs-text">
        {text}
        {queued > 0 && running > 0 ? <span className="muted"> ({queued} queued)</span> : null}
      </span>
    </Link>
  );
}
