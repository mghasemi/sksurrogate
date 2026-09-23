import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { GitBranch, Loader2 } from "lucide-react";

import { getLineage } from "../api/client";
import type { LineageRecord } from "../api/client";
import { fmtNum, fmtPct } from "./ui";

/**
 * The "traceability rail" (docs/ui-plan.md §3): a breadcrumb bar that walks
 * dataset → bundle → registry → monitoring for one model version, powered by
 * the single /api/lineage/{task}/{model_version} endpoint.
 */
export function LineageRail({ task, modelVersion }: { task: string; modelVersion: string | null }) {
  const q = useQuery<LineageRecord>({
    queryKey: ["lineage", task, modelVersion],
    queryFn: () => getLineage(task, modelVersion!),
    enabled: !!task && !!modelVersion,
    retry: false,
  });

  if (!task || !modelVersion) return null;

  if (q.isLoading) {
    return (
      <div className="rail">
        <Loader2 size={14} className="spin" />
        <span className="muted">Loading lineage for {modelVersion.slice(0, 8)}…</span>
      </div>
    );
  }

  if (q.isError || !q.data) return null; // rail is decorative — never block the page on it

  const rec = q.data;
  const fp = rec.dataset?.dataset_fingerprint ?? null;
  const state = rec.registry?.state ?? null;
  const mon = rec.monitoring;

  return (
    <div className="rail" title="Traceability rail — dataset → bundle → registry → monitoring">
      <GitBranch size={14} style={{ color: "var(--muted)" }} />

      <span className="crumb">
        Dataset{" "}
        {fp ? (
          <>
            (<Link to="/datasets" title="Open Datasets view">fp={fp.slice(0, 8)}…</Link>)
          </>
        ) : (
          "(unregistered)"
        )}
      </span>

      <span className="sep">→</span>

      <span className="crumb" title={rec.bundle.created_at}>
        Bundle{" "}
        <Link to="/bundles" title="Open Model Bundles view">
          {modelVersion.slice(0, 8)}…
        </Link>{" "}
        ({fmtNum(Object.keys(rec.bundle.metrics).length)} metrics)
      </span>

      <span className="sep">→</span>

      <span className="crumb" title={state ? `Current registry state: ${state}` : "Not in the model registry"}>
        Registry{" "}
        {state ? (
          <>
            (<Link to="/registry">{state}</Link>)
          </>
        ) : (
          "(unregistered)"
        )}
      </span>

      <span className="sep">→</span>

      <span className="crumb" title={mon ? `${fmtNum(mon.rows)} rows served` : "No inference calls logged yet"}>
        Monitoring{" "}
        {mon ? (
          <>
            (<Link to="/monitoring">{fmtNum(mon.requests)} req · err {fmtPct(mon.error_rate)}</Link>)
          </>
        ) : (
          "(no traffic)"
        )}
      </span>
    </div>
  );
}
