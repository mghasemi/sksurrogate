import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import {
  LOG_LEVELS,
  listLogs,
  subscribeToJobLogs,
} from "../api/client";
import type { JobRecord, LogEntry, LogLevel } from "../api/client";
import { listJobs } from "../api/client";
import { Badge, Card, ErrorNote, Loading, JsonView, Table } from "../components/ui";

const LEVEL_TONE: Record<LogLevel, "muted" | "info" | "warn" | "err"> = {
  DEBUG: "muted",
  INFO: "info",
  WARNING: "warn",
  ERROR: "err",
  CRITICAL: "err",
};

const KINDS = [
  "experiment",
  "sensitivity",
  "nested-cv",
  "optimize_pipeline",
  "retraining",
  "project-synthesis",
] as const;

function levelMaskKey(levels: LogLevel[]): string {
  return levels.join(",");
}

export default function LogsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobId = searchParams.get("job") || "";
  const [levels, setLevels] = useState<LogLevel[]>([...LOG_LEVELS]);
  const [kind, setKind] = useState("all");
  const [text, setText] = useState("");
  const [limit, setLimit] = useState(200);
  const [live, setLive] = useState(true);
  const [selected, setSelected] = useState<LogEntry | null>(null);
  const [liveEntries, setLiveEntries] = useState<LogEntry[]>([]);

  const jobsQ = useQuery({ queryKey: ["jobs"], queryFn: () => listJobs() });
  const jobs: JobRecord[] = jobsQ.data?.jobs ?? [];
  const activeJob = jobs.find((job) => job.job_id === jobId) ?? null;
  const running = activeJob?.status === "running" || activeJob?.status === "queued";

  const maskKey = levelMaskKey(levels);
  const logsQ = useQuery({
    queryKey: ["logs", jobId, maskKey, kind, text, limit],
    queryFn: () =>
      listLogs({
        job_id: jobId || undefined,
        levels,
        kind: kind === "all" ? undefined : kind,
        q: text || undefined,
        limit,
      }),
    refetchInterval: live && (running || !jobId) ? 3000 : false,
  });

  // Live tail for a single selected job: subscribe to the incremental stream.
  useEffect(() => {
    setLiveEntries([]);
    if (!live || !jobId || !running) return;
    let unsubscribe: (() => void) | undefined;
    try {
      unsubscribe = subscribeToJobLogs(jobId, { levels }, (frame) => {
        if (!frame.entries.length) return;
        setLiveEntries((previous) => {
          const seen = new Set(previous.map((entry) => entry.seq));
          const fresh = frame.entries.filter((entry) => !seen.has(entry.seq));
          return [...fresh.reverse(), ...previous].slice(0, 2000);
        });
      });
    } catch {
      /* the polling query still updates the view */
    }
    return () => unsubscribe?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, live, running, maskKey]);

  const entries = useMemo(() => {
    const merged = new Map<string, LogEntry>();
    for (const entry of liveEntries) merged.set(`${entry.job_id}:${entry.seq}`, entry);
    for (const entry of logsQ.data?.entries ?? []) {
      merged.set(`${entry.job_id}:${entry.seq}`, entry);
    }
    const rows = [...merged.values()].filter((entry) => levels.includes(entry.level));
    rows.sort((a, b) => b.ts.localeCompare(a.ts) || b.seq - a.seq);
    return rows;
  }, [liveEntries, logsQ.data, levels]);

  const counts = logsQ.data?.counts;
  const severityTotal = (name: LogLevel) => counts?.[name] ?? 0;

  const toggleLevel = (name: LogLevel) => {
    setLevels((current) =>
      current.includes(name)
        ? current.filter((item) => item !== name)
        : [...LOG_LEVELS].filter((item) => current.includes(item) || item === name),
    );
  };

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Logs</h1>
        <span className="desc">
          Captured log records per job — core progress lines at INFO, surrogate diagnostics at
          DEBUG, degraded fits at WARNING, failures at ERROR — with severity, job and text
          filters. Entries persist as JSONL under the API home, so history survives a restart.
        </span>
      </div>

      {logsQ.isError && <ErrorNote error={logsQ.error} />}
      {jobsQ.isError && <ErrorNote error={jobsQ.error} />}

      <Card title="Filters">
        <div className="row fixed">
          <div className="field fixed" style={{ flex: "0 1 260px", minWidth: 200 }}>
            <label>Job</label>
            <select
              value={jobId}
              onChange={(event) => {
                const value = event.target.value;
                setSearchParams(value ? { job: value } : {});
                setSelected(null);
              }}
            >
              <option value="">All jobs</option>
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {job.task_name} · {job.kind} · {job.job_id.slice(0, 8)} ({job.status})
                </option>
              ))}
            </select>
          </div>
          <div className="field fixed" style={{ flex: "0 1 170px", minWidth: 140 }}>
            <label>Kind</label>
            <select value={kind} onChange={(event) => setKind(event.target.value)}>
              <option value="all">All kinds</option>
              {KINDS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ flex: "1 1 220px" }}>
            <label>Search message / logger / task</label>
            <input
              value={text}
              placeholder="e.g. trial 12, SKSurrogate.structsearch, credit-g"
              onChange={(event) => setText(event.target.value)}
            />
          </div>
          <div className="field fixed" style={{ flex: "0 1 120px", minWidth: 100 }}>
            <label>Rows</label>
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
              {[100, 200, 500, 1000, 2000].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="row fixed" style={{ marginTop: 10, alignItems: "center" }}>
          <span className="muted" style={{ marginRight: 6 }}>
            Severity
          </span>
          {LOG_LEVELS.map((name) => {
            const active = levels.includes(name);
            const count = severityTotal(name);
            return (
              <button
                key={name}
                type="button"
                className="btn"
                style={{ opacity: active ? 1 : 0.55 }}
                onClick={() => toggleLevel(name)}
                title={active ? "Click to hide this level" : "Click to include this level"}
              >
                {name}
                {counts ? <span className="muted"> · {count}</span> : null}
              </button>
            );
          })}
          <button
            type="button"
            className="btn"
            onClick={() => setLevels(["WARNING", "ERROR", "CRITICAL"])}
          >
            Warnings only
          </button>
          <button type="button" className="btn" onClick={() => setLevels([...LOG_LEVELS])}>
            All levels
          </button>
          <span className="spacer" />
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={live}
              onChange={(event) => setLive(event.target.checked)}
            />
            live tail
          </label>
        </div>
      </Card>

      <Card
        title={`Entries (${entries.length}${logsQ.data && logsQ.data.total > entries.length ? ` of ${logsQ.data.total}` : ""})`}
        sub={jobId ? undefined : "Newest first. Select a job to follow it live."}
      >
        {logsQ.isLoading && <Loading />}
        {entries.length === 0 && !logsQ.isLoading ? (
          <p className="muted">
            No entries match these filters. Run an experiment or job, then reload — records are
            captured while the job runs.
          </p>
        ) : (
          <Table<LogEntry>
            columns={["Time", "Level", "Job", "Logger", "Message"]}
            rows={entries}
            keyOf={(entry) => `${entry.job_id}:${entry.seq}`}
            onRowClick={(entry) => setSelected(entry)}
            render={(entry) => [
              <td key="t" className="mono">
                {new Date(entry.ts).toLocaleTimeString()}
              </td>,
              <td key="l">
                <Badge tone={LEVEL_TONE[entry.level]}>{entry.level}</Badge>
              </td>,
              <td key="j" className="mono">
                {entry.task_name ?? "—"} · {entry.kind ?? "—"}
              </td>,
              <td key="g" className="mono">
                {entry.logger}
              </td>,
              <td key="m">{entry.message}</td>,
            ]}
          />
        )}
      </Card>

      {selected && (
        <Card
          title={`Entry ${selected.level} · ${new Date(selected.ts).toLocaleString()}`}
          sub={`${selected.logger} · job ${selected.job_id.slice(0, 8)}… · seq ${selected.seq}`}
        >
          <p className="mono">{selected.message}</p>
          {selected.trial !== undefined && selected.trial !== null && (
            <p className="muted">
              trial {selected.trial} · phase {selected.phase ?? "unknown"} · status{" "}
              {selected.status ?? "—"}
            </p>
          )}
          {selected.detail && (
            <pre className="mono" style={{ whiteSpace: "pre-wrap", maxHeight: 320, overflow: "auto" }}>
              {selected.detail}
            </pre>
          )}
          <JsonView data={selected} />
        </Card>
      )}
    </div>
  );
}
