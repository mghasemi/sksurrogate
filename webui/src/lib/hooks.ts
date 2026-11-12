import { useEffect, useMemo, useState } from "react";
import { getJob, health, jobLogs, subscribeToJob } from "../api/client";
import type { JobRecord, LogEntry } from "../api/client";

/**
 * Track a background job live: WebSocket stream for immediacy plus a light
 * poll of the persisted job record as a fallback (and to survive WS drops).
 */
export function useLiveJob(jobId: string | null): JobRecord | null {
  const [record, setRecord] = useState<JobRecord | null>(null);

  useEffect(() => {
    if (!jobId) {
      setRecord(null);
      return;
    }
    let unsub: (() => void) | undefined;
    try {
      unsub = subscribeToJob(jobId, setRecord);
    } catch {
      /* WebSocket unavailable — polling below still works */
    }

    getJob(jobId).then(setRecord).catch(() => {});
    const timer = window.setInterval(async () => {
      try {
        const r = await getJob(jobId);
        setRecord(r);
        if (r.status === "completed" || r.status === "failed") window.clearInterval(timer);
      } catch {
        /* transient — keep polling */
      }
    }, 1500);

    return () => {
      unsub?.();
      window.clearInterval(timer);
    };
  }, [jobId]);

  return record;
}

/** Poll /api/health every 10s. Returns null while unknown, true/false after. */
export function useHealth(): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () =>
      health()
        .then(() => alive && setOk(true))
        .catch(() => alive && setOk(false));
    check();
    const t = window.setInterval(check, 10000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);
  return ok;
}

/** Derived view over a job's captured log entries for the live activity feed. */
export interface JobActivity {
  /** Chronological (oldest first), already trimmed to the tail. */
  entries: LogEntry[];
  /** Entries that carry a search-trial index. */
  trialEvents: LogEntry[];
  /** Distinct trial indices observed so far. */
  trialsSeen: number;
  /** Trials that finished with a score (or an explicit completed status). */
  completedTrials: number;
  /** Highest objective value seen — the search maximizes the scoring metric. */
  bestScore: number | null;
  warnings: number;
  errors: number;
}

const EMPTY_ACTIVITY: JobActivity = {
  entries: [],
  trialEvents: [],
  trialsSeen: 0,
  completedTrials: 0,
  bestScore: null,
  warnings: 0,
  errors: 0,
};

function summarise(entries: LogEntry[]): JobActivity {
  const trialEvents = entries.filter((entry) => typeof entry.trial === "number");
  const indices = new Set<number>();
  const completed = new Set<number>();
  let bestScore: number | null = null;
  let warnings = 0;
  let errors = 0;
  for (const entry of entries) {
    if (typeof entry.trial === "number") indices.add(entry.trial);
    if (typeof entry.trial === "number" && entry.status === "completed") completed.add(entry.trial);
    if (typeof entry.score === "number" && Number.isFinite(entry.score)) {
      bestScore = bestScore === null ? entry.score : Math.max(bestScore, entry.score);
    }
    if (entry.level === "WARNING") warnings += 1;
    if (entry.level === "ERROR" || entry.level === "CRITICAL") errors += 1;
  }
  return {
    entries,
    trialEvents,
    trialsSeen: indices.size,
    completedTrials: completed.size,
    bestScore,
    warnings,
    errors,
  };
}

/**
 * Tail a job's captured entries for a live "what is being analysed" feed.
 *
 * Seeds from the REST tail, then polls every 1.5 s while the job is active and
 * takes one last snapshot once it terminates. `entries` is chronological so the
 * caller can render it top-down without reversing on every render.
 */
export function useJobActivity(jobId: string | null, active: boolean): JobActivity {
  const [entries, setEntries] = useState<LogEntry[]>([]);

  useEffect(() => {
    setEntries([]);
    if (!jobId) return;
    let alive = true;
    const load = async () => {
      try {
        const response = await jobLogs(jobId, { limit: 40 });
        if (alive) setEntries([...response.entries].reverse());
      } catch {
        /* transient — the next tick retries while the job is active */
      }
    };
    void load();
    if (!active) {
      return () => {
        alive = false;
      };
    }
    const timer = window.setInterval(load, 1500);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [jobId, active]);

  return useMemo(() => (entries.length ? summarise(entries) : EMPTY_ACTIVITY), [entries]);
}

/** Seconds elapsed between `start` and `end` (or now), ticking once a second. */
export function useElapsed(start: string | null | undefined, end?: string | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!start || end) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [start, end]);
  if (!start) return null;
  const begin = Date.parse(start);
  if (Number.isNaN(begin)) return null;
  const finish = end ? Date.parse(end) : now;
  if (Number.isNaN(finish)) return null;
  return Math.max(0, Math.round((finish - begin) / 1000));
}

/** `95s` → `1m 35s`; `3700s` → `1h 1m`. */
export function fmtElapsed(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes < 60) return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${(minutes % 60).toString().padStart(2, "0")}m`;
}
