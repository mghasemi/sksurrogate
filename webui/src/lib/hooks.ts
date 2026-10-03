import { useEffect, useState } from "react";
import { getJob, health, subscribeToJob } from "../api/client";
import type { JobRecord } from "../api/client";

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
