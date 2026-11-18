import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { listServing } from "../api/client";

/** Long model ids are truncated in the pill; the tooltip carries the full value. */
function shortModel(modelVersion: string | null): string {
  if (!modelVersion) return "unknown";
  return modelVersion.length > 12 ? `${modelVersion.slice(0, 8)}…` : modelVersion;
}

/**
 * Always-on top-bar declaration that the system is serving a model.
 *
 * Polls `/api/serving` (active pins only — summaries, never the history) and
 * shows one pill while any task has a serving pin, so a deployment's model is
 * visible from every page rather than only where it was set. Clicking it opens
 * the Inference page, where the pin is set and can be exercised.
 *
 * It deliberately does *not* prefix `document.title`: the jobs indicator owns
 * that, and two writers would fight over it.
 */
export function ServingIndicator() {
  const query = useQuery({
    queryKey: ["serving", "active"],
    queryFn: () => listServing(),
    refetchInterval: 10000,
    // Visible to a background tab too — the whole point is "something is being
    // served right now" being observable without looking at the page.
    refetchIntervalInBackground: true,
    staleTime: 5000,
  });

  const pins = query.data?.serving ?? [];
  if (pins.length === 0) return null;

  const primary = pins[0];
  const stale = pins.some((pin) => pin.resolve_error);
  const text =
    pins.length === 1
      ? `Serving ${shortModel(primary.model_version)} · ${primary.task_name}`
      : `Serving ${pins.length} models`;
  const detail = pins
    .map((pin) => {
      const how = pin.resolve_mode === "alias" ? `alias ${pin.requested}` : "pinned version";
      const problem = pin.resolve_error ? ` — UNRESOLVED: ${pin.resolve_error}` : "";
      return `${pin.task_name} · ${pin.model_version} (${how})${problem}`;
    })
    .join("\n");

  return (
    <Link
      className={`topbar-serving${stale ? " warn" : ""}`}
      to="/inference"
      title={detail}
      aria-label={`${pins.length} model${pins.length === 1 ? "" : "s"} being served — open the Inference page`}
    >
      <span className="dot" />
      <span className="topbar-jobs-text">{text}</span>
      {stale && <span className="topbar-jobs-text muted">unresolved</span>}
    </Link>
  );
}
