import type { ReactNode } from "react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";

export function Card({
  title,
  sub,
  actions,
  children,
}: {
  title?: string;
  sub?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h3>{title}</h3>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {actions && (
            <>
              <span className="spacer" />
              {actions}
            </>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

/** Horizontal tab bar; render the matching panel yourself next to it. */
export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: Array<{ id: string; label: ReactNode; badge?: ReactNode }>;
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          aria-selected={t.id === active}
          className={`tab${t.id === active ? " active" : ""}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
          {t.badge !== undefined && <span className="tab-badge">{t.badge}</span>}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Badge({ tone = "muted", children }: { tone?: "ok" | "warn" | "err" | "info" | "muted"; children: ReactNode }) {
  return <span className={`badge ${tone === "muted" ? "" : tone}`}>{children}</span>;
}

export function statusTone(status: string): "ok" | "warn" | "err" | "info" | "muted" {
  switch (status) {
    case "completed":
    case "production":
    case "passed":
      return "ok";
    case "running":
    case "staging":
    case "validated":
      return "info";
    case "queued":
    case "candidate":
      return "warn";
    case "failed":
    case "archived":
      return "err";
    default:
      return "muted";
  }
}

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={statusTone(status)}>{status}</Badge>;
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <Inbox size={26} />
      <div>
        <strong>{title}</strong>
        {hint && <div className="muted">{hint}</div>}
      </div>
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="error-note">
      <AlertTriangle size={15} />
      <span>{msg}</span>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="loading">
      <Loader2 size={16} className="spin" />
      {label}
    </div>
  );
}

/** Pretty-printed JSON block. */
export function JsonView({ data }: { data: unknown }) {
  return (
    <pre className="json">{JSON.stringify(data, null, 2)}</pre>
  );
}

/** Generic table with a header row and rows rendered by `render`. */
export function Table<T>({
  columns,
  rows,
  render,
  keyOf,
  onRowClick,
}: {
  columns: string[];
  rows: T[];
  render: (row: T) => ReactNode[];
  keyOf: (row: T, i: number) => string | number;
  onRowClick?: (row: T) => void;
}) {
  if (!rows.length) return <EmptyState title="Nothing here yet" />;
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={keyOf(r, i)}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
              style={onRowClick ? { cursor: "pointer" } : undefined}
            >
              {render(r)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function fmtNum(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(2)}%`;
}
