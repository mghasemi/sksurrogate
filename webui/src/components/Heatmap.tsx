import { useMemo, useRef, useState } from "react";

import type { HeatmapData } from "../api/client";
import { fmtNum } from "./ui";

/**
 * Client-side rendering of the data behind SKSurrogate's ``mltrack.heatmap``.
 *
 * - ``mode="diverging"`` for square matrices such as the Pearson correlation
 *   matrix (blue = negative, red = positive, neutral = 0).
 * - ``mode="sequential"`` for feature-by-weight tables, normalized per column.
 */

type Rgb = [number, number, number];

const NEG: Rgb = [64, 120, 224];
const POS: Rgb = [228, 92, 84];
const NEUTRAL: Rgb = [26, 34, 51];
const SEQ_LOW: Rgb = [22, 29, 45];
const SEQ_HIGH: Rgb = [79, 140, 255];

function mix(a: Rgb, b: Rgb, t: number): string {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Diverging color for a value in [-maxAbs, maxAbs]; 0 maps to the neutral panel tone. */
function divergingColor(value: number, maxAbs: number): string {
  const t = Math.max(-1, Math.min(1, value / (maxAbs || 1)));
  return t < 0 ? mix(NEUTRAL, NEG, -t) : mix(NEUTRAL, POS, t);
}

/** Sequential color for a normalized position in [0, 1]. */
function sequentialColor(t: number): string {
  return mix(SEQ_LOW, SEQ_HIGH, Math.max(0, Math.min(1, t)));
}

interface HoverState {
  row: number;
  col: number;
  x: number;
  y: number;
}

export function Heatmap({
  data,
  mode = "diverging",
}: {
  data: HeatmapData;
  mode?: "diverging" | "sequential";
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const nRows = data.labels.length;
  const nCols = data.columns.length;
  // A correlation matrix has identical row/column labels — rotate its top axis.
  const square = nRows > 0 && nRows === nCols;

  // Fit the matrix into the scroll box, but keep cells tappable/legible.
  const cell = Math.max(11, Math.min(28, Math.floor(620 / Math.max(nRows, nCols, 1)), Math.floor(480 / Math.max(nRows, 1))));
  const labelW = 152;
  const topH = square && mode === "diverging" ? cell * 2.2 + 10 : 24;

  // Per-column scale, used to normalize the sequential (weights) mode. Columns
  // use different units (correlation vs. variance), so normalizing per column
  // keeps every weight type readable; magnitude drives the color and the sign
  // stays available in the tooltip.
  const colScale = useMemo(
    () =>
      data.columns.map((_, c) => {
        let max = 0;
        for (let r = 0; r < nRows; r++) {
          const v = data.values[r]?.[c];
          if (v === null || v === undefined) continue;
          if (Math.abs(v) > max) max = Math.abs(v);
        }
        return max || 1;
      }),
    [data, nRows],
  );

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const row of data.values) {
      for (const v of row) if (v !== null && Math.abs(v) > m) m = Math.abs(v);
    }
    return Math.max(m, 1e-9);
  }, [data]);

  const colorFor = (r: number, c: number): string | undefined => {
    const v = data.values[r]?.[c];
    if (v === null || v === undefined) return undefined;
    if (mode === "diverging") return divergingColor(v, maxAbs);
    return sequentialColor(Math.abs(v) / colScale[c]);
  };

  const onMove = (e: React.MouseEvent, r: number, c: number) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ row: r, col: c, x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const legendStops = useMemo(() => {
    const stops = Array.from({ length: 13 }, (_, i) => i / 12);
    return stops.map((t) => (mode === "diverging" ? divergingColor(-maxAbs + 2 * maxAbs * t, maxAbs) : sequentialColor(t)));
  }, [mode, maxAbs]);

  if (!nRows || !nCols) {
    return <p className="muted">No matrix data to display.</p>;
  }

  const hoverValue = hover ? data.values[hover.row]?.[hover.col] : null;

  return (
    <div className="hm-wrap" ref={wrapRef}>
      <div className="hm-scroll">
        <svg width={labelW + nCols * cell} height={topH + nRows * cell} role="img" aria-label="heatmap">
          {data.columns.map((col, c) => (
            <text
              key={`c${c}`}
              x={labelW + c * cell + cell / 2}
              y={square && mode === "diverging" ? topH - 8 : 15}
              textAnchor={square && mode === "diverging" ? "start" : "middle"}
              className="hm-label"
              transform={square && mode === "diverging" ? `rotate(-42 ${labelW + c * cell + cell / 2} ${topH - 8})` : undefined}
            >
              {col.length > 18 ? col.slice(0, 17) + "…" : col}
            </text>
          ))}

          {data.labels.map((label, r) => (
            <text key={`r${r}`} x={labelW - 7} y={topH + r * cell + cell / 2 + 3.5} textAnchor="end" className="hm-label">
              {label.length > 20 ? label.slice(0, 19) + "…" : label}
            </text>
          ))}

          {data.values.map((row, r) =>
            row.map((_, c) => {
              const fill = colorFor(r, c);
              const active = hover && hover.row === r && hover.col === c;
              return (
                <rect
                  key={`${r}-${c}`}
                  x={labelW + c * cell}
                  y={topH + r * cell}
                  width={Math.max(cell - 1.5, 1)}
                  height={Math.max(cell - 1.5, 1)}
                  rx={2}
                  fill={fill ?? "var(--bg-soft)"}
                  stroke={active ? "var(--text)" : "none"}
                  strokeWidth={active ? 1 : 0}
                  onMouseMove={(e) => onMove(e, r, c)}
                  onMouseLeave={() => setHover(null)}
                />
              );
            }),
          )}
        </svg>
      </div>

      <div className="hm-legend">
        {mode === "diverging" ? (
          <>
            <span>−{fmtNum(maxAbs)}</span>
            <span className="hm-gradient" style={{ background: `linear-gradient(90deg, ${legendStops.join(",")})` }} />
            <span>+{fmtNum(maxAbs)}</span>
          </>
        ) : (
          <>
            <span>0</span>
            <span className="hm-gradient" style={{ background: `linear-gradient(90deg, ${legendStops.join(",")})` }} />
            <span>max |weight| per column</span>
          </>
        )}
      </div>

      {hover && (
        <div className="hm-tooltip" style={{ left: hover.x + 12, top: hover.y - 6 }}>
          <strong>
            {data.labels[hover.row]}
            {square ? ` ↕ ${data.columns[hover.col]}` : ""} · {data.columns[hover.col]}
          </strong>
          <span>{fmtNum(hoverValue)}</span>
        </div>
      )}
    </div>
  );
}
