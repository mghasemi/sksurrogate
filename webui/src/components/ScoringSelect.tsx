import type { CSSProperties } from "react";
import { useQuery } from "@tanstack/react-query";

import { getScoringOptions } from "../api/client";

/**
 * Dropdown of the standard scikit-learn evaluation metrics. The selected value
 * is sent as the AML/EOA `scoring` parameter, so it becomes the objective the
 * surrogate optimization maximizes (scikit-learn sign-flips `neg_*` error
 * metrics, keeping "higher is better" throughout).
 *
 * The wrapper is a plain ``.field`` so it participates in ``.row`` sizing like
 * its siblings; pass ``style`` to pin a width when a row mixes it with fixed
 * width fields.
 */
export function ScoringSelect({
  value,
  onChange,
  disabled,
  label = "Scoring",
  style,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  label?: string;
  style?: CSSProperties;
}) {
  const optionsQ = useQuery({
    queryKey: ["scoring-options"],
    queryFn: getScoringOptions,
    staleTime: Infinity,
  });

  const groups = optionsQ.data?.groups ?? [];
  const isKnown = groups.some((group) => group.options.some((option) => option.value === value));
  // Kept as a native tooltip rather than a hint line: an extra element would make
  // this field taller than its siblings and break row alignment.
  const title = optionsQ.isError
    ? "Could not load scoring options."
    : "The AML/EOA search maximizes this metric.";

  return (
    <div className="field" style={style}>
      <label>{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || optionsQ.isLoading}
        title={title}
      >
        {/* Preserve a custom/not-yet-loaded value instead of silently dropping it. */}
        {!isKnown && value && <option value={value}>{value}</option>}
        {groups.map((group) => (
          <optgroup key={group.label} label={group.label}>
            {group.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
}
