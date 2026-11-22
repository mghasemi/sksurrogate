/**
 * Visual editor for an experiment's search space.
 *
 * Controlled component: it renders a `SpaceModel` (see lib/searchspace.ts) and
 * emits `onChange(next)` on every interaction — no internal persistence. The
 * parent owns the single JSON source of truth, so the Raw sub-tab and submit
 * always see exactly what these controls produced.
 */

import { useMemo, useState } from "react";
import type { CatalogComponent } from "../api/client";
import { STACKING_COMPONENT } from "../api/client";
import type { CardModel, ParamModel, SpaceModel } from "../lib/searchspace";
import { allParamNames, cardFromCatalog } from "../lib/searchspace";
import { Badge } from "./ui";

export interface SearchSpaceEditorProps {
  model: SpaceModel;
  catalog: CatalogComponent[] | null; // null while the query is loading
  activeTask: "classification" | "regression";
  onChange(next: SpaceModel): void;
}

const ROLE_LABEL: Record<CardModel["role"], string> = {
  estimator: "estimator",
  transformer: "transformer",
  sampler: "resampler",
};

export default function SearchSpaceEditor(props: SearchSpaceEditorProps) {
  const { model, catalog, activeTask, onChange } = props;
  const [libFilter, setLibFilter] = useState("");

  const patchCard = (fqn: string, fn: (c: CardModel) => CardModel) =>
    onChange({ ...model, cards: model.cards.map((c) => (c.fqn === fqn ? fn(c) : c)) });

  const patchParam = (fqn: string, pname: string, patch: Partial<ParamModel>) =>
    patchCard(fqn, (c) => ({ ...c, params: { ...c.params, [pname]: { ...c.params[pname], ...patch } } }));

  const togglePin = (fqn: string, pname: string) =>
    // Pinning keeps low/high/items in the model so unpinning restores the range.
    patchParam(fqn, pname, { free: !model.cards.find((c) => c.fqn === fqn)?.params[pname]?.free });

  const addCard = (row: CatalogComponent) => onChange({ ...model, cards: [...model.cards, cardFromCatalog(row)] });

  const removeCard = (fqn: string) => onChange({ ...model, cards: model.cards.filter((c) => c.fqn !== fqn) });

  const moveCard = (index: number, dir: -1 | 1) => {
    const next = [...model.cards];
    const target = index + dir;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    onChange({ ...model, cards: next });
  };

  const setForbiddenRule = (i: number, patch: Partial<[string, unknown]>) => {
    if (!model.forbidden) return;
    const row: [string, unknown] = model.forbidden[i];
    const nextRow: [string, unknown] = [patch[0] !== undefined ? patch[0] : row[0], patch[1] !== undefined ? patch[1] : row[1]];
    onChange({ ...model, forbidden: model.forbidden.map((r, j) => (j === i ? nextRow : r)) });
  };

  const removeForbiddenRule = (i: number) => {
    if (!model.forbidden) return;
    onChange({ ...model, forbidden: model.forbidden.filter((_, j) => j !== i) });
  };

  const addForbiddenRule = () => {
    const names = allParamNames(model);
    const row: [string, unknown] = [names[0] ?? "parameter", null];
    onChange({ ...model, forbidden: model.forbidden ? [...model.forbidden, row] : [row] });
  };

  const toggleStacking = () => {
    if (model.stacking != null) onChange({ ...model, stacking: null });
    else onChange({ ...model, stacking: { ...STACKING_COMPONENT.stacking } });
  };

  // ---- library panel -------------------------------------------------------
  const q = libFilter.trim().toLowerCase();
  const sections: Array<{ title: string; rows: CatalogComponent[] }> = useMemo(() => {
    if (!catalog) return [];
    const match = (r: CatalogComponent) => !q || r.name.toLowerCase().includes(q) || r.fqn.toLowerCase().includes(q);
    const estimators = catalog.filter((r) => r.role === "estimator" && r.task === activeTask && match(r));
    const resamplers = activeTask === "classification" ? catalog.filter((r) => r.role === "sampler" && match(r)) : [];
    const transformers = catalog.filter((r) => r.role === "transformer" && match(r));
    return [
      { title: "Estimators", rows: estimators },
      ...(resamplers.length ? [{ title: "Resamplers (classification)", rows: resamplers }] : []),
      { title: "Transformers", rows: transformers },
    ];
  }, [catalog, q, activeTask]);

  const addedFqns = new Set(model.cards.map((c) => c.fqn));
  const nFree = model.cards.reduce(
    (acc, c) => acc + (c.known ? Object.values(c.params).filter((p) => p.free).length : 0),
    0,
  );
  const nPinned = model.cards.reduce(
    (acc, c) => acc + (c.known ? Object.values(c.params).filter((p) => !p.free).length : 0),
    0,
  );

  return (
    <div className="ss-editor">
      {/* ---------------- library ---------------- */}
      <aside className="ss-library">
        <input
          placeholder="Filter components…"
          value={libFilter}
          onChange={(e) => setLibFilter(e.target.value)}
          aria-label="Filter component library"
        />
        {catalog === null ? (
          <div className="hint">Loading component library…</div>
        ) : (
          sections.map((sec) =>
            sec.rows.length ? (
              <div key={sec.title}>
                <h4>{sec.title}</h4>
                {sec.rows.map((row) => {
                  const added = addedFqns.has(row.fqn);
                  return (
                    <div className="ss-lib-row" key={row.fqn}>
                      <span className="ss-lib-text">
                        <span className="ss-lib-name">
                          {row.name}
                          {row.preselected && <span title="Preselected for this problem type"> ★</span>}
                        </span>
                        {row.description && <span className="ss-lib-desc">{row.description}</span>}
                      </span>
                      <button
                        className="btn ss-add-btn"
                        disabled={added}
                        onClick={() => addCard(row)}
                        type="button"
                      >
                        {added ? "Added ✓" : "Add"}
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : null,
          )
        )}
      </aside>

      {/* ---------------- cards + rules ---------------- */}
      <div className="ss-main">
        {model.cards.length === 0 && (
          <div className="hint ss-empty">No components yet — add one from the library on the left.</div>
        )}

        {model.cards.map((card, i) => (
          <div className="card ss-card" key={card.fqn}>
            <div className="card-head">
              <h3>{card.name}</h3>
              <Badge tone="muted">{ROLE_LABEL[card.role]}</Badge>
              {card.task && <Badge tone="info">{card.task}</Badge>}
              {!card.known && <Badge tone="warn">unknown</Badge>}
              <span className="spacer" />
              <button className="btn" type="button" onClick={() => moveCard(i, -1)} disabled={i === 0} title="Move up">
                ↑
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => moveCard(i, 1)}
                disabled={i === model.cards.length - 1}
                title="Move down"
              >
                ↓
              </button>
              <button className="btn danger" type="button" onClick={() => removeCard(card.fqn)} title="Remove">
                ✕
              </button>
            </div>

            {card.known ? (
              Object.entries(card.params).map(([pname, pm]) => (
                <ParamControl key={pname} fqn={card.fqn} pname={pname} pm={pm} patchParam={patchParam} togglePin={togglePin} />
              ))
            ) : (
              <>
                <pre className="mono ss-verbatim">{card.rawJson}</pre>
                <div className="hint">Not in the component library — edit in Raw JSON.</div>
              </>
            )}

            {card.known && card.customEntries && Object.keys(card.customEntries).length > 0 && (
              <>
                <pre className="mono ss-verbatim">{JSON.stringify(card.customEntries, null, 2)}</pre>
                <div className="hint">Custom parameters — edit in Raw JSON.</div>
              </>
            )}
          </div>
        ))}

        {/* ---------------- forbidden rules ---------------- */}
        <div className="card ss-rules">
          <h3>Forbidden rules</h3>
          {model.forbidden && model.forbidden.length > 0 ? (
            <>
              {model.forbidden.map((rule, i) => (
                <div className="row" key={i}>
                  <input
                    value={String(rule[0])}
                    onChange={(e) => setForbiddenRule(i, [e.target.value, undefined] as unknown as Partial<[string, unknown]>)}
                    placeholder="parameter"
                    aria-label={`Forbidden rule ${i + 1}: parameter`}
                  />
                  <input
                    value={rule[1] == null ? "" : String(rule[1])}
                    onChange={(e) => setForbiddenRule(i, [undefined, e.target.value === "" ? null : e.target.value] as unknown as Partial<[string, unknown]>)}
                    placeholder="forbidden value (empty = any)"
                    aria-label={`Forbidden rule ${i + 1}: value`}
                  />
                  <button className="btn danger fixed" type="button" onClick={() => removeForbiddenRule(i)} title="Remove rule">
                    ✕
                  </button>
                </div>
              ))}
            </>
          ) : (
            <div className="hint ss-empty">No forbidden rules.</div>
          )}
          <button className="btn" type="button" onClick={addForbiddenRule}>
            Add forbidden rule
          </button>
        </div>

        {/* ---------------- stacking + summary ---------------- */}
        <div className="card ss-stacking">
          <label className="checkbox-row">
            <input type="checkbox" checked={model.stacking != null} onChange={toggleStacking} /> Stacking (out-of-fold
            StackingEstimator wrappers)
          </label>
          <div className="hint">A directive, not a pipeline step — configures the automatic stacking wrappers.</div>
        </div>

        <div className="ss-summary hint ss-empty">
          {model.cards.length} component(s) · {nFree} free parameter(s) · {nPinned} pinned. Maximum number of
          components per generated pipeline is set in Run settings → Max pipeline length (A2).
        </div>
      </div>
    </div>
  );
}

/** Parameter control for one known card: pin toggle + range/scalar/chips. */
function ParamControl({
  fqn,
  pname,
  pm,
  patchParam,
  togglePin,
}: {
  fqn: string;
  pname: string;
  pm: ParamModel;
  patchParam(fqn: string, pname: string, patch: Partial<ParamModel>): void;
  togglePin(fqn: string, pname: string): void;
}) {
  const type = pm.spec?.type ?? "real";
  const isNumeric = type === "real" || type === "integer";
  const invalidRange =
    pm.free &&
    isNumeric &&
    (typeof pm.low !== "number" || typeof pm.high !== "number" || !(pm.low <= pm.high));
  const emptySelection = pm.free && !isNumeric && pm.items.length === 0;

  return (
    <div className="ss-param">
      <span className="ss-param-name mono">{pname}</span>
      {pm.dependsOn && (
        <Badge tone="warn">
          conditional — depends_on preserved from JSON, edit in Raw mode
        </Badge>
      )}
      <span className="spacer" />
      <button className="chip ss-pin-toggle" type="button" onClick={() => togglePin(fqn, pname)} title="Toggle range / pinned value">
        {pm.free ? "range ⇄ pinned" : "pinned ⇄ range"}
      </button>

      {!pm.free ? (
        <input
          className={isNumeric ? undefined : "mono"}
          type={isNumeric ? "number" : "text"}
          step={type === "integer" ? 1 : undefined}
          value={pm.value == null ? "" : String(pm.value)}
          onChange={(e) => {
            const raw = e.target.value;
            if (!isNumeric) patchParam(fqn, pname, { value: raw });
            else patchParam(fqn, pname, { value: raw === "" ? null : Number(raw) });
          }}
          aria-label={`${pname} pinned value`}
        />
      ) : isNumeric ? (
        <span className="ss-range">
          <input
            type="number"
            step={type === "integer" ? 1 : undefined}
            min={pm.spec?.low}
            max={pm.spec?.high}
            value={typeof pm.low === "number" ? pm.low : ""}
            onChange={(e) => patchParam(fqn, pname, { low: e.target.value === "" ? undefined : Number(e.target.value) })}
            aria-label={`${pname} lower bound`}
          />
          <span className="hint">–</span>
          <input
            type="number"
            step={type === "integer" ? 1 : undefined}
            min={pm.spec?.low}
            max={pm.spec?.high}
            value={typeof pm.high === "number" ? pm.high : ""}
            onChange={(e) => patchParam(fqn, pname, { high: e.target.value === "" ? undefined : Number(e.target.value) })}
            aria-label={`${pname} upper bound`}
          />
        </span>
      ) : (
        <span className="chips">
          {(pm.options ?? pm.items).map((opt) => {
            const on = pm.items.includes(opt);
            return (
              <button
                key={String(opt)}
                type="button"
                className={`chip${on ? "" : " dropped"}`}
                onClick={() =>
                  patchParam(fqn, pname, { items: on ? pm.items.filter((x) => x !== opt) : [...pm.items, opt] })
                }
              >
                {String(opt)}
              </button>
            );
          })}
        </span>
      )}

      {(invalidRange || emptySelection) && (
        <div className="hint ss-error">
          {invalidRange ? "Invalid range — need a low ≤ high." : "Select at least one category, or pin a value."}
        </div>
      )}
    </div>
  );
}
