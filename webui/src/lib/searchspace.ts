/**
 * Model and (de)serialization for the visual search-space editor.
 *
 * Parity invariant: for any document accepted by the raw textarea,
 *   modelToSpace(spaceToModel(json)) ≡ json  (semantically; unknown parts verbatim)
 * "Unknown" means FQNs or parameters absent from the loaded catalog — they are
 * kept in `rawJson` / `customEntries` and re-emitted byte-for-byte.
 */

import type { CatalogComponent, ParamSpec } from "../api/client";

/** Reserved search-space key for the stacking directive (see client STACKING_COMPONENT). */
export const STACKING_KEY = "stacking";
/** Top-level key holding forbidden [parameter, value] rules. */
export const FORBIDDEN_KEY = "forbidden";

const NUMERIC_TYPES = new Set(["real", "integer"]);

export interface ParamModel {
  /** Catalog default (or imported) spec; absent for pinned scalars whose type we infer. */
  spec?: ParamSpec;
  /** false → parameter is pinned to `value` and emits a plain scalar. */
  free: boolean;
  /** Numeric range when free and spec.type ∈ {real, integer}. */
  low?: number;
  high?: number;
  /** Categorical: currently SELECTED values — exactly what gets serialized. */
  items: Array<string | null>;
  /** Categorical: full option list (catalog superset); chips render this and toggle membership in `items`. */
  options?: Array<string | null>;
  /** Pinned scalar value (used only when `free === false`). */
  value?: number | string | boolean | null;
  /** Preserved verbatim from imported JSON (A1: not editable in v1). */
  dependsOn?: Record<string, unknown[]>;
}

export interface CardModel {
  fqn: string;
  role: "estimator" | "transformer" | "sampler";
  task?: "classification" | "regression";
  name: string; // catalog display label, or the short class name when unknown
  description?: string;
  /** false → FQN not in the loaded catalog; the whole subtree is verbatim. */
  known: boolean;
  params: Record<string, ParamModel>;
  /** Verbatim JSON of the original params object (only when !known). */
  rawJson?: string;
  /** name → verbatim JSON for parameters outside the catalog (known cards only). */
  customEntries?: Record<string, string>;
}

export interface SpaceModel {
  cards: CardModel[]; // insertion order = key order of the JSON document
  stacking: unknown | null; // reserved directive object, or null when absent
  forbidden: Array<[string, unknown]> | null;
}

type Result<T> = T | { error: string };

function shortName(fqn: string): string {
  const parts = fqn.split(".");
  return parts[parts.length - 1];
}

function isSpecObject(v: unknown): v is ParamSpec {
  if (typeof v !== "object" || v === null) return false;
  const t = (v as ParamSpec).type;
  return t === "real" || t === "integer" || t === "categorical";
}

/** Parse a search-space JSON document into the editor model. */
export function spaceToModel(json: string, catalog: CatalogComponent[]): Result<SpaceModel> {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (e) {
    return { error: `Invalid search-space JSON: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { error: "Search space must be a JSON object mapping estimator names to parameter specs." };
  }
  const doc = { ...(raw as Record<string, unknown>) };
  const stacking = STACKING_KEY in doc ? (doc[STACKING_KEY] ?? null) : null;
  delete doc[STACKING_KEY];

  let forbidden: Array<[string, unknown]> | null = null;
  if (FORBIDDEN_KEY in doc && doc[FORBIDDEN_KEY] != null) {
    const fr = doc[FORBIDDEN_KEY];
    if (!Array.isArray(fr) || fr.some((p) => !Array.isArray(p) || p.length !== 2)) {
      return { error: '"forbidden" must be an array of [parameter, value] pairs.' };
    }
    forbidden = fr as Array<[string, unknown]>;
  }
  delete doc[FORBIDDEN_KEY];

  const byFqn = new Map(catalog.map((c) => [c.fqn, c]));
  const cards: CardModel[] = [];
  for (const [fqn, paramsRaw] of Object.entries(doc)) {
    if (typeof paramsRaw !== "object" || paramsRaw === null) continue; // defensive, matches parseSpace
    const cat = byFqn.get(fqn);
    const card: CardModel = {
      fqn,
      role: cat?.role ?? (fqn.includes("Classifier") || fqn.includes("Regressor") ? "estimator" : "transformer"),
      task: cat?.task,
      name: cat?.name ?? shortName(fqn),
      description: cat?.description,
      known: Boolean(cat),
      params: {},
    };
    if (!cat) {
      card.rawJson = JSON.stringify(paramsRaw); // verbatim block, read-only in the UI
      cards.push(card);
      continue;
    }
    const params: Record<string, ParamModel> = {};
    const customEntries: Record<string, string> = {};
    for (const [pname, val] of Object.entries(paramsRaw as Record<string, unknown>)) {
      const defaultSpec = cat.params[pname]; // catalog spec or undefined
      if (isSpecObject(val)) {
        const pm: ParamModel = {
          spec: defaultSpec ?? val,
          free: true,
          dependsOn: val.depends_on,
          items: [],
        };
        if (NUMERIC_TYPES.has(val.type)) {
          pm.items = [];
          pm.low = typeof val.low === "number" ? val.low : defaultSpec?.low;
          pm.high = typeof val.high === "number" ? val.high : defaultSpec?.high;
        } else {
          // Imported selection round-trips unchanged (parity); the catalog's full
          // item list is only the chip set offered for toggling. Categorical items
          // are plain values in every document this API accepts, hence the cast.
          pm.items = [...(val.items ?? [])] as Array<string | null>;
          if (defaultSpec && defaultSpec.type === "categorical") {
            pm.options = [...(defaultSpec.items ?? [])] as Array<string | null>;
          }
        }
        params[pname] = pm;
      } else if (typeof val === "number" || typeof val === "string" || typeof val === "boolean" || val === null) {
        // Pinned scalar: legal ParamValue. Render as pinned with the catalog spec for context.
        params[pname] = { free: false, value: val, spec: defaultSpec, items: [] };
      } else if (!cat.params.hasOwnProperty(pname)) {
        customEntries[pname] = JSON.stringify(val); // unknown parameter → verbatim
      } else {
        return { error: `Unrecognized entry for ${shortName(fqn)}.${pname}; edit it in Raw JSON.` };
      }
    }
    card.params = params;
    if (Object.keys(customEntries).length > 0) card.customEntries = customEntries;
    cards.push(card);
  }
  return { cards, stacking, forbidden };
}

/** Serialize the editor model back to a search-space JSON document string. */
export function modelToSpace(model: SpaceModel): Result<string> {
  const doc: Record<string, unknown> = {};
  for (const card of model.cards) {
    if (!card.known) {
      // Verbatim round-trip for components outside the catalog.
      try {
        doc[card.fqn] = JSON.parse(card.rawJson ?? "{}");
      } catch {
        return { error: `Stored custom block for ${shortName(card.fqn)} is not valid JSON.` };
      }
      continue;
    }
    const entry: Record<string, unknown> = {};
    for (const [pname, pm] of Object.entries(card.params)) {
      if (!pm.free) {
        entry[pname] = pm.value ?? null; // pinned scalar (number | string | boolean | null)
        continue;
      }
      const type = pm.spec?.type ?? "real";
      if (type === "categorical") {
        if (pm.items.length === 0) {
          return { error: `${shortName(card.fqn)}: ${pname} has no selected categories — pick at least one or pin a value.` };
        }
        entry[pname] = pm.dependsOn ? { type, items: pm.items, depends_on: pm.dependsOn } : { type, items: pm.items };
      } else if (NUMERIC_TYPES.has(type)) {
        if (typeof pm.low !== "number" || typeof pm.high !== "number" || !(pm.low <= pm.high)) {
          return { error: `${shortName(card.fqn)}: ${pname} range is invalid (need low ≤ high).` };
        }
        const lo = type === "integer" ? Math.round(pm.low) : pm.low;
        const hi = type === "integer" ? Math.round(pm.high) : pm.high;
        entry[pname] = pm.dependsOn ? { type, low: lo, high: hi, depends_on: pm.dependsOn } : { type, low: lo, high: hi };
      } else {
        return { error: `${shortName(card.fqn)}: unknown spec type ${type}.` };
      }
    }
    if (card.customEntries) {
      for (const [pname, raw] of Object.entries(card.customEntries)) entry[pname] = JSON.parse(raw);
    }
    doc[card.fqn] = entry;
  }
  if (model.stacking != null) doc[STACKING_KEY] = model.stacking;
  if (model.forbidden && model.forbidden.length > 0) doc[FORBIDDEN_KEY] = model.forbidden;
  return JSON.stringify(doc, null, 2);
}

/** Build a fresh card from a catalog row: every parameter free at its defaults. */
export function cardFromCatalog(row: CatalogComponent): CardModel {
  const params: Record<string, ParamModel> = {};
  for (const [pname, spec] of Object.entries(row.params)) {
    if (spec.type === "categorical") {
      params[pname] = { spec, free: true, items: [...(spec.items ?? [])] as Array<string | null> };
    } else {
      params[pname] = { spec, free: true, low: spec.low, high: spec.high, items: [] };
    }
  }
  return { fqn: row.fqn, role: row.role, task: row.task, name: row.name, description: row.description, known: true, params };
}

/** All parameter names across the model (options for forbidden-rule rows). */
export function allParamNames(model: SpaceModel): string[] {
  const names = new Set<string>();
  for (const card of model.cards) if (card.known) for (const pname of Object.keys(card.params)) names.add(pname);
  return [...names].sort();
}

/** Search-space JSON seeding an *untouched* editor from the catalog's preselected rows for a task. */
export function preselectedConfigJson(catalog: CatalogComponent[], task: "classification" | "regression"): string {
  const space: Record<string, unknown> = {};
  for (const row of catalog) if (row.preselected && row.task === task) space[row.fqn] = { ...row.params };
  return JSON.stringify(space, null, 2);
}
