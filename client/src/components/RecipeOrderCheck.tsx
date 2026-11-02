import { useEffect, useMemo, useState } from "react";
import type { RecipeContext } from "@/models/inspection";
import type { Lab } from "@/services/recipeCheck";
import { suggestRecipeOrder, type OrderPlan, type OrderSuggestion } from "@/services/lineService";

const SHIFTS: [string, number | null | undefined][] = [["Station default", undefined], ["Any order", null],
  ["Runs move ≤ 1", 1], ["Runs move ≤ 2", 2], ["Runs move ≤ 3", 3]];

interface Props {
  draft: RecipeContext[];
  /** Live re-plan: rows already filled (V1..V`decided`) and the last decided vial's measured colour. */
  decided: number;
  lineLab: Lab | null;
  /** New order of the rows from `from` on, as library recipe ids (one per vial). */
  onApply: (order: string[], from: number, enteredIds: string[]) => void;
}

/** Forecast the first vials after each changeover and suggest a run order and flushes (replay/schedule.py). */
export default function RecipeOrderCheck({ draft, decided, lineLab, onApply }: Props) {
  const [model, setModel] = useState<"lag" | "mix" | "">("");
  const [f, setF] = useState<string>("");
  const [shift, setShift] = useState(0);
  const [result, setResult] = useState<{ key: string; from: number; data: OrderSuggestion } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = useMemo(() => draft.map((r) => r.current_recipe_id).join("|"), [draft]);
  useEffect(() => { if (result && result.key !== key) setResult(null); }, [key, result]);

  async function check(from: number) {
    const rows = draft.slice(from);
    if (!rows.length) return;
    setBusy(true);
    setError(null);
    try {
      const data = await suggestRecipeOrder(rows, {
        ...(model ? { model } : {}), ...(f !== "" ? { f: Number(f) } : {}),
        ...(SHIFTS[shift][1] !== undefined ? { max_shift: SHIFTS[shift][1] } : {}),
        ...(from > 0 && lineLab ? { line_lab: lineLab } : {}),
      });
      setResult({ key, from, data });
      if (f === "") setF(String(data.f));
      if (!model) setModel(data.model);
    } catch (checkError) {
      setError(checkError instanceof Error ? checkError.message : "Order check failed");
    } finally {
      setBusy(false);
    }
  }

  const data = result?.data;
  const from = result?.from ?? 0;
  const same = data?.proposed && data.entered && data.proposed.order.join("|") === data.entered.order.join("|");
  const replan = decided > 0 && decided < draft.length && lineLab !== null;
  return (
    <div className="ln-order-check">
      <div className="ln-order-check-head">
        <div>
          <small>ORDER CHECK · BEFORE THE CHANGEOVERS</small>
          <p>Forecasts the first vials after each recipe change from what the line holds, and suggests an order and water flushes. Advice only; the order changes when you save it.</p>
        </div>
        <div className="ln-order-check-controls">
          <label>Line model
            <select value={model} onChange={(e) => setModel(e.target.value as "lag" | "mix" | "")}>
              <option value="">Station default</option>
              <option value="lag">Lag (previous dose)</option>
              <option value="mix">Mix (line contents)</option>
            </select>
          </label>
          <label>Residue f (assumed)
            <input type="number" min={0} max={0.9} step={0.05} value={f} placeholder="default"
              onChange={(e) => setF(e.target.value)} />
          </label>
          <label>Reorder
            <select value={shift} onChange={(e) => setShift(Number(e.target.value))}>
              {SHIFTS.map(([label], index) => <option key={label} value={index}>{label}</option>)}
            </select>
          </label>
          <button type="button" className="ln-primary" disabled={busy || !draft.length} onClick={() => void check(0)}>
            {busy ? "Checking…" : "Check order"}
          </button>
          {replan && (
            <button type="button" disabled={busy} onClick={() => void check(decided)}
              title="Plan the vials not yet filled from the last decided vial's measured colour">
              Re-plan from V{decided + 1}
            </button>
          )}
        </div>
      </div>
      {error && <p className="ln-tone-failure">{error}</p>}
      {data && data.missing.length > 0 && (
        <p className="ln-tone-warning">
          No colour reference for {data.missing.map((id) => data.names[id] ?? id).join(", ")}: set it with CREATE mode before planning.
        </p>
      )}
      {data?.entered && data.proposed && data.entered_no_flush && (
        <>
          <div className="ln-order-columns">
            <OrderColumn title={from ? `As entered, from V${from + 1}` : "As entered"} plan={data.entered} data={data} first={from} />
            <OrderColumn title="Suggested" plan={data.proposed} data={data} first={from} />
          </div>
          <div className="ln-order-check-foot">
            <p>
              Predicted scraps: <b>{data.entered_no_flush.scraps}</b> as entered without flushes ·{" "}
              <b>{data.entered.scraps}</b> + {data.entered.flushes} water as entered with flushes ·{" "}
              <b className="ln-tone-signal">{data.proposed.scraps}</b> + {data.proposed.flushes} water suggested.
              {" "}{data.model} model, f {data.f}{data.f === data.assumed_f ? " (station default, assumed)" : " (assumed)"}
              {data.max_shift != null ? `, runs move ≤ ${data.max_shift}` : ""}. A forecast, not a measurement.
            </p>
            <button type="button" className="ln-primary" disabled={!!same}
              onClick={() => onApply(data.proposed!.order, from, data.entered_ids)}>
              {same ? "Order already best" : "Use suggested order"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function OrderColumn({ title, plan, data, first }: { title: string; plan: OrderPlan; data: OrderSuggestion; first: number }) {
  const band = data.settings.band;
  let vial = first;
  return (
    <div className="ln-order-column">
      <h3>{title}</h3>
      {plan.steps.map((step, index) => {
        const worst = Math.max(...step.predicted_de);
        const tone = step.predicted_scraps ? "failure" : worst > band * 0.6 ? "warning" : "signal";
        const start = vial + 1;
        vial += step.vials;
        return (
          <div className="ln-order-step" key={`${index}-${step.recipe}`}>
            <i className={`ln-dot ln-dot-${tone}`} aria-hidden="true" />
            <span><b>{data.names[step.recipe] ?? step.recipe}</b> ×{step.vials}<small>V{start}{step.vials > 1 ? `–V${vial}` : ""}</small></span>
            <span className={`ln-tone-${tone}`}>
              {index || first ? `first ΔE ${step.predicted_de[0].toFixed(1)}` : "start"}
              {step.predicted_scraps ? ` · ${step.predicted_scraps} scrap${step.predicted_scraps > 1 ? "s" : ""}` : ""}
            </span>
            <em>{step.flushes ? `+${step.flushes} water first` : ""}</em>
          </div>
        );
      })}
      <p><b>{plan.scraps}</b> predicted scrap{plan.scraps === 1 ? "" : "s"} · {plan.flushes} water vial{plan.flushes === 1 ? "" : "s"}</p>
    </div>
  );
}
