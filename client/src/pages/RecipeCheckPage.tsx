import { useEffect, useState } from "react";
import PageChrome from "@/components/PageChrome";
import "./RecipeCheckPage.css";

interface VialResult {
  vial: number;
  recipe: number;
  position: number;
  endpoint_frames: number;
  lab: number[] | null;
  swatch_rgb: number[] | null;
  dE_to_recipe: number | null;
  dE_to_previous: number | null;
  toward_previous_cos: number | null;
  diverges: boolean | null;
  reference_role: "compared" | "reference" | "not_measurable" | "no_reference";
  reason: string | null;
  crop: string;
}

interface RecipeInfo {
  recipe: number;
  t100_ml: string | null;
  t200_ml: string | null;
  t400_ml: string | null;
  total_ml: string;
  dispense_order: string;
  note: string | null;
  reference: string | null;
}

interface RecipeCheck {
  policy: { band_dE2000: number; direction_cos: number };
  summary: Record<string, number>;
  recipes: RecipeInfo[];
  vials: VialResult[];
}

const BASE = "/data/run5/";

function recipeLine(recipe: RecipeInfo) {
  const parts = [["T100 clear", recipe.t100_ml], ["T200 orange", recipe.t200_ml], ["T400 blue", recipe.t400_ml]]
    .filter(([, ml]) => ml).map(([tank, ml]) => `${tank} ${ml} mL`);
  return `${parts.join(" · ")} · order ${recipe.dispense_order.replaceAll(">", "→")}`;
}

function VialCard({ vial, band, cos }: { vial: VialResult; band: number; cos: number }) {
  const swatch = vial.swatch_rgb ? `rgb(${vial.swatch_rgb.join(",")})` : "transparent";
  let chip = { label: "NOT MEASURABLE", tone: "muted" };
  if (vial.reference_role === "reference") chip = { label: "REFERENCE", tone: "muted" };
  else if (vial.reference_role === "no_reference") chip = { label: "NO REFERENCE", tone: "muted" };
  else if (vial.reference_role === "compared") chip = vial.diverges ? { label: "DIVERGES", tone: "failure" } : { label: "WITHIN BAND", tone: "signal" };
  const toward = vial.diverges && (vial.toward_previous_cos ?? 0) >= cos;
  return (
    <figure className={`rc-vial rc-${chip.tone}`}>
      <img src={BASE + vial.crop} alt={`Run 5 vial ${vial.vial}, recipe ${vial.recipe}, end of fill`} loading="lazy" />
      <figcaption>
        <div className="rc-vial-head">
          <span className="rc-swatch" style={{ background: swatch }} aria-hidden="true" />
          <strong>Vial {vial.vial}</strong>
          <small>#{vial.position} of recipe</small>
        </div>
        <span className={`rc-chip rc-chip-${chip.tone}`}>{chip.label}</span>
        {vial.reference_role === "compared" && <p>ΔE {vial.dE_to_recipe?.toFixed(1)} to recipe {vial.recipe} (band {band})</p>}
        {vial.reference_role === "reference" && <p>Proxy color for recipe {vial.recipe}</p>}
        {vial.reference_role === "no_reference" && <p>Measured; recipe {vial.recipe} has no reference</p>}
        {vial.reference_role === "not_measurable" && <p>QA withheld color: {vial.reason}</p>}
        {toward && <p className="rc-toward">→ toward previous recipe (ΔE {vial.dE_to_previous?.toFixed(1)})</p>}
      </figcaption>
    </figure>
  );
}

export default function RecipeCheckPage() {
  const [data, setData] = useState<RecipeCheck | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(BASE + "run5.json")
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))))
      .then(setData)
      .catch(() => setError("Recipe check data is unavailable in this build."));
  }, []);

  const compared = data?.vials.filter((v) => v.reference_role === "compared") ?? [];
  const maxDe = Math.max(10, ...compared.map((v) => v.dE_to_recipe ?? 0));
  const s = data?.summary;

  return (
    <PageChrome>
      <main className="rc-main">
        <section className="rc-intro">
          <span className="kicker">RETROSPECTIVE EVIDENCE</span>
          <h1>Recipe check · Run 5 (organizer-labelled)</h1>
          <p className="rc-banner" role="note">
            Retrospective analysis of the organizers' labelled recording — not live, not a blind test. Frozen models and QA;
            endpoint rule chosen on Runs 1–4.
          </p>
        </section>

        {error && <p className="history-error">{error}</p>}
        {data && s && <>
          <section className="rc-stats" aria-label="Summary">
            <div className="rc-stat rc-failure"><strong>{s.first_vials_diverging} / {s.first_vials_comparable}</strong><span>first vials after a recipe change diverge</span><small>{s.first_vials_total - s.first_vials_comparable} of {s.first_vials_total} changes not comparable</small></div>
            <div className="rc-stat rc-signal"><strong>{s.second_vials_within_band} / {s.second_vials_comparable}</strong><span>second vials back within band</span><small>ΔE2000 ≤ {data.policy.band_dE2000}</small></div>
            <div className="rc-stat"><strong>{s.vials_with_endpoint} / {s.vials}</strong><span>vials with a measurable endpoint</span><small>{s.recipes_with_reference} / {s.recipes} recipes have a proxy reference</small></div>
          </section>

          <section className="rc-changes" aria-label="Recipe changes">
            <h2>Each recipe change, three vials of the new recipe</h2>
            {data.recipes.map((recipe) => {
              const vials = data.vials.filter((v) => v.recipe === recipe.recipe).sort((a, b) => a.vial - b.vial);
              const anyCompared = vials.some((v) => v.reference_role === "compared");
              return (
                <article className={`rc-row ${anyCompared ? "" : "rc-row-compact"}`} key={recipe.recipe}>
                  <header>
                    <strong>{recipe.recipe === 5 ? "Start → recipe 5" : `Recipe ${recipe.recipe - 1} → ${recipe.recipe}`}</strong>
                    <small>{recipeLine(recipe)}</small>
                    {!anyCompared && <small className="rc-gap">No comparison: {recipe.reference ? "first/second vials not measurable" : "no measurable reference vial"}</small>}
                  </header>
                  <div className="rc-vials">
                    {vials.map((vial) => <VialCard key={vial.vial} vial={vial} band={data.policy.band_dE2000} cos={data.policy.direction_cos} />)}
                  </div>
                </article>
              );
            })}
          </section>

          <section className="rc-chart" aria-label="Color difference by vial">
            <h2>ΔE2000 to intended recipe</h2>
            <div className="rc-bars">
              <div className="rc-band" style={{ bottom: `${(data.policy.band_dE2000 / maxDe) * 100}%` }}><span>band {data.policy.band_dE2000}</span></div>
              {compared.map((v) => (
                <div className="rc-bar-col" key={v.vial}>
                  <div className={`rc-bar rc-pos-${v.position}`} style={{ height: `${((v.dE_to_recipe ?? 0) / maxDe) * 100}%` }} title={`Vial ${v.vial}: ΔE ${v.dE_to_recipe}`}>
                    <span>{v.dE_to_recipe?.toFixed(1)}</span>
                  </div>
                  <small>{v.vial}</small>
                </div>
              ))}
            </div>
            <p className="rc-legend"><i className="rc-pos-1" /> 1st vial after change <i className="rc-pos-2" /> 2nd vial</p>
          </section>

          <section className="rc-notes">
            <details>
              <summary>How this was computed</summary>
              <ul>
                <li>Organizer video 5 (640×480, 30 fps) and the per-vial recipe labels: 10 recipes × 3 vials with labelled fill start/end.</li>
                <li>Released YOLO11n + temporal U-Net + unchanged measurement QA, a fresh tracker per vial. Temporal reference = frame at labelled fill start (unreviewed).</li>
                <li>Endpoint = median Lab of QA-valid frames in the last 6 s of the fill. This rule was chosen on Runs 1–4 (22/32 coverage, 0.49 spread) and applied to Run 5 once.</li>
                <li>Proxy reference = the 3rd vial of each recipe (2nd as fallback). Divergence = ΔE2000 &gt; {data.policy.band_dE2000}; "toward previous" = direction cosine ≥ {data.policy.direction_cos}.</li>
              </ul>
            </details>
            <details>
              <summary>Limits</summary>
              <ul>
                <li>Run 5 is diagnostic-only and the only labelled run; this is not a blind test.</li>
                <li>References are proxies from the same recording, not certified correct vials.</li>
                <li>No delivery records or independent per-vial outcomes: "diverges, consistent with carryover" is the strongest claim. Dosing or optical causes are not excluded.</li>
                <li>Half of the vials and recipe changes are not comparable because QA withheld the color.</li>
                <li>A first Run 5 pass with an after-fill window (15/30 endpoints) prompted the rule comparison; the rule itself was chosen on Runs 1–4.</li>
              </ul>
            </details>
          </section>
        </>}
      </main>
    </PageChrome>
  );
}
