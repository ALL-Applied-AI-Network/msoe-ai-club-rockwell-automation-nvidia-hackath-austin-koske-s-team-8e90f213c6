import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import PageChrome from "@/components/PageChrome";
import { listLineRecipes, listLineRuns, liveEvents, reviewEvents, frameUrl, type LineMeta, type LineVial, type TrayArrival } from "@/services/lineService";
import { VERDICT_TONE, labToCss } from "@/services/lineState";
import type { RecipeDefinition } from "@/services/recipeCheck";
import "./RecipeCheckPage.css";

// The line's ledger grouped by recipe change. By default the live cameras: it fills in as the running
// camera session decides each vial. Saved and recorded runs can be picked instead.
const LIVE = "live";

const FRAME_H = 170;

/** The fill camera frame at time t, cropped to the vial region. */
function VialFrame({ runId, session, t, roi, lab }:
  { runId: string; session: string; t: number; roi: [number, number, number, number]; lab: LineVial["endpoint_lab"] }) {
  const [width, setWidth] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const scale = FRAME_H / (roi[3] - roi[1]);
  if (failed) return <div className="rc-frame" style={{ background: labToCss(lab) }} />;
  return (
    <div className="rc-frame">
      <img src={`${frameUrl(runId, "fill", t)}&session=${session}`} alt="" loading="lazy" onError={() => setFailed(true)}
           onLoad={(e) => setWidth(e.currentTarget.naturalWidth)}
           style={{ width: width ? width * scale : undefined, left: 48 - ((roi[0] + roi[2]) / 2) * scale, top: -roi[1] * scale,
                    visibility: width ? "visible" : "hidden" }} />
    </div>
  );
}

function recipeLine(recipe: RecipeDefinition | undefined) {
  if (!recipe) return "";
  const parts = [["T100 clear", recipe.t100_ml], ["T200 orange", recipe.t200_ml], ["T400 blue", recipe.t400_ml]]
    .filter(([, ml]) => ml).map(([tank, ml]) => `${tank} ${ml} mL`);
  return `${parts.join(" · ")} · order ${recipe.order.replaceAll(">", "→")}`;
}

/** The vial a recipe's reference colour came from, if it is this run's ("Run 5 vial 6 endpoint (proxy)"). */
function referenceVial(recipe: RecipeDefinition | undefined, runTitle: string) {
  const from = recipe?.reference?.from ?? "";
  const match = /^(.*?)\s*vial (\d+)/.exec(from);
  return match && runTitle.toLowerCase().startsWith(match[1].toLowerCase()) ? Number(match[2]) : null;
}

function VialCard({ vial, position, band, isReference, runId, session, roi }:
  { vial: LineVial; position: number; band: number; isReference: boolean; runId: string; session: string; roi: [number, number, number, number] }) {
  const tone = VERDICT_TONE[vial.verdict];
  return (
    <figure className={`rc-vial rc-${tone}`}>
      <VialFrame runId={runId} session={session} t={vial.fill_end_s} roi={roi} lab={vial.endpoint_lab} />
      <figcaption>
        <div className="rc-vial-head">
          <span className="rc-swatch" style={{ background: labToCss(vial.endpoint_lab) }} aria-hidden="true" />
          <strong>Vial {vial.seq}</strong>
          <small>#{position} of recipe{isReference ? " · reference" : ""}</small>
        </div>
        <span className={`rc-chip rc-chip-${tone}`}>{vial.verdict.replaceAll("_", " ")}</span>
        {vial.de_intended != null && <p>ΔE {vial.de_intended.toFixed(1)} to recipe {vial.recipe_id} (band {band})</p>}
        {vial.pattern === "carryover" && (
          <p className="rc-toward">→ toward recipe {vial.previous_recipe_id} (ΔE {vial.de_previous?.toFixed(1)}): carryover</p>
        )}
        {vial.pattern === "deviation" && <p className="rc-toward">Not toward the previous recipe: wrong recipe or dose</p>}
        {vial.pattern === "no_reference" && <p>Recipe {vial.recipe_id} has no reference colour</p>}
        {vial.pattern === "unmeasured" && <p>QA withheld the colour</p>}
        <p><small>{vial.actions.replaceAll("+", " + ")}</small></p>
      </figcaption>
    </figure>
  );
}

export default function RecipeCheckPage() {
  const runs = useQuery({ queryKey: ["line-runs"], queryFn: listLineRuns, refetchInterval: 10_000 });
  const choices = (runs.data ?? []).filter((r) => r.id === LIVE || r.sources.includes("recorded"));
  const [picked, setPicked] = useState<string>(LIVE);
  const run = choices.find((r) => r.id === picked) ?? choices[0] ?? null;
  const isLive = run?.id === LIVE;
  const events = useQuery({
    queryKey: ["line-events", run?.id],
    queryFn: async () => (isLive ? liveEvents() : { run_id: run!.id, active: false, events: await reviewEvents(run!.id) }),
    enabled: !!run,
    refetchInterval: isLive ? 2000 : false,
  });
  const library = useQuery({ queryKey: ["line-recipes"], queryFn: listLineRecipes });

  const session = events.data?.run_id ?? "";
  const meta = events.data?.events.find((e): e is LineMeta => e.type === "meta");
  const vials = (events.data?.events ?? []).flatMap((e) => (e.type === "vial" ? [e.vial] : [])).sort((a, b) => a.seq - b.seq);
  const trays = (events.data?.events ?? []).filter((e): e is TrayArrival => e.type === "tray_arrival");
  const recipes = new Map((library.data?.recipes ?? []).map((r) => [String(r.id), r]));
  const band = meta?.line.band_de2000 ?? 5;

  // Consecutive vials of one recipe form a change (the first group starts the run).
  const changes: LineVial[][] = [];
  for (const v of vials) {
    const last = changes[changes.length - 1];
    if (last && last[0].recipe_id === v.recipe_id) last.push(v);
    else changes.push([v]);
  }
  const measured = (vs: LineVial[]) => vs.filter((v) => v.verdict !== "CANT_VERIFY");
  const firsts = changes.map((c) => c[0]);
  const seconds = changes.flatMap((c) => c.slice(1, 2));
  const outOfSpec = vials.filter((v) => v.verdict === "OUT_OF_SPEC");
  const escaped = trays.filter((t) => t.status === "escaped" && t.verdict === "OUT_OF_SPEC").length;
  const compared = vials.filter((v) => v.de_intended != null);
  const maxDe = Math.max(10, ...compared.map((v) => v.de_intended ?? 0));
  const error = runs.error || events.error || library.error;

  return (
    <PageChrome>
      <main className="rc-main">
        <section className="rc-intro">
          <span className="kicker">{isLive ? (events.data?.active ? "● LIVE" : "LIVE CAMERAS") : "RECORDED RUN"}</span>
          <h1>Recipe check · {isLive ? "live line" : run?.title ?? "no runs yet"}</h1>
          <label className="rc-pick">
            <span>Run</span>
            <select value={run?.id ?? ""} onChange={(e) => setPicked(e.target.value)}>
              {choices.map((r) => <option key={r.id} value={r.id}>{r.id === LIVE ? `${r.title} · live cameras` : r.title}</option>)}
            </select>
          </label>
          <p className="rc-banner" role="note">
            {!isLive ? "Every vial the line decided in this run, from its ledger: the same results the LINE page shows."
              : events.data?.active ? "Filling in as the cameras decide each vial (the LINE page's live session)."
              : events.data?.run_id ? `The last live session (${events.data.run_id}) has stopped. Start the cameras on the LINE page to continue.`
              : "No live session yet. Start the cameras on the LINE page; each vial appears here as it is decided."}
          </p>
        </section>

        {error && <p className="history-error">Line results are unavailable: {(error as Error).message}</p>}
        {events.data && meta && !vials.length && <p className="rc-banner">No vials decided yet.</p>}
        {meta && vials.length > 0 && <>
          <section className="rc-stats" aria-label="Summary">
            <div className="rc-stat rc-failure">
              <strong>{measured(firsts).filter((v) => v.verdict === "OUT_OF_SPEC").length} / {measured(firsts).length}</strong>
              <span>first vials of a recipe out of spec</span>
              <small>{firsts.length - measured(firsts).length} of {firsts.length} could not be checked</small>
            </div>
            <div className="rc-stat rc-signal">
              <strong>{measured(seconds).filter((v) => v.verdict === "IN_SPEC").length} / {measured(seconds).length}</strong>
              <span>second vials back in spec</span>
              <small>ΔE2000 ≤ {band}</small>
            </div>
            <div className="rc-stat rc-failure">
              <strong>{escaped}</strong>
              <span>out-of-spec vials that reached the Good tray</span>
              <small>{outOfSpec.length} out of spec · {vials.filter((v) => v.actions.includes("FLUSH")).length} FLUSH calls</small>
            </div>
          </section>

          <section className="rc-changes" aria-label="Recipe changes">
            <h2>Each recipe change and the vials of the new recipe</h2>
            {changes.map((group) => {
              const recipe = recipes.get(group[0].recipe_id);
              const anyCompared = group.some((v) => v.de_intended != null);
              return (
                <article className={`rc-row ${anyCompared ? "" : "rc-row-compact"}`} key={group[0].seq}>
                  <header>
                    <strong>{group[0].previous_recipe_id ? `Recipe ${group[0].previous_recipe_id} → ${group[0].recipe_id}` : `Start → recipe ${group[0].recipe_id}`}</strong>
                    <small>{recipeLine(recipe)}</small>
                    {!anyCompared && <small className="rc-gap">No comparison: {recipe?.reference ? "colour not measurable" : "no reference colour"}</small>}
                  </header>
                  <div className="rc-vials">
                    {group.map((v, i) => <VialCard key={v.seq} vial={v} position={i + 1} band={band} runId={run!.id} session={session} roi={meta.roi}
                                                   isReference={referenceVial(recipe, run!.title) === v.seq} />)}
                  </div>
                </article>
              );
            })}
          </section>

          <section className="rc-chart" aria-label="Color difference by vial">
            <h2>ΔE2000 to the ordered recipe</h2>
            <div className="rc-bars">
              <div className="rc-band" style={{ bottom: `${(band / maxDe) * 100}%` }}><span>band {band}</span></div>
              {compared.map((v) => (
                <div className="rc-bar-col" key={v.seq}>
                  <div className={`rc-bar rc-tone-${VERDICT_TONE[v.verdict]}`} style={{ height: `${((v.de_intended ?? 0) / maxDe) * 100}%` }} title={`Vial ${v.seq}: ΔE ${v.de_intended}`}>
                    <span>{v.de_intended?.toFixed(1)}</span>
                  </div>
                  <small>{v.seq}</small>
                </div>
              ))}
            </div>
            <p className="rc-legend"><i className="rc-tone-failure" /> out of spec <i className="rc-tone-signal" /> in spec</p>
          </section>

          <section className="rc-notes">
            <details>
              <summary>How this was computed</summary>
              <ul>
                <li>The line engine's decisions for this run (its stored model samples replayed with the recipe schedule), as on the LINE page. The photo is the fill camera at the end of each fill.</li>
                <li>YOLO11n vial detector and temporal U-Net liquid segmenter, trained on Isaac Sim renders only, then the unchanged measurement QA.</li>
                <li>Out of spec = ΔE2000 to the recipe's reference &gt; {band}; carryover = moved toward the previous recipe (direction cosine ≥ {meta.line.direction_cos}).</li>
                <li>Each reference is one accepted vial's endpoint colour (marked "reference" when it is from this run); {changes.filter((c) => !meta.references[c[0].recipe_id]).length ? `recipes ${changes.filter((c) => !meta.references[c[0].recipe_id]).map((c) => c[0].recipe_id).join(", ")} had none that could be measured.` : "every recipe has one."}</li>
              </ul>
            </details>
            <details>
              <summary>Limits</summary>
              <ul>
                <li>Run 5 is CSI's only labelled run and is used for evaluation only; its references are proxies from the same recording, not certified correct vials.</li>
                <li>No delivery records or independent per-vial outcomes: "consistent with carryover" is the strongest claim.</li>
                <li>Vials where QA withheld the colour, or the recipe has no reference, are held (CANT VERIFY), not passed.</li>
              </ul>
            </details>
          </section>
        </>}
      </main>
    </PageChrome>
  );
}
