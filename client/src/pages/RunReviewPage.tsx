import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageChrome from "@/components/PageChrome";
import type { InspectionEvent, InspectionSession, RecipePlanItem, RecipeSegment } from "@/models/inspection";
import { artifactUrl, getInspectionRun, listInspectionEvents, listRecipePlan, listRecipeSegments } from "@/services/inspectionService";

function mediaSource(image?: InspectionEvent["image"] | null) {
  if (!image) return undefined;
  if (image.url) return artifactUrl(image.url);
  return image.base64 ? `data:${image.mime_type};base64,${image.base64}` : undefined;
}

function timestampLabel(value?: number | null) {
  if (value === null || value === undefined) return "Live source";
  return `${(value / 1000).toFixed(3)} s`;
}

function evidenceStatusLabel(event: InspectionEvent) {
  if (event.pipeline_mode === "model" && event.decision.status === "warning") return "QA review";
  return event.decision.status.replace("_", " ");
}

export default function RunReviewPage() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<InspectionSession | null>(null);
  const [events, setEvents] = useState<InspectionEvent[]>([]);
  const [segments, setSegments] = useState<RecipeSegment[]>([]);
  const [plan, setPlan] = useState<RecipePlanItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getInspectionRun(runId), listInspectionEvents(runId), listRecipeSegments(runId), listRecipePlan(runId)])
      .then(([runResult, eventResult, segmentResult, planResult]) => {
        setRun(runResult); setEvents(eventResult); setSegments(segmentResult);
        setPlan(planResult);
        setSelectedId(eventResult[0]?.event_id ?? null);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Could not load run"));
  }, [runId]);

  const selected = useMemo(() => events.find((event) => event.event_id === selectedId) ?? events[0], [events, selectedId]);

  return (
    <PageChrome>
      <main className="review-main">
        <section className="review-heading"><div><Link to="/history">← RUN HISTORY</Link><span className="kicker">FRAME EVIDENCE REVIEW</span><h1>{run?.run_name ?? "Loading run…"}</h1><p>{run ? `${run.source_mode.replace("_", " ")} · ${run.camera_id} · ${run.frames_processed} frames` : error}</p></div><div className="recipe-timeline"><strong>{plan.length ? `PLANNED VIAL ORDER · ${plan.length} VIALS` : `RECIPE CONTEXT TIMELINE · ${segments.length} SEGMENTS`}</strong>{plan.length ? plan.slice(0, 8).map((item) => <span key={item.plan_item_id}><i>V{item.position}</i>{item.recipe.current_recipe_id} · {item.recipe.expected_color_name ?? "color unset"} · {item.recipe.total_ml ?? 0} mL</span>) : segments.slice(0, 8).map((segment) => <span key={segment.segment_id}><i>{timestampLabel(segment.start_ms)}</i>{segment.recipe.current_recipe_id}</span>)}{(plan.length || segments.length) > 8 && <span>+{(plan.length || segments.length) - 8} more</span>}</div></section>
        <section className="review-scope-note"><strong>HOW TO READ THIS RUN</strong><span>Each frame shows the recipe context resolved from its source timestamp. The timeline count includes the run’s initial fallback context. “QA review” describes model, association, measurement, or reference qualification—not a recipe color mismatch. Recipe pass/fail is not implemented until qualified recipe references and the optional decision layer are added.</span></section>
        <div className="review-layout">
          <aside className="frame-browser panel"><div className="panel-heading compact"><div><span className="section-number">01</span><div><h2>Frames</h2><p>Newest first</p></div></div><span className="event-count">{events.length}</span></div><div className="frame-browser-list">{events.map((event) => <button className={event.event_id === selected?.event_id ? "active" : ""} key={event.event_id} type="button" onClick={() => setSelectedId(event.event_id)}><img src={mediaSource(event.image)} alt="" /><div><strong>FRAME {event.frame_number}</strong><span>{timestampLabel(event.source_timestamp_ms)}</span><small>Recipe {event.recipe.current_recipe_id} · {evidenceStatusLabel(event)}</small></div></button>)}</div></aside>
          <section className="frame-review panel">
            <div className="panel-heading compact"><div><span className="section-number">02</span><div><h2>Pipeline artifacts</h2><p>{selected?.original_filename ?? "Select a frame"}</p></div></div>{selected && <span className={`status-pill ${selected.decision.status}`}>{evidenceStatusLabel(selected)}</span>}</div>
            {!selected ? <div className="empty-history">This run has no processed frames.</div> : <>
              <div className="artifact-grid"><figure><figcaption>SOURCE FRAME</figcaption><img src={mediaSource(selected.image)} alt="Source frame" /></figure><figure><figcaption>RENDERED FRAME</figcaption>{selected.visuals?.rendered_frame ? <img src={mediaSource(selected.visuals.rendered_frame)} alt="Rendered frame" /> : <div>NOT PRODUCED BY CURRENT BRIDGE</div>}</figure><figure><figcaption>MASK</figcaption>{selected.visuals?.mask ? <img src={mediaSource(selected.visuals.mask)} alt="Mask" /> : <div>NOT PRODUCED BY CURRENT BRIDGE</div>}</figure></div>
              <div className="frame-metadata"><div><span>SOURCE TIME</span><strong>{timestampLabel(selected.source_timestamp_ms)}</strong></div><div><span>PLANNED RECIPE</span><strong>{selected.recipe.current_recipe_name}</strong><small>{selected.recipe.current_recipe_id} · vial {selected.recipe.vial_index} · expected {selected.recipe.expected_color_name ?? "color unset"}</small><small>T100 {selected.recipe.t100_ml ?? 0} · T200 {selected.recipe.t200_ml ?? 0} · T300 {selected.recipe.t300_ml ?? 0} · T400 {selected.recipe.t400_ml ?? 0} · {selected.recipe.total_ml ?? 0} mL</small></div><div><span>PIPELINE</span><strong>{selected.pipeline_mode.toUpperCase()}</strong><small>{selected.decision.model_state}</small></div><div><span>EVIDENCE / QA</span><strong>{selected.decision.headline}</strong><small>{selected.decision.recommended_action}</small></div></div>
            </>}
          </section>
        </div>
      </main>
    </PageChrome>
  );
}
