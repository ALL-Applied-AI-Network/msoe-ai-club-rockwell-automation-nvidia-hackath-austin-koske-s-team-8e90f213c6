import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageChrome from "@/components/PageChrome";
import InvestigationPanel from "@/components/InvestigationPanel";
import type { InspectionEvent, InspectionSession, RecipeSegment } from "@/models/inspection";
import { acknowledgeInspectionEvent, artifactUrl, getInspectionRun, listInspectionEvents, listRecipeSegments } from "@/services/inspectionService";

function mediaSource(image?: InspectionEvent["image"] | null) {
  if (!image) return undefined;
  if (image.url) return artifactUrl(image.url);
  return image.base64 ? `data:${image.mime_type};base64,${image.base64}` : undefined;
}

function timestampLabel(value?: number | null) {
  if (value === null || value === undefined) return "Unknown source time";
  return `${(value / 1000).toFixed(3)} s`;
}

export default function RunReviewPage() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<InspectionSession | null>(null);
  const [events, setEvents] = useState<InspectionEvent[]>([]);
  const [segments, setSegments] = useState<RecipeSegment[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([getInspectionRun(runId), listInspectionEvents(runId, 100), listRecipeSegments(runId)])
      .then(([runResult, eventResult, segmentResult]) => {
        if (!active) return;
        setRun(runResult); setEvents(eventResult); setSegments(segmentResult);
        setHasMore(eventResult.length === 100);
        setSelectedId(eventResult[0]?.event_id ?? null);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Could not load run"));
    return () => { active = false; };
  }, [runId]);

  const selected = useMemo(() => events.find((event) => event.event_id === selectedId) ?? events[0], [events, selectedId]);

  const loadMore = async () => {
    try {
      const older = await listInspectionEvents(runId, 100, Math.min(...events.map((event) => event.frame_number)));
      setEvents((current) => [...current, ...older.filter((event) => !current.some((item) => item.event_id === event.event_id))]);
      setHasMore(older.length === 100);
    } catch { setError("Could not load earlier frames"); }
  };

  const saveResponse = async () => {
    if (!selected || !reviewer.trim()) return;
    const response = { reviewer, note, action: "reviewed" as const };
    try {
      await acknowledgeInspectionEvent(runId, selected.event_id, response);
      setEvents((current) => current.map((item) => item.event_id === selected.event_id
        ? { ...item, acknowledged_at: new Date().toISOString(), acknowledgement: response } : item));
    } catch { setError("Could not save the operator response"); }
  };

  return (
    <PageChrome>
      <main className="review-main">
        <section className="review-heading"><div><Link to="/history">← RUN HISTORY</Link><span className="kicker">FRAME EVIDENCE REVIEW</span><h1>{run?.run_name ?? "Loading run…"}</h1><p>{run ? `${run.source_mode.replace("_", " ")} · ${run.camera_id} · ${run.frames_processed} frames` : error}</p></div><div className="recipe-timeline"><strong>RECIPE TIMELINE</strong>{segments.slice(0, 8).map((segment) => <span key={segment.segment_id}><i>{timestampLabel(segment.start_ms)}</i>{segment.recipe.current_recipe_id}</span>)}{segments.length > 8 && <span>+{segments.length - 8} more</span>}</div></section>
        {error && <p className="history-error">{error}</p>}
        {hasMore && <button className="button quiet" onClick={() => void loadMore()}>LOAD EARLIER FRAMES</button>}
        <div className="review-layout">
          <aside className="frame-browser panel"><div className="panel-heading compact"><div><span className="section-number">01</span><div><h2>Frames</h2><p>Newest first</p></div></div><span className="event-count">{events.length}</span></div><div className="frame-browser-list">{events.map((event) => <button className={event.event_id === selected?.event_id ? "active" : ""} key={event.event_id} type="button" onClick={() => setSelectedId(event.event_id)}><img src={mediaSource(event.image)} alt="" /><div><strong>FRAME {event.frame_number}</strong><span>{timestampLabel(event.source_timestamp_ms)}</span><small>{event.recipe.current_recipe_id} · {event.decision.status.replace("_", " ")}</small></div></button>)}</div></aside>
          <section className="frame-review panel">
            <div className="panel-heading compact"><div><span className="section-number">02</span><div><h2>Pipeline artifacts</h2><p>{selected?.original_filename ?? "Select a frame"}</p></div></div>{selected && <span className={`status-pill ${selected.decision.status}`}>{selected.decision.status.replace("_", " ")}</span>}</div>
            {!selected ? <div className="empty-history">This run has no processed frames.</div> : <>
              <div className="artifact-grid"><figure><figcaption>SOURCE FRAME</figcaption><img src={mediaSource(selected.image)} alt="Source frame" /></figure><figure><figcaption>RENDERED FRAME</figcaption>{selected.visuals?.rendered_frame ? <img src={mediaSource(selected.visuals.rendered_frame)} alt="Rendered frame" /> : <div>NOT PRODUCED BY CURRENT ADAPTER</div>}</figure><figure><figcaption>MASK</figcaption>{selected.visuals?.mask ? <img src={mediaSource(selected.visuals.mask)} alt="Mask" /> : <div>NOT PRODUCED BY CURRENT ADAPTER</div>}</figure></div>
              <div className="frame-metadata"><div><span>SOURCE TIME</span><strong>{timestampLabel(selected.source_timestamp_ms)}</strong></div><div><span>RECIPE</span><strong>{selected.recipe.current_recipe_name}</strong><small>{selected.recipe.current_recipe_id}</small></div><div><span>PIPELINE</span><strong>{selected.pipeline_mode.toUpperCase()}</strong><small>{selected.decision.model_state}</small></div><div><span>DECISION</span><strong>{selected.decision.headline}</strong><small>{selected.decision.recommended_action}</small></div></div>
              <div className="response-panel"><p>{selected.decision.evidence?.blocked_reasons?.join(" · ")}</p><p>{selected.acknowledgement ? `Response: ${selected.acknowledgement.reviewer} · ${selected.acknowledgement.action} · ${selected.acknowledgement.note}` : "No attributed operator response"}</p><label>REVIEWER<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} /></label><label>NOTE<input value={note} onChange={(e) => setNote(e.target.value)} /></label><button className="button quiet" disabled={!reviewer.trim() || Boolean(selected.acknowledged_at)} onClick={() => void saveResponse()}>SAVE REVIEW RESPONSE</button><p>This records evidence review, not product approval.</p></div>
            </>}
          </section>
        </div>
        <InvestigationPanel key={runId} sessionId={runId} frameNumber={run?.frames_processed} />
      </main>
    </PageChrome>
  );
}
