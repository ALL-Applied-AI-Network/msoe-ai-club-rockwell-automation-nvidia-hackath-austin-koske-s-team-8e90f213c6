import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageChrome from "@/components/PageChrome";
import type { RecipeContext } from "@/models/inspection";
import { createInspectionSession, listRecipePlan, replaceRecipePlan } from "@/services/inspectionService";
import { deltaE2000, loadRecipeLibrary, recipeSummary, type RecipeDefinition } from "@/services/recipeCheck";
import { Link } from "react-router-dom";
import {
  listLineOrders, listLineRuns, liveSocket, previewUrl, reviewEvents, videoUrl,
  type LineAction, type LineEvent, type LineSource, type LineVial, type LiveMessage, type TrayArrival,
  createLineRecipe, explainVial, listLineRecipes, setLineCapture, type VialExplanation,
} from "@/services/lineService";
import { ACTION_TONE, VERDICT_TONE, clock, labToCss, phase, viewAt, type LineView } from "@/services/lineState";
import "./LinePage.css";

type Mode = "live" | "review";
type Status = "idle" | "connecting" | "live" | "ended" | "error";
const SPEEDS = [1, 2, 4, 8, 16];
const BANNER_S = 4;

interface ServerClock { t: number; at: number; speed: number; paused: boolean }

function recipeContext(recipe: RecipeDefinition, vialIndex: number): RecipeContext {
  const amounts = [recipe.t100_ml, recipe.t200_ml, recipe.t400_ml].map((value) => value ?? 0);
  return {
    current_recipe_id: recipe.id,
    current_recipe_name: recipe.name,
    previous_recipe_id: null,
    vial_index: vialIndex,
    is_first_vial_after_change: false,
    expected_color_name: recipe.note ?? null,
    expected_color_hex: null,
    t100_ml: amounts[0],
    t200_ml: amounts[1],
    t300_ml: 0,
    t400_ml: amounts[2],
    total_ml: amounts.reduce((sum, value) => sum + value, 0),
    dispense_order: recipe.order,
    color_purpose: "Recipe-library operator intent",
  };
}

function normalizePlan(recipes: RecipeContext[]): RecipeContext[] {
  return recipes.map((recipe, index) => {
    const previous = index > 0 ? recipes[index - 1].current_recipe_id : null;
    return {
      ...recipe,
      vial_index: index + 1,
      previous_recipe_id: previous,
      is_first_vial_after_change: previous !== recipe.current_recipe_id,
      total_ml: [recipe.t100_ml, recipe.t200_ml, recipe.t300_ml, recipe.t400_ml]
        .reduce<number>((sum, value) => sum + (value ?? 0), 0),
    };
  });
}

export default function LinePage() {
  const queryClient = useQueryClient();
  const runs = useQuery({ queryKey: ["line-runs"], queryFn: listLineRuns });
  const [runId, setRunId] = useState<string | null>(null);
  // Recipe order: an inspection run's recipe plan (copied from CSI's machine), or "" for the run's file.
  const orders = useQuery({ queryKey: ["line-orders"], queryFn: listLineOrders, refetchInterval: 5000 });
  const [order, setOrder] = useState<string | null>(null);
  useEffect(() => { if (order === null && orders.data) setOrder(orders.data[0]?.session_id ?? ""); }, [order, orders.data]);
  // The station's library with recipes created at the line (server), or the shipped file without a station.
  const recipeLibrary = useQuery({
    queryKey: ["recipe-library"],
    queryFn: async () => {
      try { return await listLineRecipes(); } catch { return { recipes: (await loadRecipeLibrary()).recipes, capture: null }; }
    },
    refetchInterval: (query) => (query.state.data?.capture ? 2000 : false),
  });
  const capture = recipeLibrary.data?.capture ?? null;
  const [newRecipe, setNewRecipe] = useState<{ name: string; t100_ml: number; t200_ml: number; t400_ml: number } | null>(null);
  const selectedPlan = useQuery({
    queryKey: ["line-order-plan", order],
    queryFn: () => listRecipePlan(order!),
    enabled: Boolean(order),
    staleTime: Infinity,
  });
  const [recipeDraft, setRecipeDraft] = useState<RecipeContext[]>([]);
  const [draftRecipeId, setDraftRecipeId] = useState("5");
  const [draftVials, setDraftVials] = useState(1);
  const [draftName, setDraftName] = useState("CSI live recipe order");
  const [newRecipeOrder, setNewRecipeOrder] = useState(false);
  const [recipeSaving, setRecipeSaving] = useState(false);
  const [recipeMessage, setRecipeMessage] = useState<string | null>(null);
  useEffect(() => {
    if (newRecipeOrder || !order || !selectedPlan.data) return;
    setRecipeDraft(normalizePlan(selectedPlan.data.map((item) => item.recipe)));
  }, [newRecipeOrder, order, selectedPlan.data]);
  useEffect(() => {
    const first = recipeLibrary.data?.recipes[0];
    if (first && !recipeLibrary.data?.recipes.some((recipe) => recipe.id === draftRecipeId)) setDraftRecipeId(first.id);
  }, [draftRecipeId, recipeLibrary.data]);
  const [mode, setMode] = useState<Mode>("live");
  // The station's live cameras are the default; a run without them falls back to its first source.
  const [source, setSource] = useState<LineSource>("camera");
  // Live cameras: "" = the station's real cameras; a run id = that run's videos played through the camera input.
  const [simulate, setSimulate] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [cameraStats, setCameraStats] = useState<Record<string, number> | null>(null);
  const [speed, setSpeed] = useState(8);
  const [startMin, setStartMin] = useState(0);
  const [paused, setPaused] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lag, setLag] = useState(0);
  const [t, setT] = useState(0);
  const [eventCount, setEventCount] = useState(0);
  const events = useRef<LineEvent[]>([]);
  const socket = useRef<WebSocket | null>(null);
  const serverClock = useRef<ServerClock | null>(null);
  const fill = useRef<HTMLVideoElement>(null);
  const tray = useRef<HTMLVideoElement>(null);

  const run = runs.data?.find((r) => r.id === runId) ?? runs.data?.find((r) => r.sources.includes("camera"))
    ?? runs.data?.[0] ?? null;
  useEffect(() => { if (run && !runId) setRunId(run.id); }, [run, runId]);
  useEffect(() => { if (run && !run.sources.includes(source)) setSource(run.sources[0] ?? "recorded"); }, [run, source]);

  const stop = useCallback(() => {
    socket.current?.close();
    socket.current = null;
    serverClock.current = null;
    setPreview(null);
  }, []);
  const camera = source === "camera";
  useEffect(() => stop, [stop]);

  const review = useQuery({
    queryKey: ["line-review", run?.id, order],
    queryFn: () => reviewEvents(run!.id, order),
    enabled: mode === "review" && !!run?.sources.includes("recorded"),
    staleTime: Infinity,
  });
  useEffect(() => {
    if (mode !== "review") return;
    stop();
    events.current = review.data ?? [];
    setEventCount(events.current.length);
    setStatus(review.data ? "ended" : "idle");
  }, [mode, review.data, stop]);

  function startLive() {
    if (!run) return;
    stop();
    events.current = [];
    setEventCount(0);
    setError(null);
    setPaused(false);
    setSaved(null);
    setCameraStats(null);
    setStatus("connecting");
    const start = source === "recorded" || (camera && simulate) ? startMin * 60 : 0;
    const ws = liveSocket(run.id, source, camera ? 1 : speed, start, order, camera ? simulate || null : null);
    socket.current = ws;
    ws.onmessage = (message) => {
      const data = JSON.parse(message.data) as LiveMessage;
      if (data.type === "meta") events.current = [data];
      else if (data.type === "events") {
        events.current = events.current.concat(data.events);
        for (const event of data.events) {
          const meta = events.current[0];
          if (event.type === "reference") {
            if (meta?.type === "meta") meta.references[event.recipe_id] = event.lab;
            setRecipeMessage(`REFERENCE SET · ${event.recipe_id} from vial ${event.seq}.`);
            void recipeLibrary.refetch();
          } else if (event.type === "reference_failed") {
            setRecipeMessage(`Vial ${event.seq} had no usable colour; CREATE mode stays on for the next vial.`);
          }
        }
      }
      else if (data.type === "clock") {
        serverClock.current = { t: data.t, at: performance.now(), speed: data.speed, paused: data.paused };
        setLag(data.lag_s);
        setStatus("live");
        if (data.cameras) setCameraStats(data.cameras);
      } else if (data.type === "preview") setPreview(data.key);
      else if (data.type === "saved") { setSaved(data.run_id); void runs.refetch(); }
      else if (data.type === "end") { setStatus("ended"); ws.close(); }
      else if (data.type === "error") { setError(data.message); setStatus("error"); }
      setEventCount(events.current.length);
    };
    ws.onclose = () => setStatus((s) => (s === "live" || s === "connecting" ? "ended" : s));
  }

  async function createRecipe() {
    if (!newRecipe) return;
    try {
      const { name, ...ml } = newRecipe;
      const recipe = await createLineRecipe(name.trim(), ml);
      await recipeLibrary.refetch();
      setDraftRecipeId(recipe.id);
      setNewRecipe(null);
      setRecipeMessage(`${recipe.name} created. Add it to the order, then CREATE its reference from the next vial.`);
    } catch (createError) {
      setRecipeMessage(createError instanceof Error ? createError.message : "Recipe could not be created");
    }
  }

  async function toggleCapture() {
    try {
      const armed = await setLineCapture(capture ? null : draftRecipeId);
      await recipeLibrary.refetch();
      const name = recipeLibrary.data?.recipes.find((recipe) => recipe.id === armed)?.name ?? armed;
      setRecipeMessage(armed ? `The next vial the live cameras decide sets the reference for ${name}.` : "CREATE mode off.");
    } catch (captureError) {
      setRecipeMessage(captureError instanceof Error ? captureError.message : "CREATE mode could not be changed");
    }
  }

  function addRecipeToOrder() {
    const recipe = recipeLibrary.data?.recipes.find((item) => item.id === draftRecipeId);
    if (!recipe) return;
    const additions = Array.from({ length: Math.max(1, Math.min(100, draftVials)) }, (_, index) =>
      recipeContext(recipe, recipeDraft.length + index + 1));
    setRecipeDraft((current) => normalizePlan([...current, ...additions]));
    setRecipeMessage(null);
  }

  function moveRecipe(index: number, direction: -1 | 1) {
    setRecipeDraft((current) => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return normalizePlan(next);
    });
  }

  function beginNewRecipeOrder() {
    setNewRecipeOrder(true);
    setRecipeDraft([]);
    setDraftName(`CSI live order ${new Date().toLocaleString()}`);
    setRecipeMessage("New order started. Add the recipes in the same sequence as the station.");
  }

  async function saveRecipeOrder() {
    const recipes = normalizePlan(recipeDraft);
    if (!recipes.length) { setRecipeMessage("Add at least one vial before saving the recipe order."); return; }
    setRecipeSaving(true);
    setRecipeMessage(null);
    try {
      let orderId = newRecipeOrder || !order ? null : order;
      if (!orderId) {
        const session = await createInspectionSession({
          camera_id: "line-fill-camera",
          station_name: "Fluid Mixing Station 3",
          run_name: draftName.trim() || "CSI live recipe order",
          source_mode: "live_camera",
          source_name: "LINE page recipe setup",
          recipe: recipes[0],
        });
        orderId = session.session_id;
      }
      await replaceRecipePlan(orderId, recipes);
      setNewRecipeOrder(false);
      setOrder(orderId);
      await Promise.all([
        orders.refetch(),
        queryClient.invalidateQueries({ queryKey: ["line-order-plan", orderId] }),
      ]);
      setRecipeMessage(status === "live"
        ? "Recipe order saved. The live line will pick up edits to its selected order; press Restart if this is a newly selected order."
        : "Recipe order saved and selected. Press Start when the first vial is ready.");
    } catch (saveError) {
      setRecipeMessage(saveError instanceof Error ? saveError.message : "Recipe order could not be saved");
    } finally {
      setRecipeSaving(false);
    }
  }

  /** Ask the server to end the session (a camera session then saves itself and says so), then close. */
  function finish() {
    const ws = socket.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) { stop(); setStatus("idle"); return; }
    ws.send(JSON.stringify({ stop: true }));
    window.setTimeout(() => { if (ws.readyState === WebSocket.OPEN) ws.close(); }, 15000);
  }

  function changeSpeed(next: number) {
    setSpeed(next);
    if (mode === "live") socket.current?.send(JSON.stringify({ speed: next }));
    else if (fill.current) fill.current.playbackRate = next;
  }

  function togglePause() {
    const next = !paused;
    setPaused(next);
    socket.current?.send(JSON.stringify({ paused: next }));
  }

  // Clock: the server's video time (live) or the fill video's playhead (review); videos follow it.
  useEffect(() => {
    let frame = 0, lastRender = 0;
    const loop = (now: number) => {
      frame = requestAnimationFrame(loop);
      let time = fill.current?.currentTime ?? 0;
      const sc = serverClock.current;
      if (mode === "live" && sc) {
        time = sc.paused ? sc.t : sc.t + Math.min(1, (now - sc.at) / 1000) * sc.speed;
        for (const video of camera ? [] : [fill.current, tray.current]) {
          if (!video) continue;
          if (Math.abs(video.currentTime - time) > Math.max(0.6, sc.speed * 0.3)) video.currentTime = time;
          video.playbackRate = Math.min(sc.speed, 16);
          if (sc.paused || status !== "live") { if (!video.paused) video.pause(); }
          else if (video.paused) void video.play().catch(() => undefined);
        }
      } else if (mode === "review" && fill.current && tray.current) {
        const f = fill.current, tv = tray.current;
        if (Math.abs(tv.currentTime - f.currentTime) > 0.4) tv.currentTime = f.currentTime;
        tv.playbackRate = f.playbackRate;
        if (f.paused !== tv.paused) { if (f.paused) tv.pause(); else void tv.play().catch(() => undefined); }
      }
      if (now - lastRender > 66) { lastRender = now; setT(time); }
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [camera, mode, status]);

  // eventCount forces a recompute when live events arrive.
  const view = useMemo(() => viewAt(events.current, t), [t, eventCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const live = mode === "live";
  return (
    <PageChrome>
      <main className="ln-main">
        <section className="ln-intro">
          <div>
            <small className="ln-eyebrow">WHOLE-LINE VIEW · FILL STATION + FINISHED TRAY</small>
            <h1>Line runner</h1>
            <p>
              Each vial is decided when the carrier moves it on, from what the cameras have shown so far. Actions are
              recommendations only; nothing is sent to the line.
            </p>
          </div>
          <div className="ln-status">
            <span className={`ln-badge ln-${live ? (status === "live" ? (paused ? "warning" : "live") : "muted") : "muted"}`}>
              {live ? (status === "live" ? (paused ? "PAUSED" : "LIVE") : status.toUpperCase()) : "REVIEW"}
            </span>
            <strong>{clock(view.t)}</strong>
            {live && status === "live" && <small>{lag > 0.5 ? `processing ${lag.toFixed(1)} s behind` : "keeping up"}</small>}
            {camera && simulate === "mock" && <small>MOCK cameras: recorded video, not the station</small>}
            {camera && cameraStats && <small>{cameraStats.fill ?? 0} fill · {cameraStats.tray ?? 0} tray frames · {cameraStats.dropped} dropped{cameraStats.reconnects ? ` · ${cameraStats.reconnects} reconnects` : ""}</small>}
          </div>
        </section>

        <section className="ln-controls panel">
          <label>Run
            <select value={run?.id ?? ""} onChange={(e) => { stop(); setStatus("idle"); setRunId(e.target.value); }}>
              {runs.data?.map((r) => <option key={r.id} value={r.id}>{r.title}</option>)}
            </select>
          </label>
          <label>Recipe order
            <select value={order ?? ""} onChange={(e) => { stop(); setStatus("idle"); setNewRecipeOrder(false); setOrder(e.target.value); }}>
              {orders.data?.map((o) => (
                <option key={o.session_id} value={o.session_id}>
                  {o.run_name || `Run ${o.session_id.slice(0, 8)}`} · {o.vials} vial{o.vials === 1 ? "" : "s"}
                </option>
              ))}
              <option value="">Run's schedule file</option>
            </select>
          </label>
          <div className="ln-seg" role="group" aria-label="Mode">
            <button className={live ? "active" : ""} onClick={() => setMode("live")}>Live</button>
            <button className={!live ? "active" : ""} onClick={() => { setMode("review"); }} disabled={!run?.sources.includes("recorded")}>Review</button>
          </div>
          {live && (
            <>
              <label>Source
                <select value={source} onChange={(e) => setSource(e.target.value as LineSource)}>
                  {run?.sources.includes("camera") && <option value="camera">Cameras (live, models now)</option>}
                  {run?.sources.includes("recorded") && <option value="recorded">Recorded model outputs</option>}
                  {run?.sources.includes("video") && <option value="video">Video → models now</option>}
                </select>
              </label>
              {camera && run?.mock && (
                <div className="ln-seg" role="group" aria-label="Camera input">
                  <button className={simulate !== "mock" ? "active" : ""} onClick={() => setSimulate("")}>Webcams</button>
                  <button className={simulate === "mock" ? "active" : ""} onClick={() => setSimulate("mock")}>Mock</button>
                </div>
              )}
              {camera && simulate !== "mock" && runs.data?.some((r) => r.sources.includes("video")) && (
                <label>Cameras
                  <select value={simulate} onChange={(e) => setSimulate(e.target.value)}>
                    <option value="">Station cameras</option>
                    {runs.data?.filter((r) => r.sources.includes("video")).map((r) => (
                      <option key={r.id} value={r.id}>Simulate with {r.title}</option>
                    ))}
                  </select>
                </label>
              )}
              {(source === "recorded" || (camera && simulate)) && (
                <label>Start at (min)
                  <input type="number" min={0} max={28} value={startMin} onChange={(e) => setStartMin(Math.max(0, Number(e.target.value)))} />
                </label>
              )}
              <button className="ln-primary" onClick={startLive} disabled={!run || !run.sources.length}>
                {status === "live" ? "Restart" : "Start"}
              </button>
              {!camera && <button onClick={togglePause} disabled={status !== "live"}>{paused ? "Resume" : "Pause"}</button>}
              <button onClick={finish} disabled={status !== "live" && status !== "connecting"}>Stop</button>
            </>
          )}
          {!(live && camera) && <div className="ln-seg" role="group" aria-label="Speed">
            {SPEEDS.map((s) => <button key={s} className={speed === s ? "active" : ""} onClick={() => changeSpeed(s)}>{s}×</button>)}
          </div>}
        </section>

        <section className="ln-recipe-setup panel">
          <div className="ln-recipe-heading">
            <div>
              <small>RECIPE SETUP · AT THE LINE</small>
              <h2>Match the station recipe order</h2>
              <p>Select what CSI is making and how many consecutive vials. LINE assigns vial 1 to row 1, vial 2 to row 2, and so on.</p>
            </div>
            <button type="button" onClick={beginNewRecipeOrder}>New order</button>
          </div>
          {newRecipeOrder && (
            <label className="ln-order-name">Order name
              <input value={draftName} maxLength={160} onChange={(event) => setDraftName(event.target.value)} />
            </label>
          )}
          <div className="ln-recipe-picker">
            <label>Recipe
              <select value={draftRecipeId} onChange={(event) => setDraftRecipeId(event.target.value)} disabled={!recipeLibrary.data?.recipes.length}>
                {recipeLibrary.data?.recipes.map((recipe) => (
                  <option key={recipe.id} value={recipe.id}>
                    {recipe.name} · {recipeSummary(recipe) || "amounts unavailable"}{recipe.custom ? " · custom" : ""}{recipe.reference ? "" : " · no color reference"}
                  </option>
                ))}
              </select>
            </label>
            <label>Consecutive vials
              <input type="number" min={1} max={100} value={draftVials} onChange={(event) => setDraftVials(Math.max(1, Number(event.target.value) || 1))} />
            </label>
            <button className="ln-primary" type="button" onClick={addRecipeToOrder} disabled={!recipeLibrary.data}>Add to order</button>
            <button type="button" onClick={() => setNewRecipe(newRecipe ? null : { name: "", t100_ml: 0, t200_ml: 0, t400_ml: 0 })}>
              {newRecipe ? "Cancel" : "New recipe"}
            </button>
          </div>
          {newRecipe && (
            <div className="ln-recipe-picker ln-new-recipe">
              <label>Name
                <input value={newRecipe.name} maxLength={80} placeholder="e.g. Blue 50" onChange={(e) => setNewRecipe({ ...newRecipe, name: e.target.value })} />
              </label>
              {([["t100_ml", "Clear (T100) mL"], ["t200_ml", "Orange (T200) mL"], ["t400_ml", "Blue (T400) mL"]] as const).map(([key, label]) => (
                <label key={key}>{label}
                  <input type="number" min={0} max={1000} value={newRecipe[key]} onChange={(e) => setNewRecipe({ ...newRecipe, [key]: Math.max(0, Number(e.target.value) || 0) })} />
                </label>
              ))}
              <button className="ln-primary" type="button" onClick={() => void createRecipe()}
                disabled={!newRecipe.name.trim() || !(newRecipe.t100_ml + newRecipe.t200_ml + newRecipe.t400_ml)}>Create recipe</button>
            </div>
          )}
          {(() => {
            const armed = recipeLibrary.data?.recipes.find((r) => r.id === capture);
            const selected = recipeLibrary.data?.recipes.find((r) => r.id === draftRecipeId);
            const recreate = Boolean((armed ?? selected)?.reference);
            return (
              <div className={`ln-create-mode${capture ? " armed" : ""}`}>
                <p>{capture
                  ? <><b>{recreate ? "RECREATE MODE" : "CREATE MODE"}</b> · the next vial the live cameras decide {recreate ? "replaces" : "sets"} the reference for <b>{armed?.name ?? capture}</b>{recreate ? " (the current one is used until then, and kept in its history)" : ""}.</>
                  : recreate
                    ? <>{selected?.name} has a reference ({selected?.reference?.from}). RECREATE replaces it with the next vial's colour.</>
                    : <>CREATE mode: send a vial of the selected recipe through and its measured colour becomes that recipe's reference.</>}</p>
                <button type="button" className={capture ? "" : "ln-primary"} onClick={() => void toggleCapture()} disabled={!capture && !draftRecipeId}>
                  {capture ? `Cancel ${recreate ? "RECREATE" : "CREATE"}` : `${recreate ? "RECREATE" : "CREATE"} reference from next vial`}
                </button>
              </div>
            );
          })()}
          <div className="ln-recipe-draft">
            {selectedPlan.isLoading && !newRecipeOrder ? <p>Loading selected recipe order…</p> : null}
            {!selectedPlan.isLoading && !recipeDraft.length ? <p>No vials in this order yet. Select a recipe above and add it.</p> : null}
            {recipeDraft.map((recipe, index) => {
              const definition = recipeLibrary.data?.recipes.find((item) => item.id === recipe.current_recipe_id);
              return (
                <div className="ln-recipe-row" key={`${index}-${recipe.current_recipe_id}`}>
                  <strong>V{index + 1}</strong>
                  <span><b>{recipe.current_recipe_name}</b><small>{definition ? recipeSummary(definition) : `${recipe.total_ml ?? 0} mL total`}</small></span>
                  <em className={definition?.reference ? "ln-tone-signal" : "ln-tone-warning"}>{definition?.reference ? "REFERENCE READY" : "NO REFERENCE · HOLD"}</em>
                  <div>
                    <button type="button" disabled={index === 0} onClick={() => moveRecipe(index, -1)} aria-label={`Move vial ${index + 1} earlier`}>↑</button>
                    <button type="button" disabled={index === recipeDraft.length - 1} onClick={() => moveRecipe(index, 1)} aria-label={`Move vial ${index + 1} later`}>↓</button>
                    <button type="button" onClick={() => setRecipeDraft((current) => normalizePlan(current.filter((_, item) => item !== index)))} aria-label={`Remove vial ${index + 1}`}>×</button>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="ln-recipe-save">
            <p>{recipeMessage ?? (newRecipeOrder ? "Building a new order." : order ? "Editing the recipe order selected above." : "Create an order for the live line.")}</p>
            <button className="ln-primary" type="button" disabled={recipeSaving || !recipeDraft.length} onClick={() => void saveRecipeOrder()}>
              {recipeSaving ? "Saving…" : "Save & use on line"}
            </button>
          </div>
        </section>

        <p className="ln-hint">
          Recipe orders saved here also remain available in the <Link to="/monitor">Monitor</Link> and run history. Operator intent stays
          separate from measured color evidence.
        </p>
        {runs.isError && <p className="ln-banner">Could not load line runs: {(runs.error as Error).message}</p>}
        {runs.data && !runs.data.length && (
          <p className="ln-banner">No line runs configured. Set APP__LINE_RUNS to a folder of run manifests (see docs/line-view.md).</p>
        )}
        {error && <p className="ln-banner ln-banner-failure">{error}</p>}
        {run?.live && run.models === false && (
          <p className="ln-banner">The server has no models loaded: set APP__MODEL_ASSETS to use live cameras.</p>
        )}
        {saved && (
          <p className="ln-banner ln-banner-ok">
            Saved this live session as run <strong>{saved}</strong>.{" "}
            <button className="ln-link" onClick={() => { stop(); setRunId(saved); setMode("review"); }}>Open it in Review</button>
          </p>
        )}
        {mode === "review" && review.isLoading && <p className="ln-banner">Running the line engine over the recorded run…</p>}

        {run && (
          <section className="ln-stage">
            <FillCamera view={view} runId={run.id} videoRef={fill} controls={!live}
              preview={live && camera && preview ? previewUrl(preview, "fill") : undefined}
              maskPreview={live && camera && preview ? previewUrl(preview, "fill_mask") : undefined} />
            <Panel view={view} runId={run.id} />
            {run.has_tray && <TrayCamera view={view} runId={run.id} videoRef={tray}
              preview={live && camera && preview ? previewUrl(preview, "tray") : undefined}
              detectionPreview={live && camera && preview ? previewUrl(preview, "tray_detection") : undefined} />}
          </section>
        )}

        {run && <Ledger view={view} runId={run.id} />}
      </main>
    </PageChrome>
  );
}

function FillCamera({ view, runId, videoRef, controls, preview, maskPreview }: { view: LineView; runId: string; videoRef: React.RefObject<HTMLVideoElement | null>; controls: boolean; preview?: string; maskPreview?: string }) {
  const [showMasks, setShowMasks] = useState(true);
  const roi = view.meta?.roi;
  const last = view.vials[view.vials.length - 1];
  const justDecided = last && view.t - last.decision_s < BANNER_S;
  const tone = justDecided ? VERDICT_TONE[last.verdict] : "cyan";
  const recent = view.actions.filter((a) => view.t - a.issued_s < BANNER_S && a.kind !== "PASS");
  const banner = recent[recent.length - 1];
  return (
    <figure className="ln-camera">
      <figcaption><span>Fill station camera</span>{maskPreview && <button type="button" className={showMasks ? "active" : ""} onClick={() => setShowMasks((current) => !current)}>{showMasks ? "MASKS ON" : "MASKS OFF"}</button>}</figcaption>
      <div className="ln-video">
        {preview ? <img src={preview} alt="Fill station camera, live" />
          : runId === "live" ? <div className="ln-no-feed">Camera feed appears when the live session starts</div>
          : <video ref={videoRef} src={videoUrl(runId, "fill")} muted playsInline preload="auto" controls={controls} />}
        {showMasks && maskPreview && <img src={maskPreview} alt="Model masks and vial geometry over the live fill camera" />}
        <svg viewBox="0 0 640 480" aria-hidden="true">
          {roi && (view.dwell || justDecided) && (
            <rect x={roi[0]} y={roi[1]} width={roi[2] - roi[0]} height={roi[3] - roi[1]} className={`ln-roi ln-stroke-${tone}`} />
          )}
        </svg>
        {banner && (
          <div className={`ln-overlay-banner ln-tone-${ACTION_TONE[banner.kind] ?? "muted"}`}>
            <strong>{banner.seq ? `VIAL ${banner.seq}` : "LINE"}: {recent.filter((a) => a.seq === banner.seq).map((a) => a.kind.replaceAll("_", " ")).join(" + ")}</strong>
            <span>{banner.reason}</span>
          </div>
        )}
      </div>
    </figure>
  );
}

function TrayCamera({ view, runId, videoRef, preview, detectionPreview }: { view: LineView; runId: string; videoRef: React.RefObject<HTMLVideoElement | null>; preview?: string; detectionPreview?: string }) {
  const [showDetections, setShowDetections] = useState(true);
  const traySlots = view.meta?.tray_slots ?? { good: view.meta?.good_slots ?? [], bad: [] };
  const slotIndex = (name: string | null) => {
    const m = name?.match(/^R(\d)C(\d)$/);
    return m ? (Number(m[1]) - 1) * 6 + Number(m[2]) - 1 : -1;
  };
  const latest = view.arrivals[view.arrivals.length - 1];
  const showLatest = latest && view.t - latest.t < BANNER_S;
  return (
    <figure className="ln-camera">
      <figcaption><span>Finished-vial tray camera</span>{detectionPreview && <button type="button" className={showDetections ? "active" : ""} onClick={() => setShowDetections((current) => !current)}>{showDetections ? "DETECTIONS ON" : "DETECTIONS OFF"}</button>}</figcaption>
      <div className="ln-video">
        {preview ? <img src={preview} alt="Finished-vial tray camera, live" />
          : runId === "live" ? <div className="ln-no-feed">Camera feed appears when the live session starts</div>
          : <video ref={videoRef} src={videoUrl(runId, "tray")} muted playsInline preload="auto" />}
        {showDetections && detectionPreview && <img src={detectionPreview} alt="Detected vials in the Good and Bad sorting trays" />}
        <svg viewBox="0 0 640 480" aria-hidden="true">
          {view.arrivals.map((a, i) => {
            const k = slotIndex(a.slot);
            const [x, y] = k >= 0 && traySlots[a.tray]?.[k] ? traySlots[a.tray][k] : a.position ?? [-100, -100];
            const tone = a.capped === false ? "nocap" : VERDICT_TONE[a.verdict ?? ""] ?? "muted";
            return (
              <g key={i} className={`ln-slot ln-stroke-${tone}`}>
                <rect x={x - 14} y={y - 14} width={28} height={28} />
                <text x={x} y={y + 5}>{a.seq ?? "?"}</text>
                {a.capped === false && <text className="ln-nocap" x={x} y={y + 27}>NO CAP</text>}
              </g>
            );
          })}
        </svg>
        {showLatest && (
          <div className={`ln-overlay-banner ln-bottom ln-tone-${latest.capped === false ? "nocap" : latest.status === "escaped" ? "failure" : "muted"}`}>
            <strong>Vial {latest.seq ?? "?"} → {latest.tray} tray {latest.slot ?? ""}{latest.transit_s ? ` after ${Math.round(latest.transit_s)} s` : ""}</strong>
            <span>{latest.capped === false ? "NO CAP" : arrivalNote(latest)}</span>
          </div>
        )}
      </div>
    </figure>
  );
}

function arrivalNote(a: TrayArrival) {
  if (a.status === "escaped") return `we said ${a.decision}: should have been diverted`;
  if (a.status === "good_vial_in_bad_tray") return "good vial in Bad tray";
  if (a.status === "unmatched_arrival") return "no matching vial left the fill station";
  return "as decided";
}

function Panel({ view, runId }: { view: LineView; runId: string }) {
  const meta = view.meta;
  const ordered = meta?.scheduled_vials ?? 0;
  const n = Math.max(ordered, view.vials.length);
  const next = view.vials.length + 1;
  const recipe = meta?.schedule[next - 1] ?? null;
  const reference = recipe ? meta?.references[recipe] ?? null : null;
  const samples = view.dwell?.samples ?? [];
  const latestSample = samples[samples.length - 1];
  const valid = samples.filter((s) => s.qa_valid && s.lab);
  const measured = valid[valid.length - 1]?.lab ?? null;
  const typical = view.vials.length ? median(view.vials.map((v) => v.final_liquid_px)) : 5000;
  const area = samples.length ? Math.max(...samples.map((s) => s.liquid_px)) : 0;
  const last = view.vials[view.vials.length - 1];
  const decidedSeq = new Map(view.vials.map((v) => [v.seq, v]));
  const actions = view.actions.slice(-6).reverse();
  const arrived = view.arrivals.filter((a) => a.seq);
  const inTransit = view.vials.length - arrived.length - view.missed.length;
  const escaped = arrived.filter((a) => a.status === "escaped").length;
  const noCap = arrived.filter((a) => a.capped === false).map((a) => a.seq);

  return (
    <aside className="ln-panel panel">
      {!meta ? <p className="ln-muted">Press Start (live) or choose Review.</p> : (
        <>
          <div className="ln-block">
            <small>UNDER THE NOZZLE</small>
            {view.dwell ? (
              <>
                <h2>Vial {next} / {ordered} <span>{recipe ? `recipe ${recipe}` : "not in the recipe order yet"}</span></h2>
                <p className="ln-phase">{phase(view)}</p>
                {latestSample && <p className={`ln-detection ${latestSample.vial_boxes ? "ln-tone-signal" : "ln-tone-failure"}`}>
                  {latestSample.vial_boxes
                    ? `Vial geometry: ${latestSample.vial_geometry_source === "yolo" ? "YOLO" : "STATION PROFILE CANDIDATE"}${latestSample.yolo_vial_boxes ? "" : " · YOLO 0"}`
                    : "VIAL NOT DETECTED · measurement unavailable"}
                </p>}
                <div className="ln-bar" aria-label="Liquid area"><span style={{ width: `${Math.min(100, (100 * area) / typical)}%` }} /></div>
                <div className="ln-swatches">
                  <div><span style={{ background: labToCss(measured) }} className={measured ? "" : "ln-empty"} /><small>measured</small></div>
                  <div><span style={{ background: labToCss(reference) }} className={reference ? "" : "ln-empty"} /><small>reference</small></div>
                  <strong>{measured && reference ? `ΔE ${deltaE2000(measured, reference).toFixed(1)}` : "—"}</strong>
                </div>
              </>
            ) : <p className="ln-muted">Carrier moving</p>}
          </div>

          <div className="ln-block">
            <small>LAST DECISION</small>
            {last ? (
              <>
                <h2>Vial {last.seq} <span>recipe {last.recipe_id}</span></h2>
                <p className={`ln-verdict ln-tone-${VERDICT_TONE[last.verdict]}`}>
                  {last.verdict.replaceAll("_", " ")}{last.pattern !== "nominal" && <em> · {last.pattern.replaceAll("_", " ")}</em>}
                </p>
                <p className="ln-muted">
                  {last.de_intended != null ? `endpoint ΔE ${last.de_intended.toFixed(1)} to recipe` : "no QA-valid endpoint colour"}
                  {last.direction_cos != null && `, cos ${last.direction_cos.toFixed(2)} toward ${last.previous_recipe_id}`}
                </p>
                <Why vial={last} actions={view.actions} />
                {last.verdict !== "IN_SPEC" && <AiExplain key={`${runId}-${last.seq}`} runId={runId} vial={last} view={view} />}
              </>
            ) : <p className="ln-muted">No vial decided yet</p>}
          </div>

          <div className="ln-block">
            <small>ACTIONS</small>
            <ul className="ln-actions">
              {actions.map((a, i) => (
                <li key={`${a.issued_s}-${a.kind}-${i}`}>
                  <span className="ln-muted">{clock(a.issued_s)}</span>
                  <strong className={`ln-tone-${ACTION_TONE[a.kind] ?? "muted"}`}>{a.kind.replaceAll("_", " ")}</strong>
                  <span>{a.seq ? `vial ${a.seq}` : "line"}</span>
                  {a.deadline_s != null && a.kind !== "PASS" && a.deadline_s - view.t >= 1 && <em>due {Math.round(a.deadline_s - view.t)} s</em>}
                  {a.kind !== "PASS" && <small className="ln-reason">{a.reason}</small>}
                </li>
              ))}
              {!actions.length && <li className="ln-muted">None yet</li>}
            </ul>
          </div>

          <div className="ln-block">
            <small>RUN</small>
            <div className="ln-strip" style={{ gridTemplateColumns: `repeat(${Math.max(n, 1)}, 1fr)` }}>
              {Array.from({ length: n }, (_, i) => {
                const v = decidedSeq.get(i + 1);
                return <span key={i} title={`vial ${i + 1}`} className={`${v ? `ln-fill-${VERDICT_TONE[v.verdict]}` : ""} ${i + 1 === next && view.dwell ? "ln-current" : ""}`} />;
              })}
            </div>
          </div>

          <div className="ln-block ln-tray-stats">
            <small>FINISHED TRAY</small>
            <dl>
              <div><dt>in tray</dt><dd>{arrived.length}</dd></div>
              <div><dt>in transit</dt><dd>{Math.max(0, inTransit)}</dd></div>
              <div><dt>should have been diverted</dt><dd className={escaped ? "ln-tone-failure" : ""}>{escaped}</dd></div>
              <div><dt>missing cap</dt><dd className={noCap.length ? "ln-tone-nocap" : ""}>{noCap.length ? noCap.join(", ") : 0}</dd></div>
            </dl>
          </div>
        </>
      )}
    </aside>
  );
}

const CAUSE: Record<string, string> = {
  carryover: "carryover from the previous recipe",
  wrong_recipe_or_dose: "wrong recipe or dosing",
  unknown: "not established",
};

/** Rule-based explanation of one vial's decision: the cause and every action's reason. */
function Why({ vial, actions }: { vial: LineVial; actions: LineAction[] }) {
  const own = actions.filter((a) => a.seq === vial.seq && a.kind !== "PASS");
  if (vial.verdict === "IN_SPEC" && !own.length) return null;
  return (
    <div className="ln-why">
      {vial.rule_cause !== "none" && <p><small>CAUSE (RULE)</small> {CAUSE[vial.rule_cause] ?? vial.rule_cause.replaceAll("_", " ")}</p>}
      <ul>{own.map((a, i) => <li key={`${a.kind}-${i}`}><strong className={`ln-tone-${ACTION_TONE[a.kind] ?? "muted"}`}>{a.kind.replaceAll("_", " ")}</strong> {a.reason}</li>)}</ul>
    </div>
  );
}

// Answers survive re-renders and switching between the panel and the ledger.
const explained = new Map<string, VialExplanation>();

/** On request: Nemotron's explanation of a decided vial. Advisory; the verdict and actions are the rules'. */
function AiExplain({ runId, vial, view }: { runId: string; vial: LineVial; view: LineView }) {
  const key = `${runId}:${vial.seq}:${vial.verdict}:${vial.recipe_id}`;
  const [answer, setAnswer] = useState<VialExplanation | null>(() => explained.get(key) ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ask = async () => {
    setBusy(true);
    setError(null);
    try {
      const decided = view.vials.filter((v) => v.seq <= vial.seq + 2);
      const result = await explainVial(runId, vial, view.actions, decided, view.meta?.schedule[vial.seq] ?? null);
      explained.set(key, result);
      setAnswer(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Explanation unavailable");
    } finally {
      setBusy(false);
    }
  };
  if (!answer) {
    return (
      <div className="ln-ai">
        <button type="button" onClick={(e) => { e.stopPropagation(); void ask(); }} disabled={busy}>
          {busy ? "ASKING NEMOTRON…" : "EXPLAIN WITH AI"}
        </button>
        {error && <small className="ln-tone-failure">{error}</small>}
      </div>
    );
  }
  const e = answer.explanation;
  return (
    <div className="ln-ai ln-ai-answer">
      <p className="ln-ai-head">
        <small>AI · ADVISORY</small> {answer.model.replace(/^nvidia\//, "")}
        {answer.source === "recorded" ? ` · recorded ${answer.recorded ?? ""}` : answer.latency_s != null ? ` · ${answer.latency_s} s` : ""}
      </p>
      <p><strong>{e.cause.replaceAll("_", " ")}</strong> <span className="ln-muted">confidence {Math.round(e.confidence * 100)}%</span></p>
      {e.explanation && <p>{e.explanation}</p>}
      {e.next_check && <p><small>NEXT CHECK</small> {e.next_check}</p>}
      {answer.contradictions.length > 0 && <p className="ln-tone-warning">The measurements contradict this: {answer.contradictions.map((c) => c.replaceAll("_", " ")).join("; ")}</p>}
      {answer.note && <p className="ln-muted">{answer.note}; showing the recorded answer for this vial.</p>}
    </div>
  );
}

function Ledger({ view, runId }: { view: LineView; runId: string }) {
  const arrivals = new Map(view.arrivals.filter((a) => a.seq).map((a) => [a.seq!, a]));
  const [open, setOpen] = useState<number | null>(null);
  if (!view.vials.length) return null;
  return (
    <section className="ln-ledger panel">
      <div className="panel-heading compact"><div><h2>Ledger</h2><p>Vials decided so far, newest first · click a row for why</p></div></div>
      <div className="ln-table-wrap">
        <table>
          <thead><tr><th>Vial</th><th>Recipe</th><th>Decided</th><th>Colour</th><th>ΔE</th><th>Verdict</th><th>Pattern</th><th>Actions</th><th>Tray</th></tr></thead>
          <tbody>
            {[...view.vials].reverse().map((v) => {
              const a = arrivals.get(v.seq);
              return (
                <Fragment key={v.seq}>
                <tr className="ln-row" onClick={() => setOpen(open === v.seq ? null : v.seq)} aria-expanded={open === v.seq}>
                  <td>{v.seq}</td>
                  <td>{v.recipe_id}{v.previous_recipe_id ? <small> after {v.previous_recipe_id}</small> : null}</td>
                  <td>{clock(v.decision_s)}</td>
                  <td><span className="ln-chip-swatch" style={{ background: labToCss(v.endpoint_lab) }} /></td>
                  <td>{v.de_intended != null ? v.de_intended.toFixed(1) : "—"}</td>
                  <td className={`ln-tone-${VERDICT_TONE[v.verdict]}`}>{v.verdict.replaceAll("_", " ")}</td>
                  <td>{v.pattern.replaceAll("_", " ")}</td>
                  <td>{v.actions.replaceAll("+", " + ")}</td>
                  <td>{a ? `${a.tray}${a.slot ? ` ${a.slot}` : ""}${a.capped === false ? " · NO CAP" : ""}${a.status === "escaped" ? " · escaped" : ""}` : view.missed.includes(v.seq) ? "not seen" : "in transit"}</td>
                </tr>
                {open === v.seq && (
                  <tr className="ln-row-detail"><td colSpan={9}>
                    <p className="ln-muted">
                      {v.de_intended != null ? `ΔE ${v.de_intended.toFixed(1)} to recipe ${v.recipe_id}` : "no QA-valid endpoint colour"}
                      {v.de_previous != null && ` · ΔE ${v.de_previous.toFixed(1)} to previous recipe ${v.previous_recipe_id}`}
                      {v.direction_cos != null && ` · shift toward it cos ${v.direction_cos.toFixed(2)}`}
                    </p>
                    <Why vial={v} actions={view.actions} />
                    {v.verdict !== "IN_SPEC" && <AiExplain runId={runId} vial={v} view={view} />}
                    {v.verdict === "IN_SPEC" && <p className="ln-muted">Within the recipe band: routed to the Good tray.</p>}
                  </td></tr>
                )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function median(values: number[]) {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}
