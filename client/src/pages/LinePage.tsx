import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import PageChrome from "@/components/PageChrome";
import { deltaE2000 } from "@/services/recipeCheck";
import {
  listLineRuns, liveSocket, reviewEvents, videoUrl,
  type LineEvent, type LiveMessage, type TrayArrival,
} from "@/services/lineService";
import { ACTION_TONE, VERDICT_TONE, clock, labToCss, phase, viewAt, type LineView } from "@/services/lineState";
import "./LinePage.css";

type Mode = "live" | "review";
type Status = "idle" | "connecting" | "live" | "ended" | "error";
const SPEEDS = [1, 2, 4, 8, 16];
const BANNER_S = 4;

interface ServerClock { t: number; at: number; speed: number; paused: boolean }

export default function LinePage() {
  const runs = useQuery({ queryKey: ["line-runs"], queryFn: listLineRuns });
  const [runId, setRunId] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("live");
  const [source, setSource] = useState<"recorded" | "video">("recorded");
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

  const run = runs.data?.find((r) => r.id === runId) ?? runs.data?.[0] ?? null;
  useEffect(() => { if (run && !runId) setRunId(run.id); }, [run, runId]);
  useEffect(() => { if (run && !run.sources.includes(source)) setSource(run.sources[0] ?? "recorded"); }, [run, source]);

  const stop = useCallback(() => {
    socket.current?.close();
    socket.current = null;
    serverClock.current = null;
  }, []);
  useEffect(() => stop, [stop]);

  const review = useQuery({
    queryKey: ["line-review", run?.id],
    queryFn: () => reviewEvents(run!.id),
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
    setStatus("connecting");
    const ws = liveSocket(run.id, source, speed, source === "recorded" ? startMin * 60 : 0);
    socket.current = ws;
    ws.onmessage = (message) => {
      const data = JSON.parse(message.data) as LiveMessage;
      if (data.type === "meta") events.current = [data];
      else if (data.type === "events") events.current = events.current.concat(data.events);
      else if (data.type === "clock") {
        serverClock.current = { t: data.t, at: performance.now(), speed: data.speed, paused: data.paused };
        setLag(data.lag_s);
        setStatus("live");
      } else if (data.type === "end") setStatus("ended");
      else if (data.type === "error") { setError(data.message); setStatus("error"); }
      setEventCount(events.current.length);
    };
    ws.onclose = () => setStatus((s) => (s === "live" || s === "connecting" ? "ended" : s));
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
        for (const video of [fill.current, tray.current]) {
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
  }, [mode, status]);

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
          </div>
        </section>

        <section className="ln-controls panel">
          <label>Run
            <select value={run?.id ?? ""} onChange={(e) => { stop(); setStatus("idle"); setRunId(e.target.value); }}>
              {runs.data?.map((r) => <option key={r.id} value={r.id}>{r.title}</option>)}
            </select>
          </label>
          <div className="ln-seg" role="group" aria-label="Mode">
            <button className={live ? "active" : ""} onClick={() => setMode("live")}>Live</button>
            <button className={!live ? "active" : ""} onClick={() => { setMode("review"); }} disabled={!run?.sources.includes("recorded")}>Review</button>
          </div>
          {live && (
            <>
              <label>Source
                <select value={source} onChange={(e) => setSource(e.target.value as "recorded" | "video")}>
                  {run?.sources.includes("recorded") && <option value="recorded">Recorded model outputs</option>}
                  {run?.sources.includes("video") && <option value="video">Video → models now</option>}
                </select>
              </label>
              {source === "recorded" && (
                <label>Start at (min)
                  <input type="number" min={0} max={28} value={startMin} onChange={(e) => setStartMin(Math.max(0, Number(e.target.value)))} />
                </label>
              )}
              <button className="ln-primary" onClick={startLive} disabled={!run || !run.sources.length}>
                {status === "live" ? "Restart" : "Start"}
              </button>
              <button onClick={togglePause} disabled={status !== "live"}>{paused ? "Resume" : "Pause"}</button>
              <button onClick={() => { stop(); setStatus("idle"); }} disabled={status !== "live" && status !== "connecting"}>Stop</button>
            </>
          )}
          <div className="ln-seg" role="group" aria-label="Speed">
            {SPEEDS.map((s) => <button key={s} className={speed === s ? "active" : ""} onClick={() => changeSpeed(s)}>{s}×</button>)}
          </div>
        </section>

        {runs.isError && <p className="ln-banner">Could not load line runs: {(runs.error as Error).message}</p>}
        {runs.data && !runs.data.length && (
          <p className="ln-banner">No line runs configured. Set APP__LINE_RUNS to a folder of run manifests (see docs/line-view.md).</p>
        )}
        {error && <p className="ln-banner ln-banner-failure">{error}</p>}
        {mode === "review" && review.isLoading && <p className="ln-banner">Running the line engine over the recorded run…</p>}

        {run && (
          <section className="ln-stage">
            <FillCamera view={view} runId={run.id} videoRef={fill} controls={!live} />
            <Panel view={view} />
            {run.has_tray && <TrayCamera view={view} runId={run.id} videoRef={tray} />}
          </section>
        )}

        {run && <Ledger view={view} />}
      </main>
    </PageChrome>
  );
}

function FillCamera({ view, runId, videoRef, controls }: { view: LineView; runId: string; videoRef: React.RefObject<HTMLVideoElement | null>; controls: boolean }) {
  const roi = view.meta?.roi;
  const last = view.vials[view.vials.length - 1];
  const justDecided = last && view.t - last.decision_s < BANNER_S;
  const tone = justDecided ? VERDICT_TONE[last.verdict] : "cyan";
  const recent = view.actions.filter((a) => view.t - a.issued_s < BANNER_S && a.kind !== "PASS");
  const banner = recent[recent.length - 1];
  return (
    <figure className="ln-camera">
      <figcaption>Fill station camera</figcaption>
      <div className="ln-video">
        <video ref={videoRef} src={videoUrl(runId, "fill")} muted playsInline preload="auto" controls={controls} />
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

function TrayCamera({ view, runId, videoRef }: { view: LineView; runId: string; videoRef: React.RefObject<HTMLVideoElement | null> }) {
  const slots = view.meta?.good_slots ?? [];
  const slotIndex = (name: string | null) => {
    const m = name?.match(/^R(\d)C(\d)$/);
    return m ? (Number(m[1]) - 1) * 6 + Number(m[2]) - 1 : -1;
  };
  const latest = view.arrivals[view.arrivals.length - 1];
  const showLatest = latest && view.t - latest.t < BANNER_S;
  return (
    <figure className="ln-camera">
      <figcaption>Finished-vial tray camera</figcaption>
      <div className="ln-video">
        <video ref={videoRef} src={videoUrl(runId, "tray")} muted playsInline preload="auto" />
        <svg viewBox="0 0 640 480" aria-hidden="true">
          {view.arrivals.map((a, i) => {
            const k = slotIndex(a.slot);
            const [x, y] = k >= 0 ? slots[k] : a.position ?? [-100, -100];
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

function Panel({ view }: { view: LineView }) {
  const meta = view.meta;
  const n = meta?.scheduled_vials ?? 0;
  const next = view.vials.length + 1;
  const recipe = meta?.schedule[next - 1] ?? null;
  const reference = recipe ? meta?.references[recipe] ?? null : null;
  const samples = view.dwell?.samples ?? [];
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
            {view.dwell && next <= n ? (
              <>
                <h2>Vial {next} / {n} <span>recipe {recipe}</span></h2>
                <p className="ln-phase">{phase(view)}</p>
                <div className="ln-bar" aria-label="Liquid area"><span style={{ width: `${Math.min(100, (100 * area) / typical)}%` }} /></div>
                <div className="ln-swatches">
                  <div><span style={{ background: labToCss(measured) }} className={measured ? "" : "ln-empty"} /><small>measured</small></div>
                  <div><span style={{ background: labToCss(reference) }} className={reference ? "" : "ln-empty"} /><small>reference</small></div>
                  <strong>{measured && reference ? `ΔE ${deltaE2000(measured, reference).toFixed(1)}` : "—"}</strong>
                </div>
              </>
            ) : <p className="ln-muted">{view.dwell ? "Station still · no vial expected" : "Carrier moving"}</p>}
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

function Ledger({ view }: { view: LineView }) {
  const arrivals = new Map(view.arrivals.filter((a) => a.seq).map((a) => [a.seq!, a]));
  if (!view.vials.length) return null;
  return (
    <section className="ln-ledger panel">
      <div className="panel-heading compact"><div><h2>Ledger</h2><p>Vials decided so far, newest first</p></div></div>
      <div className="ln-table-wrap">
        <table>
          <thead><tr><th>Vial</th><th>Recipe</th><th>Decided</th><th>Colour</th><th>ΔE</th><th>Verdict</th><th>Pattern</th><th>Actions</th><th>Tray</th></tr></thead>
          <tbody>
            {[...view.vials].reverse().map((v) => {
              const a = arrivals.get(v.seq);
              return (
                <tr key={v.seq}>
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
