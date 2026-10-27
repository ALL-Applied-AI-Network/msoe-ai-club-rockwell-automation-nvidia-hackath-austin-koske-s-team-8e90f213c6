import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
} from "react";
import { Link } from "react-router-dom";
import type {
  CreateSessionRequest,
  HealthStatus,
  InspectionEvent,
  InspectionSession,
  OperatorStatus,
  SimulationMode,
  SourceMode,
} from "@/models/inspection";
import {
  acknowledgeInspectionEvent,
  addRecipeSegment,
  artifactUrl,
  createInspectionSession,
  getHealth,
  importRecipeSegments,
  inspectionEventUrl,
  submitInspectionFrame,
  uploadRecording,
} from "@/services/inspectionService";

const initialConfig: CreateSessionRequest = {
  camera_id: "fillcam-1",
  station_name: "Fluid Mixing Station 3",
  run_name: `Inspection ${new Date().toLocaleDateString()}`,
  source_mode: "live_camera",
  source_name: null,
  recipe: {
    current_recipe_id: "recipe-07",
    current_recipe_name: "Recipe 07 · Ocean Blue",
    previous_recipe_id: "recipe-06",
    vial_index: 1,
    is_first_vial_after_change: true,
  },
};

const sourceLabels: Record<SourceMode, { label: string; detail: string }> = {
  live_camera: { label: "LIVE CAMERA", detail: "Browser or attached USB camera" },
  images: { label: "IMAGES", detail: "JPEG, PNG, or WebP batches" },
  recording: { label: "RECORDING", detail: "MP4, WebM, or MOV replay" },
};

const statusLabel: Record<OperatorStatus, string> = {
  starting: "STARTING",
  processing: "PROCESSING",
  running_ok: "RUNNING WELL",
  warning: "REVIEW",
  failure: "FAILURE DETECTED",
  unavailable: "UNAVAILABLE",
  stopped: "STOPPED",
};

function imageSource(image: InspectionEvent["image"] | null | undefined) {
  if (!image) return undefined;
  if (image.url) return artifactUrl(image.url);
  return image.base64 ? `data:${image.mime_type};base64,${image.base64}` : undefined;
}

function timestampLabel(value?: number | null) {
  if (value === null || value === undefined) return "LIVE";
  const seconds = value / 1000;
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function cameraFrame(video: HTMLVideoElement) {
  const width = Math.min(960, video.videoWidth);
  const height = Math.round((video.videoHeight / video.videoWidth) * width);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context || !width || !height) throw new Error("Camera is not ready");
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, width, height);
  return new Promise<{ blob: Blob; width: number; height: number }>((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob ? resolve({ blob, width, height }) : reject(new Error("Could not encode camera frame")),
      "image/jpeg",
      0.82,
    );
  });
}

function demoFrame() {
  const width = 960;
  const height = 540;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable");
  const background = context.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#101b22");
  background.addColorStop(1, "#162c31");
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(111, 154, 154, .12)";
  for (let x = 0; x < width; x += 48) {
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  const vialX = 390;
  const vialY = 75;
  context.fillStyle = "rgba(222, 241, 235, .15)";
  context.strokeStyle = "#b6d9d0";
  context.lineWidth = 5;
  context.beginPath(); context.roundRect(vialX, vialY, 180, 390, 26); context.fill(); context.stroke();
  const fluid = context.createLinearGradient(0, 280, 0, 445);
  fluid.addColorStop(0, "#52cbd0"); fluid.addColorStop(1, "#176f91");
  context.fillStyle = fluid;
  context.beginPath(); context.roundRect(vialX + 13, 268, 154, 180, 14); context.fill();
  context.fillStyle = "#dff8ef";
  context.font = "600 18px ui-monospace, monospace";
  context.fillText("DEMO CAMERA FRAME · FILL STATION", 28, 42);
  context.strokeStyle = "#74e7b7";
  context.lineWidth = 3;
  context.strokeRect(vialX - 18, vialY - 18, 216, 426);
  return new Promise<{ blob: Blob; width: number; height: number }>((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob ? resolve({ blob, width, height }) : reject(new Error("Demo failed")),
      "image/jpeg",
      0.86,
    );
  });
}

async function imageDimensions(file: File) {
  const bitmap = await createImageBitmap(file);
  const dimensions = { width: bitmap.width, height: bitmap.height };
  bitmap.close();
  return dimensions;
}

function waitForMedia(video: HTMLVideoElement, eventName: "loadedmetadata" | "loadeddata" | "seeked") {
  return new Promise<void>((resolve, reject) => {
    const onDone = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("The recording could not be decoded")); };
    const cleanup = () => {
      video.removeEventListener(eventName, onDone);
      video.removeEventListener("error", onError);
    };
    video.addEventListener(eventName, onDone, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

export default function OperatorDashboard() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [config, setConfig] = useState<CreateSessionRequest>(initialConfig);
  const [sourceMode, setSourceMode] = useState<SourceMode>("live_camera");
  const [session, setSession] = useState<InspectionSession | null>(null);
  const [events, setEvents] = useState<InspectionEvent[]>([]);
  const [isCameraLive, setIsCameraLive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [view, setView] = useState<"raw" | "output">("output");
  const [message, setMessage] = useState("Ready to create an inspection run");
  const [segmentStartSeconds, setSegmentStartSeconds] = useState(0);
  const [sampleSeconds, setSampleSeconds] = useState(1.5);
  const [dragActive, setDragActive] = useState(false);
  const [ingestion, setIngestion] = useState<{ label: string; current: number; total: number } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const frameNumberRef = useRef(0);
  const busyRef = useRef(false);
  const sessionRef = useRef<InspectionSession | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recipeInputRef = useRef<HTMLInputElement>(null);
  const cancelIngestionRef = useRef(false);

  const latestEvent = events[0] ?? null;
  const sessionId = session?.session_id;
  const resultImage = latestEvent?.visuals?.rendered_frame ?? latestEvent?.image;
  const maskImage = latestEvent?.visuals?.mask;
  const status: OperatorStatus = isProcessing
    ? "processing"
    : latestEvent?.decision.status ?? session?.status ?? "stopped";

  const mergeEvent = useCallback((event: InspectionEvent) => {
    setEvents((current) => [
      event,
      ...current.filter((item) => item.event_id !== event.event_id),
    ].slice(0, 40));
    setView("output");
  }, []);

  useEffect(() => {
    let mounted = true;
    const check = () => getHealth().then((result) => mounted && setHealth(result)).catch(() => mounted && setHealth(null));
    void check();
    const timer = window.setInterval(check, 8000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => { sessionRef.current = session; }, [session]);

  useEffect(() => {
    if (!sessionId) return;
    const socket = new WebSocket(inspectionEventUrl(sessionId));
    socket.onopen = () => setMessage("Live event channel connected");
    socket.onmessage = (event) => mergeEvent(JSON.parse(event.data) as InspectionEvent);
    socket.onclose = () => setMessage("Event channel disconnected");
    return () => socket.close();
  }, [mergeEvent, sessionId]);

  const ensureSession = useCallback(async (sourceName?: string) => {
    if (sessionRef.current) return sessionRef.current;
    const request: CreateSessionRequest = {
      ...config,
      source_mode: sourceMode,
      source_name: sourceName || config.source_name,
    };
    const created = await createInspectionSession(request);
    sessionRef.current = created;
    setSession(created);
    setMessage(`Inspection run ${created.session_id.slice(0, 8)} started`);
    return created;
  }, [config, sourceMode]);

  const submitBlob = useCallback(async (
    blob: Blob,
    width: number,
    height: number,
    sourceTimestampMs?: number,
    originalFilename?: string,
    simulate?: SimulationMode,
    sourceName?: string,
  ) => {
    const activeSession = await ensureSession(sourceName);
    frameNumberRef.current += 1;
    const frameId = `${activeSession.camera_id}-${String(frameNumberRef.current).padStart(6, "0")}`;
    const returned = await submitInspectionFrame(
      activeSession.session_id,
      frameId,
      blob,
      width,
      height,
      sourceTimestampMs,
      originalFilename,
      simulate,
    );
    mergeEvent(returned);
    setSession((current) => current ? {
      ...current,
      recipe: returned.recipe,
      status: returned.decision.status,
      frames_processed: returned.frame_number,
      updated_at: returned.created_at,
    } : current);
    return returned;
  }, [ensureSession, mergeEvent]);

  const sendGeneratedFrame = useCallback(async (
    source: () => Promise<{ blob: Blob; width: number; height: number }>,
    simulate?: SimulationMode,
  ) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setIsProcessing(true);
    try {
      const frame = await source();
      const active = await ensureSession();
      const elapsed = Math.max(0, Date.now() - new Date(active.created_at).getTime());
      const returned = await submitBlob(frame.blob, frame.width, frame.height, elapsed, undefined, simulate);
      setMessage(`Frame ${returned.frame_number} persisted and processed`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Frame processing failed");
    } finally {
      busyRef.current = false;
      setIsProcessing(false);
    }
  }, [ensureSession, submitBlob]);

  const captureCamera = useCallback(() => {
    const video = videoRef.current;
    if (video?.videoWidth) void sendGeneratedFrame(() => cameraFrame(video));
  }, [sendGeneratedFrame]);

  const stopCamera = useCallback(() => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraLive(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startCamera = async () => {
    try {
      await ensureSession("Attached browser camera");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraLive(true);
      setView("raw");
      const intervalSeconds = Number.isFinite(sampleSeconds) && sampleSeconds > 0 ? sampleSeconds : 1.5;
      setMessage(`Camera live · sampling every ${intervalSeconds.toFixed(1)} seconds`);
      timerRef.current = window.setInterval(captureCamera, intervalSeconds * 1000);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Camera could not start");
    }
  };

  const resetSession = useCallback(() => {
    stopCamera();
    cancelIngestionRef.current = true;
    sessionRef.current = null;
    frameNumberRef.current = 0;
    setSession(null);
    setEvents([]);
    setIngestion(null);
    setMessage("Ready for a new persisted inspection run");
  }, [stopCamera]);

  const selectSource = (mode: SourceMode) => {
    if (sessionRef.current) resetSession();
    setSourceMode(mode);
    setConfig((current) => ({ ...current, source_mode: mode, source_name: null }));
    setMessage(`${sourceLabels[mode].label} source selected`);
  };

  const ingestImages = async (files: File[]) => {
    const images = files.filter((file) => file.type.startsWith("image/"));
    if (!images.length) throw new Error("Choose JPEG, PNG, or WebP images");
    cancelIngestionRef.current = false;
    setIngestion({ label: "IMAGE BATCH", current: 0, total: images.length });
    setIsProcessing(true);
    try {
      await ensureSession(images.length === 1 ? images[0].name : `${images.length} image batch`);
      for (let index = 0; index < images.length; index += 1) {
        if (cancelIngestionRef.current) break;
        const file = images[index];
        const { width, height } = await imageDimensions(file);
        await submitBlob(file, width, height, index * 1000, file.name, undefined, file.name);
        setIngestion({ label: "IMAGE BATCH", current: index + 1, total: images.length });
      }
      setMessage(`${images.length} source image${images.length === 1 ? "" : "s"} persisted and queued`);
    } finally {
      setIsProcessing(false);
    }
  };

  const ingestRecording = async (file: File) => {
    if (!file.type.startsWith("video/") && !/\.(mp4|webm|mov)$/i.test(file.name)) {
      throw new Error("Choose an MP4, WebM, or MOV recording");
    }
    cancelIngestionRef.current = false;
    setIsProcessing(true);
    const active = await ensureSession(file.name);
    setMessage("Persisting original recording…");
    await uploadRecording(active.session_id, file);

    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    const metadataReady = waitForMedia(video, "loadedmetadata");
    const firstFrameReady = waitForMedia(video, "loadeddata");
    video.src = objectUrl;
    await metadataReady;
    await firstFrameReady;
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      URL.revokeObjectURL(objectUrl);
      throw new Error("Recording duration is unavailable");
    }

    const intervalSeconds = Number.isFinite(sampleSeconds) && sampleSeconds > 0 ? sampleSeconds : 1.5;
    const intervalMs = Math.max(100, Math.round(intervalSeconds * 1000));
    const total = Math.max(1, Math.floor((video.duration * 1000) / intervalMs) + 1);
    const width = video.videoWidth;
    const height = video.videoHeight;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context || !width || !height) throw new Error("Recording dimensions are unavailable");
    setIngestion({ label: file.name, current: 0, total });
    try {
      for (let index = 0; index < total; index += 1) {
        if (cancelIngestionRef.current) break;
        const timestampMs = Math.min(index * intervalMs, Math.max(0, Math.floor(video.duration * 1000) - 1));
        if (index > 0) {
          const seeked = waitForMedia(video, "seeked");
          video.currentTime = timestampMs / 1000;
          await seeked;
        }
        context.drawImage(video, 0, 0, width, height);
        const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
          (value) => value ? resolve(value) : reject(new Error("Could not encode video frame")),
          "image/jpeg",
          0.86,
        ));
        await submitBlob(
          blob,
          width,
          height,
          timestampMs,
          `${file.name}@${timestampMs}ms.jpg`,
          undefined,
          file.name,
        );
        setIngestion({ label: file.name, current: index + 1, total });
      }
      setMessage(cancelIngestionRef.current ? "Recording ingestion stopped" : `Recording complete · ${total} timestamped frames persisted`);
    } finally {
      URL.revokeObjectURL(objectUrl);
      setIsProcessing(false);
    }
  };

  const handleFiles = async (files: File[]) => {
    try {
      if (sourceMode === "images") await ingestImages(files);
      else if (sourceMode === "recording" && files[0]) await ingestRecording(files[0]);
      else setMessage("Select IMAGES or RECORDING before dropping files");
    } catch (error) {
      setIsProcessing(false);
      setMessage(error instanceof Error ? error.message : "Ingestion failed");
    }
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    void handleFiles(Array.from(event.dataTransfer.files));
  };

  const onContextSubmit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      if (!sessionRef.current) {
        await ensureSession();
        setMessage("Inspection run created; recipe segment starts at 0.0 s");
        return;
      }
      const segment = await addRecipeSegment(sessionRef.current.session_id, {
        start_ms: Math.round(segmentStartSeconds * 1000),
        recipe: config.recipe,
      });
      setSession((current) => current ? { ...current, recipe: segment.recipe } : current);
      setMessage(`${segment.recipe.current_recipe_name} assigned from ${segmentStartSeconds.toFixed(1)} s`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recipe context could not be saved");
    }
  };

  const onRecipeImport = async (file?: File) => {
    if (!file) return;
    try {
      const active = await ensureSession();
      const segments = await importRecipeSegments(active.session_id, file);
      const latest = segments.at(-1);
      if (latest) {
        setSession((current) => current ? { ...current, recipe: latest.recipe } : current);
      }
      setMessage(`${segments.length} timestamped recipe segments imported`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recipe CSV import failed");
    } finally {
      if (recipeInputRef.current) recipeInputRef.current.value = "";
    }
  };

  const acknowledge = async () => {
    if (!session || !latestEvent) return;
    await acknowledgeInspectionEvent(session.session_id, latestEvent.event_id);
    setEvents((current) => current.map((item) => item.event_id === latestEvent.event_id
      ? { ...item, acknowledged_at: new Date().toISOString() }
      : item));
    setMessage("Operator acknowledgement persisted");
  };

  return (
    <div className="shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div><strong>LINEWATCH</strong><small>CHANGEOVER INTELLIGENCE</small></div>
        </Link>
        <nav className="main-nav"><Link className="active" to="/">MONITOR</Link><Link to="/history">RUN HISTORY</Link></nav>
        <div className="topbar-status">
          <span className={`connection-dot ${health ? "online" : ""}`} />
          <div>
            <strong>{health ? "SYSTEM ONLINE" : "BACKEND OFFLINE"}</strong>
            <small>{health ? `${health.persistence.toUpperCase()} · QUEUE ${health.queue_depth}` : "CHECK API ON :8443"}</small>
          </div>
        </div>
      </header>

      <main>
        <section className="page-intro">
          <div><span className="kicker">OPERATOR STATION / MULTI-SOURCE INSPECTION</span><h1>Changeover monitor</h1><p>Ingest live cameras, labeled image sets, or recordings while preserving recipe and source timing.</p></div>
          <div className={`state-chip state-${status}`}><span className="pulse" />{statusLabel[status]}</div>
        </section>

        <section className="source-mode-bar" aria-label="Inspection source">
          {(Object.keys(sourceLabels) as SourceMode[]).map((mode) => (
            <button className={sourceMode === mode ? "active" : ""} key={mode} type="button" onClick={() => selectSource(mode)}>
              <strong>{sourceLabels[mode].label}</strong><span>{sourceLabels[mode].detail}</span>
            </button>
          ))}
        </section>

        <section className="pipeline-strip" aria-label="Integration pipeline">
          {["SOURCE", "PERSIST + QUEUE", "MODEL ADAPTER", "DECISION", "OPERATOR"].map((stage, index) => (
            <div className="pipeline-stage" key={stage}><span>{String(index + 1).padStart(2, "0")}</span><strong>{stage}</strong>{index < 4 && <i aria-hidden="true">→</i>}</div>
          ))}
        </section>

        <div className="dashboard-grid">
          <section className="panel camera-panel">
            <div className="panel-heading">
              <div><span className="section-number">01</span><div><h2>Post-pipeline frame</h2><p>Persisted source, rendered frame, and mask artifacts</p></div></div>
              <div className="view-toggle"><button className={view === "output" ? "active" : ""} onClick={() => setView("output")} type="button">POST-PIPELINE</button><button className={view === "raw" ? "active" : ""} onClick={() => setView("raw")} type="button">SOURCE</button></div>
            </div>

            <div className="viewport">
              <video ref={videoRef} className={view === "raw" && isCameraLive ? "visible" : ""} autoPlay muted playsInline />
              {view === "output" && latestEvent && <div className="pipeline-visual"><img src={imageSource(resultImage)} alt="Latest post-pipeline frame" />{maskImage && <img className="mask-overlay" src={imageSource(maskImage)} alt="Liquid mask overlay" />}</div>}
              {!isCameraLive && (view === "raw" || !latestEvent) && <div className="empty-camera"><div className="vial-glyph"><span /></div><strong>NO FRAME SELECTED</strong><p>Start a camera or ingest images/recording frames.</p></div>}
              <div className="viewport-label"><span className={isCameraLive && view === "raw" ? "live-dot" : ""} />{view === "raw" ? sourceLabels[sourceMode].label : maskImage ? "POST-PIPELINE · MASK" : "POST-PIPELINE · SOURCE FALLBACK"}</div>
              <div className="frame-code">{latestEvent ? `${latestEvent.frame_id} · ${timestampLabel(latestEvent.source_timestamp_ms)}` : "WAITING FOR FRAME"}</div>
              {isProcessing && <div className="scan-line" />}
            </div>

            {sourceMode === "live_camera" ? (
              <div className="camera-actions">
                {!isCameraLive ? <button className="button primary" onClick={() => void startCamera()} type="button"><span className="button-icon">●</span> START CAMERA</button> : <button className="button quiet" onClick={stopCamera} type="button">STOP CAMERA</button>}
                <button className="button quiet" onClick={() => void sendGeneratedFrame(demoFrame)} disabled={isProcessing} type="button">SEND DEMO FRAME</button>
                <label className="sample-control"><span>SAMPLE SEC</span><input type="number" min="0.1" step="0.1" value={sampleSeconds} onChange={(event) => setSampleSeconds(Number(event.target.value))} /></label>
              </div>
            ) : (
              <div className={`drop-zone ${dragActive ? "active" : ""}`} onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragActive(false)} onDrop={onDrop} onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" hidden multiple={sourceMode === "images"} accept={sourceMode === "images" ? "image/jpeg,image/png,image/webp" : "video/mp4,video/webm,video/quicktime,.mov"} onClick={(event) => event.stopPropagation()} onChange={(event) => { const files = Array.from(event.target.files ?? []); event.target.value = ""; void handleFiles(files); }} />
                <strong>DROP {sourceMode === "images" ? "IMAGES" : "A RECORDING"} HERE</strong>
                <span>Drag files here or use the file picker · originals and pipeline outputs are retained</span>
                <button className="button quiet file-picker-button" type="button" onClick={(event) => { event.stopPropagation(); fileInputRef.current?.click(); }}>CHOOSE FILES</button>
                {sourceMode === "recording" && <label className="sample-control" onClick={(event) => event.stopPropagation()}><span>FRAME INTERVAL SEC</span><input type="number" min="0.1" step="0.1" value={sampleSeconds} onChange={(event) => setSampleSeconds(Number(event.target.value))} /></label>}
              </div>
            )}

            {ingestion && <div className="ingestion-progress"><div><strong>{ingestion.label}</strong><span>{ingestion.current} / {ingestion.total}</span></div><div className="progress-track"><span style={{ width: `${Math.round((ingestion.current / ingestion.total) * 100)}%` }} /></div>{isProcessing && <button type="button" onClick={() => { cancelIngestionRef.current = true; }}>STOP AFTER CURRENT FRAME</button>}</div>}

            <div className={`visual-note ${maskImage ? "ready" : "pending"}`}><strong>{maskImage ? "MASK ARTIFACT SAVED" : "MASK OUTPUT PENDING"}</strong><span>{maskImage ? "Source, rendered output, and mask are available from run history." : "The mock adapter persists source frames; the production adapter will add rendered and mask files."}</span></div>

            <form className="context-form" onSubmit={onContextSubmit}>
              <div className="context-title"><div><h3>Run and recipe timeline</h3><p>Every frame resolves the recipe segment active at its source timestamp.</p></div><span>{session ? `RUN ${session.session_id.slice(0, 8)}` : "NOT STARTED"}</span></div>
              <label><span>RUN NAME</span><input value={config.run_name} disabled={Boolean(session)} onChange={(event) => setConfig({ ...config, run_name: event.target.value })} /></label>
              <label><span>CAMERA ID</span><input value={config.camera_id} disabled={Boolean(session)} onChange={(event) => setConfig({ ...config, camera_id: event.target.value })} /></label>
              <label><span>SEGMENT START SEC</span><input type="number" min="0" step="0.1" value={segmentStartSeconds} onChange={(event) => setSegmentStartSeconds(Number(event.target.value))} /></label>
              <label><span>ACTIVE RECIPE ID</span><input value={config.recipe.current_recipe_id} onChange={(event) => setConfig({ ...config, recipe: { ...config.recipe, current_recipe_id: event.target.value } })} /></label>
              <label className="wide-field"><span>ACTIVE RECIPE NAME</span><input value={config.recipe.current_recipe_name} onChange={(event) => setConfig({ ...config, recipe: { ...config.recipe, current_recipe_name: event.target.value } })} /></label>
              <label><span>PREVIOUS RECIPE ID</span><input value={config.recipe.previous_recipe_id ?? ""} onChange={(event) => setConfig({ ...config, recipe: { ...config.recipe, previous_recipe_id: event.target.value } })} /></label>
              <label><span>VIAL INDEX</span><input type="number" min="1" value={config.recipe.vial_index} onChange={(event) => setConfig({ ...config, recipe: { ...config.recipe, vial_index: Number(event.target.value) } })} /></label>
              <label className="check-field"><input type="checkbox" checked={config.recipe.is_first_vial_after_change} onChange={(event) => setConfig({ ...config, recipe: { ...config.recipe, is_first_vial_after_change: event.target.checked } })} /><span>FIRST VIAL AFTER RECIPE CHANGE</span></label>
              <button className="button primary lock-button" type="submit">{session ? "ADD RECIPE SEGMENT" : "START INSPECTION RUN"}</button>
              <button className="button quiet" type="button" onClick={() => recipeInputRef.current?.click()}>IMPORT RECIPE CSV</button>
              <input ref={recipeInputRef} hidden type="file" accept=".csv,text/csv" onChange={(event) => void onRecipeImport(event.target.files?.[0])} />
              {session && <button className="button quiet" type="button" onClick={resetSession}>NEW RUN</button>}
            </form>
          </section>

          <aside className="side-column">
            <section className={`panel decision-panel decision-${status}`}><div className="panel-heading compact"><div><span className="section-number">02</span><div><h2>Operator decision</h2><p>Human-in-the-loop guidance</p></div></div><span className="mode-badge">{latestEvent?.pipeline_mode.toUpperCase() ?? "NO RESULT"}</span></div><div className="decision-body"><div className="decision-icon">{status === "failure" ? "!" : status === "warning" ? "?" : "✓"}</div><span className="decision-label">CURRENT STATUS</span><h2>{latestEvent?.decision.headline ?? "Awaiting inspection frame"}</h2><p>{latestEvent?.decision.detail ?? "Create a run and ingest a source to exercise the durable pipeline."}</p><div className="recommended-action"><span>OPERATOR ACTION</span><strong>{latestEvent?.decision.recommended_action ?? "Choose source and recipe context"}</strong></div>{latestEvent?.decision.status === "failure" && <button className="button acknowledge" type="button" disabled={Boolean(latestEvent.acknowledged_at)} onClick={() => void acknowledge()}>{latestEvent.acknowledged_at ? "ACKNOWLEDGED" : "ACKNOWLEDGE + REVIEW"}</button>}</div></section>

            <section className="panel progress-panel"><div className="panel-heading compact"><div><span className="section-number">03</span><div><h2>Recipe progress</h2><p>{latestEvent?.recipe.current_recipe_name ?? session?.recipe.current_recipe_name ?? "No active segment"}</p></div></div><strong className="progress-value">{latestEvent?.decision.recipe_progress_percent ?? 0}%</strong></div><div className="progress-content"><div className="progress-track"><span style={{ width: `${latestEvent?.decision.recipe_progress_percent ?? 0}%` }} /></div><div className="measurements">{(latestEvent?.decision.measurements ?? [{ label: "Vial", value: "Waiting", state: "neutral" as const }, { label: "Fill", value: "Waiting", state: "neutral" as const }, { label: "Color", value: "Waiting", state: "neutral" as const }]).map((measurement) => <div className={`measurement ${measurement.state}`} key={measurement.label}><span>{measurement.label.toUpperCase()}</span><strong>{measurement.value}</strong></div>)}</div><div className="model-note"><span>PIPELINE SOURCE</span><p>{latestEvent?.decision.model_state ?? "No frame processed"}</p></div></div></section>

            <section className="panel events-panel"><div className="panel-heading compact"><div><span className="section-number">04</span><div><h2>Recent frames</h2><p>Persisted newest first</p></div></div><span className="event-count">{events.length}</span></div><div className="event-list">{events.length === 0 ? <div className="empty-events">No persisted inspection frames yet.</div> : events.map((item) => <div className={`event-row ${item.decision.status}`} key={item.event_id}><time>{timestampLabel(item.source_timestamp_ms)}</time><span className="event-marker" /><div><strong>{item.decision.headline}</strong><p>{item.recipe.current_recipe_id} · {item.original_filename ?? item.frame_id}</p></div></div>)}</div><div className="message-bar">{message}</div>{session && <Link className="review-link" to={`/runs/${session.session_id}`}>OPEN FULL FRAME REVIEW →</Link>}</section>

            <section className="simulation-tools"><span>INTEGRATION TEST CONTROLS</span><div><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "warning")} disabled={isProcessing}>SIMULATE WARNING</button><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "failure")} disabled={isProcessing}>SIMULATE FAILURE</button></div><p>Clearly labeled test states; not model predictions.</p></section>
          </aside>
        </div>
      </main>

      <footer><div><strong>LINEWATCH · OPERATOR-IN-THE-LOOP PROTOTYPE</strong><span>Rockwell Automation + NVIDIA Innovation Lab · Fall 2026</span><span>FastAPI / React / SQLite · No PLC control</span></div><div className="team-credit"><strong>© 2026 Project Team</strong><span>Austin Koske · Alayna Ostler · Cameron Delpage · Zoe Kirkman · Srijan Kumar · Josiah Mathews · Hayden Fons · Reagan Burkemper</span></div></footer>
    </div>
  );
}
