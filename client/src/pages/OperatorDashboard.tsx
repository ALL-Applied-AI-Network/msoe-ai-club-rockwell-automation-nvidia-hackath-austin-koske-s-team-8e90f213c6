import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
} from "react";
import { Link } from "react-router-dom";
import { judgeVial, loadRecipeLibrary, recipeSummary, type RecipeLibrary, type VialVerdict } from "@/services/recipeCheck";
import type {
  CreateSessionRequest,
  HealthStatus,
  InspectionEvent,
  InspectionSession,
  OperatorStatus,
  SimulationMode,
  SourceMode,
  OperatorResponse,
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
  uploadReference,
  listInspectionEvents,
} from "@/services/inspectionService";

const initialConfig: CreateSessionRequest = {
  camera_id: "fillcam-1",
  station_name: "Demo station (no line connection)",
  run_name: `Inspection ${new Date().toLocaleDateString()}`,
  source_mode: "live_camera",
  source_name: null,
  recipe: {
    current_recipe_id: "unconfirmed",
    current_recipe_name: "Recipe not confirmed",
    previous_recipe_id: null,
    vial_index: 1,
    is_first_vial_after_change: false,
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
  if (value === null || value === undefined) return "NO SOURCE CLOCK";
  const seconds = value / 1000;
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function cameraFrame(video: HTMLVideoElement) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context || !width || !height) throw new Error("Camera is not ready");
  context.drawImage(video, 0, 0, width, height);
  return new Promise<{ blob: Blob; width: number; height: number }>((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob ? resolve({ blob, width, height }) : reject(new Error("Could not encode camera frame")),
      "image/png",
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

function waitForMedia(video: HTMLVideoElement, eventName: "loadedmetadata" | "loadeddata" | "seeked", signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const onDone = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("The recording could not be decoded")); };
    const onAbort = () => { cleanup(); reject(new DOMException("Run changed", "AbortError")); };
    const timeout = window.setTimeout(() => { cleanup(); reject(new Error("Recording decode timed out")); }, 15000);
    const cleanup = () => {
      video.removeEventListener(eventName, onDone);
      video.removeEventListener("error", onError);
      signal?.removeEventListener("abort", onAbort);
      window.clearTimeout(timeout);
    };
    video.addEventListener(eventName, onDone, { once: true });
    video.addEventListener("error", onError, { once: true });
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

interface VialRecord {
  number: number;
  recipeId: string;
  previousId: string | null;
  sessionId: string;
  eventId: string | null;
  verdict: VialVerdict;
  decision?: "confirmed" | "overridden";
}

/** Simulated line action. Nothing is sent to the PLC, sorter or Teledyne. */
function simulatedAction(vial: VialRecord): { lane: "GOOD" | "SCRAP" | "HOLD"; text: string } {
  const verdict = vial.verdict;
  if (verdict.kind === "match") return { lane: "GOOD", text: "Route to GOOD lane" };
  if (verdict.kind === "differs") return verdict.towardPrevious
    ? { lane: "SCRAP", text: `Divert to SCRAP · shifted toward recipe ${vial.previousId} (possible carryover); flush before next vial` }
    : { lane: "SCRAP", text: "Divert to SCRAP · color does not match the recipe" };
  return { lane: "HOLD", text: verdict.kind === "no_reference" ? "Hold for manual check · no reference color for this recipe" : "Hold for manual check · color could not be measured" };
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
  const [connected, setConnected] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [lastReceived, setLastReceived] = useState(0);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [referenceStatus, setReferenceStatus] = useState("Use a configured reference or choose an earlier aligned image");
  const [response, setResponse] = useState<OperatorResponse>({ reviewer: "", action: "reviewed", note: "" });
  const [, setReviewQueue] = useState<InspectionEvent[]>([]);
  const [library, setLibrary] = useState<RecipeLibrary | null>(null);
  const [nextRecipe, setNextRecipe] = useState("5");
  const [activeVial, setActiveVial] = useState<{ number: number; recipeId: string; previousId: string | null; startFrame: number } | null>(null);
  const [vialLog, setVialLog] = useState<VialRecord[]>([]);
  const [useClipReference, setUseClipReference] = useState(true);
  const activeVialRef = useRef(activeVial);
  activeVialRef.current = activeVial;
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
  const generationRef = useRef(0);
  const abortRef = useRef(new AbortController());
  const creatingRef = useRef<Promise<InspectionSession> | null>(null);
  const referenceInputRef = useRef<HTMLInputElement>(null);
  const acknowledgedIdsRef = useRef(new Set<string>());

  const latestEvent = events[0] ?? null;
  const sessionId = session?.session_id;
  const resultImage = latestEvent?.visuals?.rendered_frame ?? latestEvent?.image;
  const maskImage = latestEvent?.visuals?.mask;
  const sourceUnavailable = sourceMode === "live_camera" && Boolean(latestEvent) &&
    (!isCameraLive || !connected || !health || now - lastReceived > Math.max(5000, sampleSeconds * 3000));
  const status: OperatorStatus = processingError ? "unavailable" : sourceUnavailable
    ? (isCameraLive ? "unavailable" : "stopped") : isProcessing
    ? "processing"
    : latestEvent?.decision.status ?? session?.status ?? "stopped";
  const displayHeadline = processingError ? "Processing unavailable" : sourceUnavailable
    ? (isCameraLive ? "Live inspection unavailable" : "Live inspection stopped") : latestEvent?.decision.headline;
  const displayDetail = processingError ?? (sourceUnavailable
    ? "The frame below is historical. Resume a connected source before treating results as current."
    : latestEvent?.decision.detail);

  const mergeEvent = useCallback((event: InspectionEvent) => {
    if (event.session_id !== sessionRef.current?.session_id) return;
    setEvents((current) => [
      { ...event, acknowledged_at: event.acknowledged_at ?? current.find((item) => item.event_id === event.event_id)?.acknowledged_at },
      ...current.filter((item) => item.event_id !== event.event_id),
    ].sort((a, b) => b.frame_number - a.frame_number).slice(0, 40));
    setLastReceived(Date.now());
    setProcessingError(null);
    if (["warning", "failure"].includes(event.decision.status) && !event.acknowledged_at && !acknowledgedIdsRef.current.has(event.event_id)) {
      setReviewQueue((current) => [event, ...current.filter((item) => item.event_id !== event.event_id)].slice(0, 100));
    }
    setView("output");
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
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
    const generation = generationRef.current;
    const socket = new WebSocket(inspectionEventUrl(sessionId));
    socket.onopen = () => { if (generation === generationRef.current) { setConnected(true); setMessage("Live event channel connected"); } };
    socket.onmessage = (event) => { if (generation === generationRef.current) mergeEvent(JSON.parse(event.data) as InspectionEvent); };
    socket.onclose = () => { if (generation === generationRef.current) { setConnected(false); setMessage("Event channel disconnected; restart the run to reconnect"); } };
    return () => { socket.onclose = null; socket.close(); };
  }, [mergeEvent, sessionId]);

  const ensureSession = useCallback(async (sourceName?: string) => {
    if (sessionRef.current) return sessionRef.current;
    if (creatingRef.current) return creatingRef.current;
    const generation = generationRef.current;
    const request: CreateSessionRequest = {
      ...config,
      source_mode: sourceMode,
      source_name: sourceName || config.source_name,
    };
    const creating = createInspectionSession(request).then((created) => {
      if (generation !== generationRef.current) throw new DOMException("Run changed", "AbortError");
      sessionRef.current = created;
      setSession(created);
      setMessage(`Inspection run ${created.session_id.slice(0, 8)} started`);
      return created;
    });
    creatingRef.current = creating;
    try { return await creating; }
    finally { if (creatingRef.current === creating) creatingRef.current = null; }
  }, [config, sourceMode]);

  const submitBlob = useCallback(async (
    blob: Blob,
    width: number,
    height: number,
    sourceTimestampMs?: number,
    originalFilename?: string,
    simulate?: SimulationMode,
    sourceName?: string,
    generation = generationRef.current,
  ) => {
    if (generation !== generationRef.current) throw new DOMException("Run changed", "AbortError");
    const activeSession = await ensureSession(sourceName);
    if (generation !== generationRef.current) throw new DOMException("Run changed", "AbortError");
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
      abortRef.current.signal,
      sourceMode === "recording" ? "recording_seek" : sourceMode === "live_camera" ? "browser_elapsed" : "unknown",
    );
    if (generation !== generationRef.current || sessionRef.current?.session_id !== activeSession.session_id) return returned;
    mergeEvent(returned);
    setSession((current) => current?.session_id === activeSession.session_id ? {
      ...current,
      recipe: returned.recipe,
      status: returned.decision.status,
      frames_processed: returned.frame_number,
      updated_at: returned.created_at,
    } : current);
    return returned;
  }, [ensureSession, mergeEvent, sourceMode]);

  const sendGeneratedFrame = useCallback(async (
    source: () => Promise<{ blob: Blob; width: number; height: number }>,
    simulate?: SimulationMode,
  ) => {
    if (busyRef.current) return;
    const generation = generationRef.current;
    busyRef.current = true;
    setIsProcessing(true);
    try {
      const frame = await source();
      if (generation !== generationRef.current) return;
      const active = await ensureSession();
      const elapsed = Math.max(0, Date.now() - new Date(active.created_at).getTime());
      const returned = await submitBlob(frame.blob, frame.width, frame.height, elapsed, undefined, simulate, undefined, generation);
      if (generation === generationRef.current) setMessage(`Frame ${returned.frame_number} persisted and processed`);
    } catch (error) {
      if (generation === generationRef.current) setProcessingError(error instanceof Error ? error.message : "Frame processing failed");
    } finally {
      if (generation === generationRef.current) { busyRef.current = false; setIsProcessing(false); }
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

  useEffect(() => {
    if (!isCameraLive) return;
    const interval = Math.max(.1, Number.isFinite(sampleSeconds) ? sampleSeconds : 1.5) * 1000;
    const timer = window.setInterval(captureCamera, interval);
    timerRef.current = timer;
    return () => window.clearInterval(timer);
  }, [captureCamera, isCameraLive, sampleSeconds]);

  const startCamera = async () => {
    const generation = generationRef.current;
    try {
      await ensureSession("Attached browser camera");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (generation !== generationRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      stream.getTracks().forEach((track) => { track.onended = () => { stopCamera(); setProcessingError("Camera stream ended"); }; });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraLive(true);
      setView("raw");
      const intervalSeconds = Number.isFinite(sampleSeconds) && sampleSeconds > 0 ? sampleSeconds : 1.5;
      setMessage(`Camera live · sampling every ${intervalSeconds.toFixed(1)} seconds`);
      setProcessingError(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Camera could not start");
    }
  };

  const resetSession = useCallback(() => {
    generationRef.current += 1;
    abortRef.current.abort();
    abortRef.current = new AbortController();
    creatingRef.current = null;
    busyRef.current = false;
    stopCamera();
    cancelIngestionRef.current = true;
    sessionRef.current = null;
    frameNumberRef.current = 0;
    setSession(null);
    setEvents([]);
    setReviewQueue([]);
    acknowledgedIdsRef.current.clear();
    setIsProcessing(false);
    setProcessingError(null);
    setConnected(false);
    setLastReceived(0);
    setReferenceStatus("Use a configured reference or choose an earlier aligned image");
    setIngestion(null);
    setMessage("Ready for a new persisted inspection run");
  }, [stopCamera]);

  const selectSource = (mode: SourceMode) => {
    if (sessionRef.current) resetSession();
    setSourceMode(mode);
    setSampleSeconds(mode === "recording" ? 0.5 : 1.5);
    setConfig((current) => ({ ...current, source_mode: mode, source_name: null }));
    setMessage(`${sourceLabels[mode].label} source selected`);
  };

  const ingestImages = async (files: File[]) => {
    const generation = generationRef.current;
    const images = files.filter((file) => file.type.startsWith("image/"));
    if (!images.length) throw new Error("Choose JPEG, PNG, or WebP images");
    cancelIngestionRef.current = false;
    setIngestion({ label: "IMAGE BATCH", current: 0, total: images.length });
    setIsProcessing(true);
    try {
      await ensureSession(images.length === 1 ? images[0].name : `${images.length} image batch`);
      for (let index = 0; index < images.length; index += 1) {
        if (cancelIngestionRef.current || generation !== generationRef.current) break;
        const file = images[index];
        const { width, height } = await imageDimensions(file);
        await submitBlob(file, width, height, undefined, file.name, undefined, file.name, generation);
        if (generation === generationRef.current) setIngestion({ label: "IMAGE BATCH", current: index + 1, total: images.length });
      }
      if (generation === generationRef.current) setMessage(`${images.length} source image${images.length === 1 ? "" : "s"} persisted and queued`);
    } finally {
      if (generation === generationRef.current) setIsProcessing(false);
    }
  };

  const ingestRecording = async (file: File) => {
    if (!file.type.startsWith("video/") && !/\.(mp4|webm|mov)$/i.test(file.name)) {
      throw new Error("Choose an MP4, WebM, or MOV recording");
    }
    const generation = generationRef.current;
    const signal = abortRef.current.signal;
    cancelIngestionRef.current = false;
    setIsProcessing(true);
    let objectUrl = "";
    try {
      const active = await ensureSession(file.name);
      if (generation !== generationRef.current) return;
      setMessage("Persisting original recording…");
      await uploadRecording(active.session_id, file, signal);
      objectUrl = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.preload = "auto";
      video.muted = true;
      const firstFrame = waitForMedia(video, "loadeddata", signal);
      video.src = objectUrl;
      await firstFrame;
      if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error("Recording duration is unavailable");
      if (useClipReference && health?.pipeline_mode === "model") {
        // The vial is empty before dosing starts: use the clip's first frame as the temporal reference.
        const still = document.createElement("canvas");
        still.width = video.videoWidth; still.height = video.videoHeight;
        still.getContext("2d")?.drawImage(video, 0, 0);
        const blob = await new Promise<Blob>((resolve, reject) => still.toBlob((value) => value ? resolve(value) : reject(new Error("Could not encode reference frame")), "image/png"));
        const result = await uploadReference(active.session_id, new File([blob], `${file.name}-first-frame.png`, { type: "image/png" }), signal);
        setReferenceStatus(`First frame of ${file.name} · ${result.status}`);
      }
      const intervalMs = Math.max(100, Math.round((Number.isFinite(sampleSeconds) && sampleSeconds > 0 ? sampleSeconds : 1.5) * 1000));
      const total = Math.max(1, Math.floor(video.duration * 1000 / intervalMs) + 1);
      const width = video.videoWidth, height = video.videoHeight;
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context || !width || !height) throw new Error("Recording dimensions are unavailable");
      setIngestion({ label: file.name, current: 0, total });
      for (let index = 0; index < total; index += 1) {
        if (cancelIngestionRef.current || generation !== generationRef.current) break;
        const targetMs = Math.min(index * intervalMs, Math.max(0, Math.floor(video.duration * 1000) - 1));
        if (index > 0) {
          const seeked = waitForMedia(video, "seeked", signal);
          video.currentTime = targetMs / 1000;
          await seeked;
        }
        context.drawImage(video, 0, 0, width, height);
        const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
          (value) => value ? resolve(value) : reject(new Error("Could not encode video frame")), "image/png"));
        const timestampMs = Math.round(video.currentTime * 1000);
        await submitBlob(blob, width, height, timestampMs, `${file.name}@${timestampMs}ms.png`, undefined, file.name, generation);
        if (generation === generationRef.current) setIngestion({ label: file.name, current: index + 1, total });
      }
      if (generation === generationRef.current) setMessage(cancelIngestionRef.current ? "Recording ingestion stopped" : `Recording complete · ${total} sampled frames persisted (browser seek clock)`);
      if (generation === generationRef.current && !cancelIngestionRef.current) await closeVial(active.session_id);
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (generation === generationRef.current) setIsProcessing(false);
    }
  };

  const handleFiles = async (files: File[]) => {
    if (busyRef.current) return;
    if (sourceMode === "recording" && sessionRef.current?.frames_processed) resetSession();
    const generation = generationRef.current;
    busyRef.current = true;
    setProcessingError(null);
    try {
      if (sourceMode === "images") await ingestImages(files);
      else if (sourceMode === "recording" && files[0]) await ingestRecording(files[0]);
      else setMessage("Select IMAGES or RECORDING before dropping files");
    } catch (error) {
      if (generation === generationRef.current) {
        setIsProcessing(false);
        setProcessingError(error instanceof Error ? error.message : "Ingestion failed");
      }
    } finally {
      if (generation === generationRef.current) busyRef.current = false;
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

  useEffect(() => {
    void loadRecipeLibrary().then(setLibrary).catch(() => setLibrary(null));
  }, []);

  const lastRecipeId = vialLog[0]?.recipeId ?? null;

  const setRecipeForNextVial = () => {
    // Carryover is judged against the last *different* recipe, which may be several vials back.
    const priorRecipe = vialLog.find((vial) => vial.recipeId !== nextRecipe)?.recipeId ?? null;
    const vial = { number: (vialLog[0]?.number ?? 0) + 1, recipeId: nextRecipe, previousId: priorRecipe,
      startFrame: sourceMode === "live_camera" && sessionRef.current ? (latestEvent?.frame_number ?? 0) + 1 : 1 };
    setActiveVial(vial);
    const name = library?.recipes.find((recipe) => recipe.id === nextRecipe)?.name ?? `Recipe ${nextRecipe}`;
    setConfig((current) => ({ ...current, recipe: { ...current.recipe, current_recipe_id: nextRecipe, current_recipe_name: name,
      previous_recipe_id: priorRecipe, vial_index: vial.number, is_first_vial_after_change: Boolean(lastRecipeId && lastRecipeId !== nextRecipe) } }));
    setMessage(`Vial ${vial.number}: recipe ${nextRecipe} received (manual entry simulating the recipe feed)`);
  };

  const closeVial = async (vialSessionId: string) => {
    const vial = activeVialRef.current;
    if (!vial || !library) return;
    const saved = await listInspectionEvents(vialSessionId, 500);
    const vialEvents = saved.filter((event) => event.frame_number >= vial.startFrame);
    const newest = [...vialEvents].sort((a, b) => b.frame_number - a.frame_number)[0];
    const record: VialRecord = { ...vial, sessionId: vialSessionId, eventId: newest?.event_id ?? null,
      verdict: judgeVial(vialEvents, library, vial.recipeId, vial.previousId) };
    setVialLog((current) => [record, ...current]);
    setActiveVial(null);
    const next = library.recipes.findIndex((recipe) => recipe.id === vial.recipeId);
    setMessage(`Vial ${vial.number} finished · ${simulatedAction(record).lane} (simulated)`);
    if (next >= 0) setNextRecipe(vial.recipeId);
  };

  const decideVial = async (vial: VialRecord, decision: "confirmed" | "overridden") => {
    if (!response.reviewer.trim() || !vial.eventId) return;
    const action = simulatedAction(vial);
    const body: OperatorResponse = { reviewer: response.reviewer,
      action: decision === "overridden" ? "investigate" : action.lane === "GOOD" ? "reviewed" : "hold_for_review",
      note: `Vial ${vial.number} recipe ${vial.recipeId}: ${decision} simulated ${action.lane}${response.note ? ` · ${response.note}` : ""}` };
    try {
      await acknowledgeInspectionEvent(vial.sessionId, vial.eventId, body);
      setVialLog((current) => current.map((item) => item.number === vial.number ? { ...item, decision } : item));
      setMessage(`Vial ${vial.number}: operator ${decision} the simulated action (saved; nothing sent to the line)`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save the decision");
    }
  };

  const acknowledge = async (event = latestEvent) => {
    if (!session || !event || !response.reviewer.trim()) return;
    const generation = generationRef.current;
    try {
      await acknowledgeInspectionEvent(session.session_id, event.event_id, response);
      if (generation !== generationRef.current) return;
      acknowledgedIdsRef.current.add(event.event_id);
      setEvents((current) => current.map((item) => item.event_id === event.event_id
        ? { ...item, acknowledged_at: new Date().toISOString(), acknowledgement: response } : item));
      setReviewQueue((current) => current.filter((item) => item.event_id !== event.event_id));
      setMessage("Operator response persisted; this does not approve product release");
    } catch (error) {
      if (generation === generationRef.current) setMessage(error instanceof Error ? error.message : "Could not save response");
    }
  };

  const installReference = async (file?: File) => {
    if (!file || busyRef.current) return;
    const generation = generationRef.current;
    busyRef.current = true;
    stopCamera();
    try {
      const active = await ensureSession();
      if (generation !== generationRef.current) return;
      const result = await uploadReference(active.session_id, file, abortRef.current.signal);
      if (generation === generationRef.current) { setReferenceStatus(`${file.name} · ${result.status}`); setEvents([]); }
    } catch (error) {
      if (generation === generationRef.current) setReferenceStatus(error instanceof Error ? error.message : "Reference upload failed");
    } finally {
      if (generation === generationRef.current) busyRef.current = false;
    }
  };

  useEffect(() => {
    if (!sessionId) return;
    let active = true;
    void listInspectionEvents(sessionId, 100, undefined, true).then((items) => {
      if (active) setReviewQueue((current) => [...current, ...items.filter((item) => !current.some((event) => event.event_id === item.event_id))].filter((item) => !acknowledgedIdsRef.current.has(item.event_id)));
    }).catch(() => undefined);
    return () => { active = false; };
  }, [sessionId]);

  const measurements = latestEvent?.decision.measurements ?? [];

  return (
    <div className="shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div><strong>LINEWATCH</strong><small>VIAL FILL INSPECTION</small></div>
        </Link>
        <nav className="main-nav"><Link className="active" to="/">MONITOR</Link><Link to="/recipe-check">RECIPE CHECK</Link><Link to="/history">RUN HISTORY</Link></nav>
        <div className="topbar-status">
          <span className={`connection-dot ${health ? "online" : ""}`} />
          <div>
            <strong>{health ? "SYSTEM ONLINE" : "BACKEND OFFLINE"}</strong>
            <small>{health ? `${health.pipeline_mode === "model" ? "MODELS LOADED" : "MOCK MODE"} · QUEUE ${health.queue_depth}` : "CHECK THE API SERVICE"}</small>
          </div>
        </div>
      </header>

      <main>
        <section className="page-intro simple-intro">
          <div><h1>Fill monitor</h1><ol className="steps"><li><b>1</b> Enter the next vial&apos;s recipe</li><li><b>2</b> Fill it (camera or recording)</li><li><b>3</b> Act on the vial result</li></ol></div>
          <div className="intro-actions"><div className={`state-chip state-${status}`}><span className="pulse" />{statusLabel[status]}</div>{session && <button className="button quiet" type="button" onClick={resetSession}>NEW RUN</button>}</div>
        </section>

        <section className="recipe-card" aria-label="Next vial recipe">
          <div className="recipe-card-title"><span className="section-number">1</span><div><h2>Next vial recipe</h2><p>Entered by the operator before filling · stands in for a live recipe feed from the line</p></div></div>
          <label><span>RECIPE</span><select value={nextRecipe} disabled={isProcessing} onChange={(event) => setNextRecipe(event.target.value)}>{(library?.recipes ?? []).map((recipe) => <option key={recipe.id} value={recipe.id}>{recipe.name} · {recipeSummary(recipe)}{recipe.reference ? "" : " · no reference color"}</option>)}</select></label>
          <div className="recipe-previous"><span>LAST VIAL</span><strong>{lastRecipeId ? `Recipe ${lastRecipeId}` : "—"}</strong></div>
          <button className="button primary" type="button" disabled={isProcessing || !library} onClick={setRecipeForNextVial}>SET RECIPE</button>
          <div className={`recipe-active ${activeVial ? "on" : ""}`}>{activeVial ? <><strong>Vial {activeVial.number} · recipe {activeVial.recipeId}</strong><span>{lastRecipeId && lastRecipeId !== activeVial.recipeId ? `Recipe change from ${lastRecipeId}` : activeVial.previousId ? `Same recipe · carryover checked against recipe ${activeVial.previousId}` : "First vial"} · now fill it</span></> : <span>No vial armed · set a recipe first</span>}</div>
        </section>

        <section className="source-mode-bar" aria-label="Inspection source">
          {(Object.keys(sourceLabels) as SourceMode[]).map((mode) => (
            <button className={sourceMode === mode ? "active" : ""} key={mode} type="button" onClick={() => selectSource(mode)}>
              <strong>{sourceLabels[mode].label}</strong><span>{sourceLabels[mode].detail}</span>
            </button>
          ))}
        </section>

        <div className="dashboard-grid">
          <section className="panel camera-panel">
            <div className="panel-heading">
              <div><span className="section-number">2</span><div><h2>Fill</h2><p>Green = liquid the model proposes · blue = area actually sampled for color</p></div></div>
              <div className="view-toggle"><button className={view === "output" ? "active" : ""} onClick={() => setView("output")} type="button">POST-PIPELINE</button><button className={view === "raw" ? "active" : ""} onClick={() => setView("raw")} type="button">SOURCE</button></div>
            </div>

            <div className="viewport">
              <video ref={videoRef} className={view === "raw" && isCameraLive ? "visible" : ""} autoPlay muted playsInline />
              {latestEvent && (view === "output" || !isCameraLive) && <div className="pipeline-visual"><img src={imageSource(view === "raw" ? latestEvent.image : resultImage)} alt={view === "raw" ? "Original source frame" : "Latest post-pipeline frame"} />{view === "output" && maskImage && !latestEvent.visuals?.rendered_frame && <img className="mask-overlay" src={imageSource(maskImage)} alt="Liquid mask overlay" />}</div>}
              {!isCameraLive && !latestEvent && <div className="empty-camera"><div className="vial-glyph"><span /></div><strong>NO FRAME YET</strong><p>{sourceMode === "live_camera" ? "Press START CAMERA below." : `Drop ${sourceMode === "images" ? "images" : "a recording"} below to begin.`}</p></div>}
              <div className="viewport-label"><span className={isCameraLive && view === "raw" ? "live-dot" : ""} />{view === "raw" ? "ORIGINAL PIXELS" : "MODEL VIEW"}</div>
              <div className="frame-code">{latestEvent ? `Frame ${latestEvent.frame_number} · ${timestampLabel(latestEvent.source_timestamp_ms)}` : "WAITING FOR FRAME"}</div>
            </div>

            {sourceMode === "live_camera" ? (
              <div className="camera-actions">
                {!isCameraLive ? <button className="button primary" onClick={() => void startCamera()} type="button"><span className="button-icon">●</span> START CAMERA</button> : <button className="button quiet" onClick={stopCamera} type="button">STOP CAMERA</button>}
                {activeVial && session && <button className="button primary" type="button" onClick={() => void closeVial(session.session_id)}>FINISH VIAL {activeVial.number}</button>}
                {health?.pipeline_mode === "mock" && <button className="button quiet" onClick={() => void sendGeneratedFrame(demoFrame)} disabled={isProcessing} type="button">SEND DEMO FRAME</button>}
              </div>
            ) : (
              <div className={`drop-zone ${dragActive ? "active" : ""}`} onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragActive(false)} onDrop={onDrop} onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" hidden multiple={sourceMode === "images"} accept={sourceMode === "images" ? "image/jpeg,image/png,image/webp" : "video/mp4,video/webm,video/quicktime,.mov"} onClick={(event) => event.stopPropagation()} onChange={(event) => { const files = Array.from(event.target.files ?? []); event.target.value = ""; void handleFiles(files); }} />
                <strong>DROP {sourceMode === "images" ? "IMAGES" : "ONE VIAL'S RECORDING"} HERE</strong>
                <span>{activeVial ? `Fills vial ${activeVial.number} (recipe ${activeVial.recipeId})` : "Set the next vial's recipe first"} · or click to choose a file</span>
              </div>
            )}

            {ingestion && <div className="ingestion-progress"><div><strong>{ingestion.label}</strong><span>{ingestion.current} / {ingestion.total}</span></div><div className="progress-track"><span style={{ width: `${Math.round((ingestion.current / ingestion.total) * 100)}%` }} /></div>{isProcessing && <button type="button" onClick={() => { cancelIngestionRef.current = true; }}>STOP AFTER CURRENT FRAME</button>}</div>}
            <div className="message-bar">{message}</div>

            <details className="advanced-setup">
              <summary>Advanced setup <span>reference image · sampling · raw QA evidence · recipe timeline</span></summary>
              <div className="advanced-body">
                <label className="check-field"><input type="checkbox" checked={useClipReference} onChange={(event) => setUseClipReference(event.target.checked)} /><span>Use each recording&apos;s first frame (empty vial) as the reference</span></label>
                <label className="sample-control"><span>{sourceMode === "live_camera" ? "SAMPLE SEC" : "FRAME INTERVAL SEC"}</span><input type="number" min="0.1" step="0.1" value={sampleSeconds} onChange={(event) => setSampleSeconds(Number(event.target.value))} /></label>
                <section className="evidence-controls">
                  <button className="button quiet" type="button" disabled={isProcessing || health?.pipeline_mode !== "model"} onClick={() => referenceInputRef.current?.click()}>CHOOSE REFERENCE IMAGE</button>
                  <input ref={referenceInputRef} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { void installReference(event.target.files?.[0]); event.target.value = ""; }} />
                  <p>{referenceStatus}. Upload alone does not certify an empty vial or approve a recipe.</p>
                  {latestEvent?.decision.evidence && <div className="qa-evidence"><strong>MEASUREMENT EVIDENCE</strong><p>Proposal pixels: {String(latestEvent.decision.evidence.proposal_pixels ?? "—")} · Sampling pixels: {String(latestEvent.decision.evidence.sampling_pixels ?? "—")}</p><p>{latestEvent.decision.evidence.blocked_reasons?.join(" · ") ?? latestEvent.decision.evidence.quality_flags?.join(" · ")}</p></div>}
                </section>
                <form className="context-form" onSubmit={onContextSubmit}>
                  <div className="context-title"><div><h3>Run and recipe timeline</h3><p>Every frame is tagged with the recipe segment active at its timestamp.</p></div><span>{session ? `RUN ${session.session_id.slice(0, 8)}` : "NOT STARTED"}</span></div>
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
                </form>
              </div>
            </details>
          </section>

          <aside className="side-column">
            <section className={`panel decision-panel decision-${status}`}><div className="panel-heading compact"><div><div><h2>Latest frame</h2><p>What the model sees right now</p></div></div><span className="mode-badge">{latestEvent?.pipeline_mode.toUpperCase() ?? "NO RESULT"}</span></div><div className="decision-body"><h2>{displayHeadline ?? "Waiting for a frame"}</h2><p>{displayDetail ?? "Choose a source to start."}</p>{measurements.length > 0 && <div className="measurements">{measurements.map((measurement) => <div className={`measurement ${measurement.state}`} key={measurement.label}><span>{measurement.label.toUpperCase()}</span><strong>{measurement.value}</strong></div>)}</div>}<div className="recommended-action"><span>OPERATOR ACTION</span><strong>{latestEvent?.decision.recommended_action ?? "Choose a source"}</strong></div>{latestEvent?.decision.status === "failure" && <button className="button acknowledge" type="button" disabled={Boolean(latestEvent.acknowledged_at) || !response.reviewer.trim()} onClick={() => void acknowledge()}>{latestEvent.acknowledged_at ? "ACKNOWLEDGED" : "ACKNOWLEDGE + REVIEW"}</button>}</div></section>

            <section className="panel vial-panel"><h2><span className="section-number">3</span> Vial results</h2>
              <p className="vial-panel-note">Simulated line actions · nothing is sent to the PLC, sorter or Teledyne. Today&apos;s sorter is not recipe-aware.</p>
              <label>OPERATOR<input value={response.reviewer} placeholder="Your name (to confirm or override)" onChange={(e) => setResponse({ ...response, reviewer: e.target.value })} /></label>
              {vialLog.length === 0 ? <p className="vial-empty">No finished vials yet.</p> : vialLog.map((vial) => {
                const action = simulatedAction(vial);
                const verdict = vial.verdict;
                return <article className={`vial-result lane-${action.lane.toLowerCase()}`} key={vial.number}>
                  <header><strong>Vial {vial.number} · recipe {vial.recipeId}</strong><span className="vial-chip">{verdict.kind === "match" ? "IN SPEC" : verdict.kind === "differs" ? "OUT OF SPEC" : "CAN'T VERIFY"}</span></header>
                  <p>{verdict.kind === "match" || verdict.kind === "differs" ? `ΔE ${verdict.deltaE.toFixed(1)} to recipe ${vial.recipeId} (band ${library?.band_de2000 ?? 5})${verdict.deltaEPrevious != null ? ` · ΔE ${verdict.deltaEPrevious.toFixed(1)} to recipe ${vial.previousId}` : ""} · ${verdict.samples} samples` : verdict.kind === "no_reference" ? `Measured (${verdict.samples} samples) but recipe ${vial.recipeId} has no reference color` : "QA could not measure the finished color"}</p>
                  <div className="vial-action"><span>SIMULATED ACTION</span><strong>{action.text}</strong></div>
                  {vial.decision ? <p className="vial-decided">Operator {vial.decision} · saved</p> : <div className="vial-buttons"><button type="button" disabled={!response.reviewer.trim() || !vial.eventId} onClick={() => void decideVial(vial, "confirmed")}>CONFIRM</button><button type="button" disabled={!response.reviewer.trim() || !vial.eventId} onClick={() => void decideVial(vial, "overridden")}>OVERRIDE</button></div>}
                </article>;
              })}
            </section>

            <section className="panel events-panel"><div className="panel-heading compact"><div><div><h2>Recent frames</h2><p>Latest 40 · newest first</p></div></div><span className="event-count">{events.length}</span></div><div className="event-list">{events.length === 0 ? <div className="empty-events">No frames yet.</div> : events.map((item) => <div className={`event-row ${item.decision.status}`} key={item.event_id}><time>{timestampLabel(item.source_timestamp_ms)}</time><span className="event-marker" /><div><strong>{item.decision.headline}</strong><p>{item.recipe.current_recipe_id} · {item.original_filename ?? item.frame_id}</p></div></div>)}</div>{session && <Link className="review-link" to={`/runs/${session.session_id}`}>OPEN FULL FRAME REVIEW →</Link>}</section>

            {health?.pipeline_mode === "mock" && <section className="simulation-tools"><span>INTEGRATION TEST CONTROLS</span><div><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "warning")} disabled={isProcessing}>SIMULATE WARNING</button><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "failure")} disabled={isProcessing}>SIMULATE FAILURE</button></div><p>Clearly labeled test states; not model predictions.</p></section>}
          </aside>
        </div>

      </main>

      <footer><div><strong>LINEWATCH · OPERATOR-IN-THE-LOOP PROTOTYPE</strong><span>Rockwell Automation + NVIDIA Innovation Lab · Fall 2026</span><span>FastAPI / React / SQLite · No PLC control</span></div><div className="team-credit"><strong>© 2026 Project Team</strong><span>Austin Koske · Alayna Ostler · Cameron Delpage · Zoe Kirkman · Srijan Kumar · Josiah Mathews · Hayden Fons · Reagan Burkemper</span></div></footer>
    </div>
  );
}
