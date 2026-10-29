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
  RecipeContext,
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
  replaceRecipePlan,
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
    current_recipe_name: "Recipe 07 · Pale Blue",
    previous_recipe_id: "recipe-06",
    vial_index: 1,
    is_first_vial_after_change: true,
    expected_color_name: "Pale blue",
    expected_color_hex: "#287fb5",
    t100_ml: 20,
    t200_ml: 0,
    t300_ml: 0,
    t400_ml: 50,
    total_ml: 70,
    dispense_order: "T400>T100",
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

type RecipeImportState = {
  fileName: string;
  state: "staged" | "imported";
  segmentCount?: number;
};

function normalizeRecipePlan(recipes: RecipeContext[]): RecipeContext[] {
  return recipes.map((recipe, index) => {
    const previous = recipes[index - 1];
    return {
      ...recipe,
      vial_index: index + 1,
      previous_recipe_id: previous?.current_recipe_id ?? null,
      is_first_vial_after_change: !previous
        || previous.current_recipe_id !== recipe.current_recipe_id,
    };
  });
}

type TankAmountField = "t100_ml" | "t200_ml" | "t300_ml" | "t400_ml";

const tankAmountFields: { field: TankAmountField; label: string }[] = [
  { field: "t100_ml", label: "TANK 100" },
  { field: "t200_ml", label: "TANK 200" },
  { field: "t300_ml", label: "TANK 300" },
  { field: "t400_ml", label: "TANK 400" },
];

function formulaTotal(recipe: RecipeContext): number {
  return (recipe.t100_ml ?? 0)
    + (recipe.t200_ml ?? 0)
    + (recipe.t300_ml ?? 0)
    + (recipe.t400_ml ?? 0);
}

function formatMl(value?: number | null): string {
  return (value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

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
  const [recipeImport, setRecipeImport] = useState<RecipeImportState | null>(null);
  const [recipePlan, setRecipePlan] = useState<RecipeContext[]>([
    { ...initialConfig.recipe },
  ]);
  const [activePlanIndex, setActivePlanIndex] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const frameNumberRef = useRef(0);
  const busyRef = useRef(false);
  const sessionRef = useRef<InspectionSession | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recipeInputRef = useRef<HTMLInputElement>(null);
  const cancelIngestionRef = useRef(false);
  const pendingRecipeFileRef = useRef<File | null>(null);

  const latestEvent = events[0] ?? null;
  const modelEvidenceOnly = latestEvent?.pipeline_mode === "model"
    || (!latestEvent && health?.pipeline_mode === "model");
  const sessionId = session?.session_id;
  const resultImage = latestEvent?.visuals?.rendered_frame ?? latestEvent?.image;
  const maskImage = latestEvent?.visuals?.mask;
  const status: OperatorStatus = isProcessing
    ? "processing"
    : latestEvent?.decision.status ?? session?.status ?? "stopped";

  const applyRecipePlan = useCallback((recipes: RecipeContext[]) => {
    const normalized = normalizeRecipePlan(recipes);
    setRecipePlan(normalized);
    if (!sessionRef.current && normalized[0]) {
      setConfig((current) => ({ ...current, recipe: normalized[0] }));
    }
  }, []);

  const updatePlanRecipe = (index: number, patch: Partial<RecipeContext>) => {
    applyRecipePlan(recipePlan.map((recipe, recipeIndex) => (
      recipeIndex === index ? { ...recipe, ...patch } : recipe
    )));
  };

  const updateTankAmount = (
    index: number,
    field: TankAmountField,
    rawValue: string,
  ) => {
    const amount = Math.max(0, Number.parseFloat(rawValue) || 0);
    const recipe = { ...recipePlan[index], [field]: amount };
    updatePlanRecipe(index, { [field]: amount, total_ml: formulaTotal(recipe) });
  };

  const movePlanRecipe = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= recipePlan.length) return;
    const reordered = [...recipePlan];
    [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
    applyRecipePlan(reordered);
  };

  const addPlanRecipe = () => {
    const last = recipePlan.at(-1);
    applyRecipePlan([
      ...recipePlan,
      {
        current_recipe_id: last?.current_recipe_id ?? "recipe-new",
        current_recipe_name: last?.current_recipe_name ?? "New recipe",
        previous_recipe_id: last?.current_recipe_id ?? null,
        vial_index: recipePlan.length + 1,
        is_first_vial_after_change: false,
        expected_color_name: last?.expected_color_name ?? "Not set",
        expected_color_hex: last?.expected_color_hex ?? "#82918f",
        t100_ml: last?.t100_ml ?? 0,
        t200_ml: last?.t200_ml ?? 0,
        t300_ml: last?.t300_ml ?? 0,
        t400_ml: last?.t400_ml ?? 0,
        total_ml: last?.total_ml ?? 0,
        dispense_order: last?.dispense_order ?? null,
      },
    ]);
  };

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

  // While a run is active, keep its stored order current so the LINE view follows new vials.
  useEffect(() => {
    if (!session) return;
    const timer = window.setTimeout(() => {
      void replaceRecipePlan(session.session_id, normalizeRecipePlan(recipePlan)).catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [recipePlan, session]);

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
    const planForRun = normalizeRecipePlan(recipePlan);
    const request: CreateSessionRequest = {
      ...config,
      source_mode: sourceMode,
      source_name: sourceName || config.source_name,
      recipe: planForRun[activePlanIndex] ?? planForRun[0] ?? config.recipe,
    };
    const created = await createInspectionSession(request);
    await replaceRecipePlan(created.session_id, planForRun);
    sessionRef.current = created;
    setSession(created);
    const pendingRecipe = pendingRecipeFileRef.current;
    if (!pendingRecipe) {
      setMessage(`Inspection run ${created.session_id.slice(0, 8)} started · ${planForRun.length} planned vial${planForRun.length === 1 ? "" : "s"}`);
      return created;
    }

    const segments = await importRecipeSegments(created.session_id, pendingRecipe);
    const importedPlan = segments.length
      ? normalizeRecipePlan(segments.map((segment) => segment.recipe))
      : planForRun;
    await replaceRecipePlan(created.session_id, importedPlan);
    setRecipePlan(importedPlan);
    setActivePlanIndex(0);
    const active = importedPlan[0] ? { ...created, recipe: importedPlan[0] } : created;
    pendingRecipeFileRef.current = null;
    sessionRef.current = active;
    setSession(active);
    setRecipeImport({
      fileName: pendingRecipe.name,
      state: "imported",
      segmentCount: segments.length,
    });
    setMessage(
      `Run ${created.session_id.slice(0, 8)} started · ${segments.length} recipe segments and ${importedPlan.length} planned vials imported`,
    );
    return active;
  }, [activePlanIndex, config, recipePlan, sourceMode]);

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
    pendingRecipeFileRef.current = null;
    setRecipeImport(null);
    setActivePlanIndex(0);
    setMessage("Ready for a new persisted inspection run");
  }, [stopCamera]);

  const selectSource = (mode: SourceMode) => {
    if (sessionRef.current) {
      setMessage("Source is locked to this run. Choose NEW RUN before changing source.");
      return;
    }
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
      if (!sessionRef.current) {
        pendingRecipeFileRef.current = file;
        setRecipeImport({ fileName: file.name, state: "staged" });
        setMessage(`${file.name} staged · it will attach when this run starts`);
        return;
      }
      const active = sessionRef.current;
      const segments = await importRecipeSegments(active.session_id, file);
      if (segments.length) {
        const importedPlan = normalizeRecipePlan(segments.map((segment) => segment.recipe));
        await replaceRecipePlan(active.session_id, importedPlan);
        setRecipePlan(importedPlan);
        setActivePlanIndex(0);
        const updated = { ...active, recipe: importedPlan[0] };
        sessionRef.current = updated;
        setSession(updated);
        setConfig((current) => ({ ...current, recipe: importedPlan[0] }));
      }
      setRecipeImport({ fileName: file.name, state: "imported", segmentCount: segments.length });
      setMessage(segments.length
        ? `${segments.length} timestamped recipe segments imported into run ${active.session_id.slice(0, 8)}`
        : `${file.name} was already imported; no duplicate segments were added`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Recipe CSV import failed");
    } finally {
      if (recipeInputRef.current) recipeInputRef.current.value = "";
    }
  };

  const activatePlanRecipe = async (index: number) => {
    const recipe = recipePlan[index];
    if (!recipe) return;
    try {
      const active = await ensureSession();
      const elapsed = Math.max(0, Date.now() - new Date(active.created_at).getTime());
      const lastSourceTime = latestEvent?.source_timestamp_ms;
      const startMs = lastSourceTime === null || lastSourceTime === undefined
        ? elapsed
        : lastSourceTime + 1;
      const segment = await addRecipeSegment(active.session_id, {
        start_ms: startMs,
        recipe,
      });
      const updated = { ...active, recipe: segment.recipe };
      sessionRef.current = updated;
      setSession(updated);
      setConfig((current) => ({ ...current, recipe: segment.recipe }));
      setActivePlanIndex(index);
      setSegmentStartSeconds(startMs / 1000);
      setMessage(`Vial ${recipe.vial_index} activated · ${recipe.current_recipe_name} · ${recipe.expected_color_name ?? "color target not set"}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Planned vial could not be activated");
    }
  };

  const choosePlanRecipe = (index: number) => {
    const recipe = recipePlan[index];
    if (!recipe) return;
    if (sessionRef.current) {
      void activatePlanRecipe(index);
      return;
    }
    setActivePlanIndex(index);
    setConfig((current) => ({ ...current, recipe }));
    setMessage(`Vial ${recipe.vial_index} selected as the starting plan step`);
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
            <small>{health ? `${health.pipeline_mode.toUpperCase()} · ${health.pipeline_status.replaceAll("_", " ").toUpperCase()} · QUEUE ${health.queue_depth}` : "CHECK API ON :8443"}</small>
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
            <button className={sourceMode === mode ? "active" : ""} key={mode} type="button" disabled={Boolean(session)} title={session ? "Choose NEW RUN before changing source" : undefined} onClick={() => selectSource(mode)}>
              <strong>{sourceLabels[mode].label}</strong><span>{sourceLabels[mode].detail}</span>
            </button>
          ))}
        </section>

        <section className="pipeline-strip" aria-label="Integration pipeline">
          {["SOURCE + CLOCK", "PERSIST + QUEUE", "MODEL BRIDGE", "QA + ARTIFACTS", "OPERATOR REVIEW"].map((stage, index) => (
            <div className="pipeline-stage" key={stage}><span>{String(index + 1).padStart(2, "0")}</span><strong>{stage}</strong>{index < 4 && <i aria-hidden="true">→</i>}</div>
          ))}
        </section>

        <section className="architecture-summary" aria-label="Runtime architecture">
          <div><span>SHARED MODEL CORE</span><strong>integration/model_pipeline.py</strong><p>Srijan integration consolidated here: YOLO vial geometry → paired Temporal CNN liquid proposal → association → measurement QA.</p></div>
          <div><span>API BRIDGE</span><strong>api/pipelines/temporal_cnn_bridge.py</strong><p>Loads models once, preserves QA state per run, and maps model output into the inspection contract.</p></div>
          <div><span>OPERATOR APPLICATION</span><strong>FastAPI + SQLite/artifacts + React</strong><p>Resolves timestamped recipe context, stores source/render/mask evidence, and presents review—not PLC control.</p></div>
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
                {health?.pipeline_mode === "mock" && <button className="button quiet" onClick={() => void sendGeneratedFrame(demoFrame)} disabled={isProcessing} type="button">SEND DEMO FRAME</button>}
                <label className="sample-control"><span>SAMPLE SEC</span><input type="number" min="0.1" step="0.1" value={sampleSeconds} onChange={(event) => setSampleSeconds(Number(event.target.value))} /></label>
              </div>
            ) : (
              <div className={`drop-zone ${dragActive ? "active" : ""}`} onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragActive(false)} onDrop={onDrop} onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" hidden multiple={sourceMode === "images"} accept={sourceMode === "images" ? "image/jpeg,image/png,image/webp" : "video/mp4,video/webm,video/quicktime,.mov"} onClick={(event) => event.stopPropagation()} onChange={(event) => { const files = Array.from(event.target.files ?? []); event.target.value = ""; void handleFiles(files); }} />
                <strong>DROP {sourceMode === "images" ? "IMAGES" : "A RECORDING"} HERE</strong>
                <span>{sourceMode === "images" ? "Image order supplies 0 s, 1 s, 2 s… timestamps · originals and outputs are retained" : "Media seek time supplies each frame timestamp · original recording and outputs are retained"}</span>
                <button className="button quiet file-picker-button" type="button" onClick={(event) => { event.stopPropagation(); fileInputRef.current?.click(); }}>CHOOSE FILES</button>
                {sourceMode === "recording" && <label className="sample-control" onClick={(event) => event.stopPropagation()}><span>FRAME INTERVAL SEC</span><input type="number" min="0.1" step="0.1" value={sampleSeconds} onChange={(event) => setSampleSeconds(Number(event.target.value))} /></label>}
              </div>
            )}

            {ingestion && <div className="ingestion-progress"><div><strong>{ingestion.label}</strong><span>{ingestion.current} / {ingestion.total}</span></div><div className="progress-track"><span style={{ width: `${Math.round((ingestion.current / ingestion.total) * 100)}%` }} /></div>{isProcessing && <button type="button" onClick={() => { cancelIngestionRef.current = true; }}>STOP AFTER CURRENT FRAME</button>}</div>}

            <div className={`visual-note ${maskImage ? "ready" : "pending"}`}><strong>{maskImage ? "MASK ARTIFACT SAVED" : "MASK OUTPUT PENDING"}</strong><span>{maskImage ? "Source, rendered output, and mask are available from run history." : health?.pipeline_mode === "model" ? "The model bridge will attach rendered and mask artifacts after successful inference." : "The mock pipeline persists source frames without fabricating model visuals."}</span></div>

            <form className="context-form" onSubmit={onContextSubmit}>
              <div className="context-title"><div><h3>Planned vial order</h3><p>Define what should arrive, then activate each vial as production advances.</p></div><span>{session ? `RUN ${session.session_id.slice(0, 8)}` : "NOT STARTED"}</span></div>
              <label><span>RUN NAME</span><input value={config.run_name} disabled={Boolean(session)} onChange={(event) => setConfig({ ...config, run_name: event.target.value })} /></label>
              <label><span>CAMERA ID</span><input value={config.camera_id} disabled={Boolean(session)} onChange={(event) => setConfig({ ...config, camera_id: event.target.value })} /></label>
              <div className="recipe-plan-editor">
                <div className="recipe-plan-head"><span>ORDER</span><span>RECIPE</span><span>EXPECTED COLOR</span><span>ACTIONS</span></div>
                <div className="recipe-plan-list">
                  {recipePlan.map((recipe, index) => (
                    <div className={`recipe-plan-row ${index === activePlanIndex ? "active" : ""}`} key={`${recipe.vial_index}-${index}`}>
                      <div className="vial-order"><strong>VIAL {index + 1}</strong><small>{index === activePlanIndex ? "ACTIVE / NEXT" : `STEP ${index + 1}`}</small></div>
                      <div className="plan-recipe-fields">
                        <input aria-label={`Vial ${index + 1} recipe ID`} disabled={Boolean(session)} value={recipe.current_recipe_id} onChange={(event) => updatePlanRecipe(index, { current_recipe_id: event.target.value })} />
                        <input aria-label={`Vial ${index + 1} recipe name`} disabled={Boolean(session)} value={recipe.current_recipe_name} onChange={(event) => updatePlanRecipe(index, { current_recipe_name: event.target.value })} />
                      </div>
                      <div className="plan-color-fields">
                        <input aria-label={`Vial ${index + 1} color swatch`} className="color-picker" disabled={Boolean(session)} type="color" value={recipe.expected_color_hex ?? "#82918f"} onChange={(event) => updatePlanRecipe(index, { expected_color_hex: event.target.value })} />
                        <input aria-label={`Vial ${index + 1} expected color`} disabled={Boolean(session)} placeholder="e.g. Blue" value={recipe.expected_color_name ?? ""} onChange={(event) => updatePlanRecipe(index, { expected_color_name: event.target.value })} />
                      </div>
                      <div className="plan-actions">
                        {!session && <><button type="button" disabled={index === 0} onClick={() => movePlanRecipe(index, -1)} title="Move vial up">↑</button><button type="button" disabled={index === recipePlan.length - 1} onClick={() => movePlanRecipe(index, 1)} title="Move vial down">↓</button><button type="button" disabled={recipePlan.length === 1} onClick={() => { applyRecipePlan(recipePlan.filter((_, itemIndex) => itemIndex !== index)); setActivePlanIndex(0); }} title="Remove vial">×</button></>}
                        <button className="activate-plan" type="button" onClick={() => choosePlanRecipe(index)}>{session ? "ACTIVATE" : "START HERE"}</button>
                      </div>
                      <div className="plan-formula">
                        <span className="formula-heading">FILL STATION 3 VOLUME (mL)</span>
                        {tankAmountFields.map(({ field, label }) => <label className="tank-dose" key={field}><span>{label}</span><input aria-label={`Vial ${index + 1} ${label} milliliters`} disabled={Boolean(session)} min="0" step="0.1" type="number" value={recipe[field] ?? 0} onChange={(event) => updateTankAmount(index, field, event.target.value)} /></label>)}
                        <div className="formula-total"><span>TOTAL</span><strong>{formatMl(formulaTotal(recipe))} mL</strong></div>
                      </div>
                    </div>
                  ))}
                </div>
                {!session && <button className="button quiet add-vial-button" type="button" onClick={addPlanRecipe}>+ ADD VIAL</button>}
                <div className="recipe-plan-note"><strong>PLAN ≠ PREDICTION</strong><span>The expected color is operator intent. Activating a vial records when that plan step becomes active; vision independently reports what it observes.</span></div>
              </div>
              <div className="active-plan-summary"><span>SELECTED PLAN STEP</span><strong>Vial {config.recipe.vial_index} · {config.recipe.current_recipe_name}</strong><small><i style={{ background: config.recipe.expected_color_hex ?? "#82918f" }} />Expected {config.recipe.expected_color_name ?? "color not set"}</small><small>T100 {formatMl(config.recipe.t100_ml)} · T200 {formatMl(config.recipe.t200_ml)} · T300 {formatMl(config.recipe.t300_ml)} · T400 {formatMl(config.recipe.t400_ml)} · {formatMl(formulaTotal(config.recipe))} mL total</small></div>
              {!session ? <button className="button primary lock-button" type="submit">START RUN + SAVE ORDER</button> : <div className="activation-help"><strong>NEXT VIAL</strong><span>Choose ACTIVATE on the matching plan row. The source-time boundary is recorded automatically.</span></div>}
              <button className="button quiet" type="button" onClick={() => recipeInputRef.current?.click()}>IMPORT RECIPE CSV</button>
              <input ref={recipeInputRef} hidden type="file" accept=".csv,text/csv" onChange={(event) => void onRecipeImport(event.target.files?.[0])} />
              {session && <button className="button quiet" type="button" onClick={resetSession}>NEW RUN</button>}
              {recipeImport && <div className={`recipe-import-status ${recipeImport.state}`}><strong>{recipeImport.state === "staged" ? "CSV STAGED" : "CSV IMPORTED"}</strong><span>{recipeImport.fileName}{recipeImport.segmentCount === undefined ? " · will attach when the run starts" : ` · ${recipeImport.segmentCount} new timestamped segments · run ${session?.session_id.slice(0, 8) ?? "pending"}`}</span></div>}
              <div className="recipe-import-scope"><strong>ORDER + CONTEXT</strong><span>CSV rows populate the vial plan and timestamped replay context. Color labels describe operator intent; qualified references and the decision layer are still required for recipe pass/fail.</span></div>
            </form>
          </section>

          <aside className="side-column">
            <section className={`panel decision-panel decision-${status}`}><div className="panel-heading compact"><div><span className="section-number">02</span><div><h2>Evidence and QA</h2><p>Human-in-the-loop guidance</p></div></div><span className="mode-badge">{latestEvent?.pipeline_mode.toUpperCase() ?? health?.pipeline_mode.toUpperCase() ?? "NO RESULT"}</span></div><div className="decision-body"><div className="decision-icon">{status === "failure" ? "!" : status === "warning" ? "?" : "✓"}</div><span className="decision-label">EVIDENCE / QA STATUS</span><h2>{latestEvent?.decision.headline ?? "Awaiting inspection frame"}</h2><p>{latestEvent?.decision.detail ?? "Create a run and ingest a source to exercise the durable pipeline."}</p><div className="recommended-action"><span>OPERATOR ACTION</span><strong>{latestEvent?.decision.recommended_action ?? "Choose source and recipe context"}</strong></div>{latestEvent?.decision.status === "failure" && <button className="button acknowledge" type="button" disabled={Boolean(latestEvent.acknowledged_at)} onClick={() => void acknowledge()}>{latestEvent.acknowledged_at ? "ACKNOWLEDGED" : "ACKNOWLEDGE + REVIEW"}</button>}</div></section>

            <section className="panel progress-panel"><div className="panel-heading compact"><div><span className="section-number">03</span><div><h2>{modelEvidenceOnly ? "Recipe context" : "Recipe progress"}</h2><p>{latestEvent?.recipe.current_recipe_name ?? session?.recipe.current_recipe_name ?? "No active segment"} · expected {latestEvent?.recipe.expected_color_name ?? session?.recipe.expected_color_name ?? "color unset"}</p></div></div><strong className={`progress-value ${modelEvidenceOnly ? "not-scored" : ""}`}>{modelEvidenceOnly ? "NOT SCORED" : `${latestEvent?.decision.recipe_progress_percent ?? 0}%`}</strong></div><div className="progress-content"><div className="progress-track"><span style={{ width: modelEvidenceOnly ? "0%" : `${latestEvent?.decision.recipe_progress_percent ?? 0}%` }} /></div><div className="measurements">{(latestEvent?.decision.measurements ?? [{ label: "Vial", value: "Waiting", state: "neutral" as const }, { label: "Fill", value: "Waiting", state: "neutral" as const }, { label: "Color", value: "Waiting", state: "neutral" as const }]).map((measurement) => <div className={`measurement ${measurement.state}`} key={measurement.label}><span>{measurement.label.toUpperCase()}</span><strong>{measurement.value}</strong></div>)}</div>{modelEvidenceOnly && <div className="recipe-scope-note"><strong>PLAN ASSIGNED, NOT YET COMPARED</strong><span>The active vial supplies intended recipe and color. Acceptance still needs a qualified recipe reference and decision layer.</span></div>}<div className="model-note"><span>PIPELINE SOURCE</span><p>{latestEvent?.decision.model_state ?? "No frame processed"}</p></div></div></section>

            <section className="panel events-panel"><div className="panel-heading compact"><div><span className="section-number">04</span><div><h2>Recent frames</h2><p>Persisted newest first · plan context + QA state</p></div></div><span className="event-count">{events.length}</span></div><div className="event-list">{events.length === 0 ? <div className="empty-events">No persisted inspection frames yet.</div> : events.map((item) => <div className={`event-row ${item.decision.status}`} key={item.event_id}><time>{timestampLabel(item.source_timestamp_ms)}</time><span className="event-marker" /><div><strong>{item.pipeline_mode === "model" && item.decision.status === "warning" ? `QA review · ${item.decision.headline}` : item.decision.headline}</strong><p>Vial {item.recipe.vial_index} · planned {item.recipe.expected_color_name ?? item.recipe.current_recipe_id} · {item.original_filename ?? item.frame_id}</p></div></div>)}</div><div className="message-bar">{message}</div>{session && <Link className="review-link" to={`/runs/${session.session_id}`}>OPEN FULL FRAME REVIEW →</Link>}</section>

            {health?.pipeline_mode === "mock" && <section className="simulation-tools"><span>INTEGRATION TEST CONTROLS</span><div><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "warning")} disabled={isProcessing}>SIMULATE WARNING</button><button type="button" onClick={() => void sendGeneratedFrame(demoFrame, "failure")} disabled={isProcessing}>SIMULATE FAILURE</button></div><p>Clearly labeled test states; not model predictions.</p></section>}
          </aside>
        </div>
      </main>

      <footer><div><strong>LINEWATCH · OPERATOR-IN-THE-LOOP PROTOTYPE</strong><span>Rockwell Automation + NVIDIA Innovation Lab · Fall 2026</span><span>FastAPI / React / SQLite · No PLC control</span></div><div className="team-credit"><strong>© 2026 Project Team</strong><span>Austin Koske · Alayna Ostler · Cameron Delpage · Zoe Kirkman · Srijan Kumar · Josiah Mathews · Hayden Fons · Reagan Burkemper</span></div></footer>
    </div>
  );
}
