import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  getHealth,
  postCameraFrame,
  websocketUrl,
} from "../services/inspectionApi";
import type {
  CameraFrameEvent,
  HealthResponse,
  InspectionContext,
  OperatorSummary,
  StreamPhase,
} from "../types";

const FRAME_INTERVAL_MS = 1000;
const STALE_AFTER_MS = 5000;

const stoppedSummary: OperatorSummary = {
  state: "STOPPED",
  title: "Camera is stopped",
  detail: "Start the camera to begin inspection.",
  updatedAt: null,
};

function parseNumbers(value: string, count: number, label: string): number[] | null {
  if (!value.trim()) return null;
  const values = value.split(",").map((part) => Number(part.trim()));
  if (values.length !== count || values.some((item) => !Number.isFinite(item))) {
    throw new Error(label + " must contain " + count + " comma-separated numbers.");
  }
  return values;
}

function validateContext(context: InspectionContext): void {
  parseNumbers(context.expectedColorRgb, 3, "Clean reference RGB");
  parseNumbers(context.colorRoi, 4, "Color ROI");
  parseNumbers(context.liquidPromptBox, 4, "Liquid prompt box");
}

function deriveOperatorSummary(
  event: CameraFrameEvent,
  context: InspectionContext,
): OperatorSummary {
  const decision = event.inspection.decision;
  const color = decision.color_inspection;
  const now = Date.now();

  if (color.alert) {
    return {
      state: "FAILURE",
      title: "Color deviation detected",
      detail:
        "ΔE " +
        Number(color.delta_e_2000 || 0).toFixed(2) +
        " exceeds the configured threshold " +
        Number(color.threshold || 0).toFixed(2) +
        ".",
      updatedAt: now,
    };
  }

  if (decision.sample_usability.usable === false) {
    return {
      state: "FAILURE",
      title: "Frame failed quality checks",
      detail:
        decision.sample_usability.reasons.join(", ") ||
        "The model marked this sample as unusable.",
      updatedAt: now,
    };
  }

  if (event.inspection.status === "segmentation_unavailable") {
    return {
      state: "WARNING",
      title: "Segmentation is unavailable",
      detail:
        "The frame was accepted, but SAM 2 is not configured on the backend.",
      updatedAt: now,
    };
  }

  const recipe = context.currentRecipeId.trim();
  return {
    state: "RUNNING_OK",
    title: recipe ? "Recipe " + recipe + " is progressing normally" : "Inspection is healthy",
    detail:
      event.inspection.status === "ok"
        ? "The latest frame produced a liquid-mask proposal."
        : "The latest frame was processed; no liquid prompt was available.",
    updatedAt: now,
  };
}

type UseCameraInspectionResult = {
  videoRef: RefObject<HTMLVideoElement | null>;
  phase: StreamPhase;
  operator: OperatorSummary;
  health: HealthResponse | null;
  latestEvent: CameraFrameEvent | null;
  events: CameraFrameEvent[];
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
  setSelectedDeviceId: (value: string) => void;
  frameCount: number;
  requestMs: number | null;
  droppedFrames: number;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  refreshDevices: () => Promise<void>;
};

export function useCameraInspection(
  context: InspectionContext,
): UseCameraInspectionResult {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const loopRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const startedAtRef = useRef(0);
  const frameNumberRef = useRef(0);
  const contextRef = useRef(context);
  const intentionallyStoppedRef = useRef(true);
  const lastEventAtRef = useRef(0);

  const [phase, setPhase] = useState<StreamPhase>("idle");
  const [operator, setOperator] = useState<OperatorSummary>(stoppedSummary);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latestEvent, setLatestEvent] = useState<CameraFrameEvent | null>(null);
  const [events, setEvents] = useState<CameraFrameEvent[]>([]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [frameCount, setFrameCount] = useState(0);
  const [requestMs, setRequestMs] = useState<number | null>(null);
  const [droppedFrames, setDroppedFrames] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    contextRef.current = context;
  }, [context]);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const available = await navigator.mediaDevices.enumerateDevices();
    const cameras = available.filter((device) => device.kind === "videoinput");
    setDevices(cameras);
    setSelectedDeviceId((current) => current || cameras[0]?.deviceId || "");
  }, []);

  const handleFrameEvent = useCallback((event: CameraFrameEvent) => {
    lastEventAtRef.current = Date.now();
    setLatestEvent(event);
    setEvents((current) => [event, ...current].slice(0, 30));
    setOperator(deriveOperatorSummary(event, contextRef.current));
    setError(null);
  }, []);

  const connectSocket = useCallback(
    (cameraId: string): Promise<void> =>
      new Promise((resolve, reject) => {
        const socket = new WebSocket(websocketUrl(cameraId));
        let opened = false;
        const timeout = window.setTimeout(() => {
          socket.close();
          reject(new Error("WebSocket connection timed out."));
        }, 5000);

        socket.onopen = () => {
          opened = true;
          window.clearTimeout(timeout);
          socketRef.current = socket;
          resolve();
        };
        socket.onmessage = (message) => {
          try {
            const event = JSON.parse(String(message.data)) as CameraFrameEvent;
            if (event.type === "camera.frame") handleFrameEvent(event);
          } catch {
            setError("The backend sent an unreadable inspection event.");
          }
        };
        socket.onerror = () => {
          if (!opened) {
            window.clearTimeout(timeout);
            reject(new Error("WebSocket connection failed."));
          }
        };
        socket.onclose = () => {
          if (socketRef.current === socket) socketRef.current = null;
          if (!intentionallyStoppedRef.current) {
            setPhase("failed");
            setOperator({
              state: "FAILURE",
              title: "Live result channel disconnected",
              detail: "Stop and restart the camera to reconnect.",
              updatedAt: Date.now(),
            });
          }
        };
      }),
    [handleFrameEvent],
  );

  const sendFrame = useCallback(async () => {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!video || !streamRef.current || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    if (inFlightRef.current) {
      setDroppedFrames((current) => current + 1);
      return;
    }
    if (!video.videoWidth || !video.videoHeight) return;

    inFlightRef.current = true;
    const requestStarted = performance.now();
    try {
      const width = Math.min(video.videoWidth, 960);
      const height = Math.round((video.videoHeight * width) / video.videoWidth);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const drawing = canvas.getContext("2d");
      if (!drawing) throw new Error("The browser could not create a frame canvas.");
      drawing.translate(width, 0);
      drawing.scale(-1, 1);
      drawing.drawImage(video, 0, 0, width, height);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.78),
      );
      if (!blob) throw new Error("The browser could not encode the camera frame.");

      const activeContext = contextRef.current;
      const promptBox = parseNumbers(
        activeContext.liquidPromptBox,
        4,
        "Liquid prompt box",
      );
      frameNumberRef.current += 1;
      const frameId =
        activeContext.cameraId + "-" + String(frameNumberRef.current).padStart(6, "0");
      const form = new FormData();
      form.append("image", blob, frameId + ".jpg");
      form.append("source_id", frameId);
      form.append(
        "pts_seconds",
        ((performance.now() - startedAtRef.current) / 1000).toFixed(3),
      );
      form.append("width", String(width));
      form.append("height", String(height));
      form.append(
        "detections_json",
        JSON.stringify(
          promptBox
            ? [
                {
                  xyxy: promptBox,
                  class_id: 1,
                  label: "liquid",
                  confidence: 1,
                  source: "operator_prompt",
                },
              ]
            : [],
        ),
      );

      if (activeContext.currentRecipeId.trim()) {
        form.append("current_recipe_id", activeContext.currentRecipeId.trim());
      }
      if (activeContext.previousRecipeId.trim()) {
        form.append("previous_recipe_id", activeContext.previousRecipeId.trim());
      }
      if (activeContext.vialIndex.trim()) {
        form.append("vial_index", activeContext.vialIndex.trim());
      }
      form.append(
        "is_first_vial_after_change",
        String(activeContext.firstVialAfterChange),
      );
      const expected = parseNumbers(
        activeContext.expectedColorRgb,
        3,
        "Clean reference RGB",
      );
      const roi = parseNumbers(activeContext.colorRoi, 4, "Color ROI");
      if (expected) form.append("expected_color_rgb_json", JSON.stringify(expected));
      if (roi) form.append("color_roi_xyxy_json", JSON.stringify(roi));

      await postCameraFrame(activeContext.cameraId, form);
      setFrameCount(frameNumberRef.current);
      setRequestMs(Math.round(performance.now() - requestStarted));
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Frame upload failed.";
      setError(message);
      setOperator({
        state: "WARNING",
        title: "Frame processing is delayed",
        detail: message,
        updatedAt: Date.now(),
      });
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  const teardown = useCallback(() => {
    if (loopRef.current !== null) window.clearInterval(loopRef.current);
    loopRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    socketRef.current?.close();
    socketRef.current = null;
    inFlightRef.current = false;
  }, []);

  const stop = useCallback(() => {
    intentionallyStoppedRef.current = true;
    teardown();
    setPhase("stopped");
    setOperator(stoppedSummary);
  }, [teardown]);

  const start = useCallback(async () => {
    intentionallyStoppedRef.current = false;
    setError(null);
    setPhase("starting");
    setOperator({
      state: "STARTING",
      title: "Starting camera session",
      detail: "Checking the backend and requesting camera access.",
      updatedAt: Date.now(),
    });
    try {
      validateContext(contextRef.current);
      const backendHealth = await getHealth();
      setHealth(backendHealth);
      const cameraId = contextRef.current.cameraId.trim() || "fillcam-1";
      contextRef.current = { ...contextRef.current, cameraId };
      await connectSocket(cameraId);
      const media = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: selectedDeviceId
          ? {
              deviceId: { exact: selectedDeviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = media;
      if (!videoRef.current) throw new Error("Camera viewport is not mounted.");
      videoRef.current.srcObject = media;
      await videoRef.current.play();
      frameNumberRef.current = 0;
      startedAtRef.current = performance.now();
      lastEventAtRef.current = Date.now();
      setFrameCount(0);
      setDroppedFrames(0);
      setLatestEvent(null);
      setEvents([]);
      setPhase("streaming");
      setOperator({
        state: "STARTING",
        title: "Camera connected",
        detail: "Waiting for the first processed frame.",
        updatedAt: Date.now(),
      });
      await refreshDevices();
      void sendFrame();
      loopRef.current = window.setInterval(() => void sendFrame(), FRAME_INTERVAL_MS);
    } catch (caught) {
      intentionallyStoppedRef.current = true;
      teardown();
      const message =
        caught instanceof Error ? caught.message : "The camera session could not start.";
      setError(message);
      setPhase("failed");
      setOperator({
        state: "FAILURE",
        title: "Camera session failed to start",
        detail: message,
        updatedAt: Date.now(),
      });
    }
  }, [connectSocket, refreshDevices, selectedDeviceId, sendFrame, teardown]);

  useEffect(() => {
    void refreshDevices();
    const deviceListener = () => void refreshDevices();
    navigator.mediaDevices?.addEventListener("devicechange", deviceListener);

    const healthTimer = window.setInterval(() => {
      void getHealth().then(setHealth).catch(() => setHealth(null));
    }, 10000);
    void getHealth().then(setHealth).catch(() => setHealth(null));

    const staleTimer = window.setInterval(() => {
      if (
        streamRef.current &&
        lastEventAtRef.current &&
        Date.now() - lastEventAtRef.current > STALE_AFTER_MS
      ) {
        setOperator({
          state: "WARNING",
          title: "Inspection results are stale",
          detail: "The local camera is live, but no processed frame arrived recently.",
          updatedAt: Date.now(),
        });
      }
    }, 1000);

    return () => {
      navigator.mediaDevices?.removeEventListener("devicechange", deviceListener);
      window.clearInterval(healthTimer);
      window.clearInterval(staleTimer);
      intentionallyStoppedRef.current = true;
      teardown();
    };
  }, [refreshDevices, teardown]);

  return {
    videoRef,
    phase,
    operator,
    health,
    latestEvent,
    events,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    frameCount,
    requestMs,
    droppedFrames,
    error,
    start,
    stop,
    refreshDevices,
  };
}

