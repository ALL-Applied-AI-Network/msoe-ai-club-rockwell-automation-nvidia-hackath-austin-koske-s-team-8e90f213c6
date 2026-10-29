import { api } from "@/services/api";
import type {
  CreateSessionRequest,
  ArtifactRecord,
  HealthStatus,
  InspectionEvent,
  InspectionSession,
  RecipeSegment,
  RecipeSegmentCreate,
  RecipeContext,
  RecipePlanItem,
  SimulationMode,
  OperatorResponse,
} from "@/models/inspection";

export async function getHealth(): Promise<HealthStatus> {
  const base = import.meta.env.VITE_API_BASE_URL;
  const response = await fetch(base && /^https?:\/\//.test(base) ? `${new URL(base).origin}/health` : "/health");
  if (!response.ok) throw new Error("Inspection API is unavailable");
  return response.json() as Promise<HealthStatus>;
}

export async function createInspectionSession(
  body: CreateSessionRequest,
): Promise<InspectionSession> {
  const response = await api.post<InspectionSession>("/sessions", body);
  return response.data;
}

export async function submitInspectionFrame(
  sessionId: string,
  frameId: string,
  frame: Blob,
  width: number,
  height: number,
  sourceTimestampMs?: number,
  originalFilename?: string,
  simulate?: SimulationMode,
  signal?: AbortSignal,
  sourceClock = "unknown",
): Promise<InspectionEvent> {
  const response = await api.post<InspectionEvent>(
    `/sessions/${sessionId}/frames`,
    frame,
    {
      headers: { "Content-Type": frame.type || "image/jpeg" },
      signal,
      params: {
        frame_id: frameId,
        width,
        height,
        source_timestamp_ms: sourceTimestampMs,
        original_filename: originalFilename,
        simulate,
        source_clock: sourceClock,
      },
    },
  );
  return response.data;
}

export async function listInspectionRuns(): Promise<InspectionSession[]> {
  const response = await api.get<InspectionSession[]>("/sessions", {
    params: { limit: 250 },
  });
  return response.data;
}

export async function getInspectionRun(runId: string): Promise<InspectionSession> {
  const response = await api.get<InspectionSession>(`/sessions/${runId}`);
  return response.data;
}

export async function listInspectionEvents(
  runId: string,
  limit = 250,
  beforeFrame?: number,
  reviewOnly = false,
): Promise<InspectionEvent[]> {
  const response = await api.get<InspectionEvent[]>(`/sessions/${runId}/events`, {
    params: { limit, before_frame: beforeFrame, review_only: reviewOnly },
  });
  return response.data;
}

export async function addRecipeSegment(
  runId: string,
  segment: RecipeSegmentCreate,
): Promise<RecipeSegment> {
  const response = await api.post<RecipeSegment>(
    `/sessions/${runId}/recipe-segments`,
    segment,
  );
  return response.data;
}

export async function listRecipeSegments(runId: string): Promise<RecipeSegment[]> {
  const response = await api.get<RecipeSegment[]>(
    `/sessions/${runId}/recipe-segments`,
  );
  return response.data;
}

export async function listRecipePlan(runId: string): Promise<RecipePlanItem[]> {
  const response = await api.get<RecipePlanItem[]>(
    `/sessions/${runId}/recipe-plan`,
  );
  return response.data;
}

export async function replaceRecipePlan(
  runId: string,
  recipes: RecipeContext[],
): Promise<RecipePlanItem[]> {
  const response = await api.put<RecipePlanItem[]>(
    `/sessions/${runId}/recipe-plan`,
    { recipes },
  );
  return response.data;
}

export async function importRecipeSegments(
  runId: string,
  file: File,
  signal?: AbortSignal,
): Promise<RecipeSegment[]> {
  const response = await api.post<RecipeSegment[]>(
    `/sessions/${runId}/recipe-segments/import`,
    file,
    { headers: { "Content-Type": "text/csv" }, signal },
  );
  return response.data;
}

export async function uploadRecording(
  runId: string,
  file: File,
  signal?: AbortSignal,
): Promise<ArtifactRecord> {
  const response = await api.post<ArtifactRecord>(
    `/sessions/${runId}/recordings`,
    file,
    {
      headers: { "Content-Type": file.type || "application/octet-stream" },
      signal,
      params: { filename: file.name },
    },
  );
  return response.data;
}

export function artifactUrl(path?: string | null): string | undefined {
  if (!path) return undefined;
  if (/^https?:\/\//.test(path)) return path;
  const base = import.meta.env.VITE_API_BASE_URL || "/api/v1";
  if (/^https?:\/\//.test(base)) {
    const origin = new URL(base).origin;
    return `${origin}${path}`;
  }
  return path;
}

export async function acknowledgeInspectionEvent(
  sessionId: string,
  eventId: string,
  body?: OperatorResponse,
): Promise<void> {
  await api.put(
    `/sessions/${sessionId}/events/${eventId}/acknowledgement`,
    body,
  );
}

export async function uploadReference(runId: string, file: File, signal?: AbortSignal): Promise<{ status: string }> {
  const response = await api.post(`/sessions/${runId}/reference`, file, {
    headers: { "Content-Type": file.type }, signal,
  });
  return response.data;
}

export function inspectionEventUrl(sessionId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || "/api/v1";
  if (/^https?:\/\//.test(base)) {
    return `${base.replace(/^http/, "ws")}/sessions/${sessionId}/events`;
  }
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  const path = `${base}/sessions/${sessionId}/events`.replace(/\/+/g, "/");
  return `${scheme}//${window.location.host}${path}`;
}
