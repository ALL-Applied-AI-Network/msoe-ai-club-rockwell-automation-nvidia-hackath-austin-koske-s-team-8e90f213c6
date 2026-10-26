import type { HealthResponse } from "../types";

const apiBase = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return apiBase + path;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail || "Request failed with status " + response.status;
  } catch {
    return "Request failed with status " + response.status;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(apiUrl("/health"));
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as HealthResponse;
}

export async function postCameraFrame(
  cameraId: string,
  form: FormData,
): Promise<void> {
  const response = await fetch(
    apiUrl("/v1/cameras/" + encodeURIComponent(cameraId) + "/frames"),
    { method: "POST", body: form },
  );
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
}

export function websocketUrl(cameraId: string): string {
  const configured = (import.meta.env.VITE_WS_BASE_URL || "").replace(/\/$/, "");
  if (configured) {
    return configured + "/ws/frontend/" + encodeURIComponent(cameraId);
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return (
    protocol +
    "//" +
    window.location.host +
    "/ws/frontend/" +
    encodeURIComponent(cameraId)
  );
}

