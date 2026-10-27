import type { Investigation } from "@/models/investigation";
import { api } from "@/services/api";

export async function listInvestigationExamples(signal?: AbortSignal): Promise<{ id: string; label: string }[]> {
  return (await api.get("/investigation/examples", { signal })).data;
}

export async function getInvestigation(runId: string, signal?: AbortSignal): Promise<Investigation> {
  return (await api.get(`/sessions/${runId}/investigation`, { signal })).data;
}

export async function getInvestigationExample(id: string, signal?: AbortSignal): Promise<Investigation> {
  return (await api.get(`/investigation/examples/${id}`, { signal })).data;
}

export async function importInvestigation(runId: string, file: File, signal?: AbortSignal): Promise<Investigation> {
  return (await api.post(`/sessions/${runId}/investigation`, file, {
    headers: { "Content-Type": "application/json" }, signal,
  })).data;
}
