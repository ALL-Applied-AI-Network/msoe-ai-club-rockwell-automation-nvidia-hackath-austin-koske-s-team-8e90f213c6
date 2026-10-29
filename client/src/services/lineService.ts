import { api } from "./api";
import type { Lab } from "./recipeCheck";

// Events from the live line engine (replay/live.py). "t" is the video time at which each is known.
export interface LineMeta {
  type: "meta";
  t: number;
  schedule: string[];
  scheduled_vials: number;
  references: Record<string, Lab | null>;
  line: { band_de2000: number; direction_cos: number; scrap_deadline_s: number };
  roi: [number, number, number, number];
  good_slots: [number, number][];
  trays: Record<string, [number, number, number, number]>;
  transit_window_s: [number, number];
}

export interface LineSample {
  type: "sample";
  t: number;
  frame: number;
  liquid_px: number;
  qa_valid: boolean;
  lab: Lab | null;
  presence: number;
}

export interface LineVial {
  seq: number;
  recipe_id: string;
  previous_recipe_id: string | null;
  arrival_s: number;
  fill_start_s: number;
  fill_end_s: number;
  decision_s: number;
  departure_s: number;
  endpoint_lab: Lab | null;
  de_intended: number | null;
  de_previous: number | null;
  direction_cos: number | null;
  verdict: "IN_SPEC" | "OUT_OF_SPEC" | "CANT_VERIFY";
  pattern: string;
  actions: string;
  rule_cause: string;
  final_liquid_px: number;
}

export interface LineAction {
  type: "action";
  t: number;
  kind: string;
  seq: number | null;
  issued_s: number;
  deadline_s: number | null;
  reason: string;
  plc_command: string;
}

export interface TrayArrival {
  type: "tray_arrival";
  t: number;
  arrival_s: number;
  tray: "good" | "bad";
  slot: string | null;
  capped: boolean | null;
  position: [number, number] | null;
  seq: number | null;
  decision?: string | null;
  verdict?: string;
  transit_s?: number;
  status: "routed_as_decided" | "escaped" | "good_vial_in_bad_tray" | "unmatched_arrival";
}

export type LineEvent =
  | LineMeta
  | LineSample
  | LineAction
  | TrayArrival
  | { type: "dwell_start" | "dwell_end"; t: number; frame: number }
  | { type: "vial"; t: number; vial: LineVial }
  | { type: "tray_missed"; t: number; seq: number; decision: string | null };

export type LiveMessage =
  | LineMeta
  | { type: "events"; events: LineEvent[]; catch_up?: boolean }
  | { type: "clock"; t: number; lag_s: number; speed: number; paused: boolean }
  | { type: "end"; t: number }
  | { type: "error"; message: string };

export interface LineRun {
  id: string;
  title: string;
  sources: ("recorded" | "video")[];
  has_tray: boolean;
}

export async function listLineRuns(): Promise<LineRun[]> {
  return (await api.get<LineRun[]>("/line/runs")).data;
}

export async function reviewEvents(runId: string): Promise<LineEvent[]> {
  return (await api.get<{ events: LineEvent[] }>(`/line/runs/${encodeURIComponent(runId)}/events`)).data.events;
}

export function videoUrl(runId: string, camera: "fill" | "tray"): string {
  return `${api.defaults.baseURL}/line/runs/${encodeURIComponent(runId)}/video/${camera}`;
}

export function liveSocket(runId: string, source: string, speed: number, start: number): WebSocket {
  const base = new URL(api.defaults.baseURL ?? "/api/v1", window.location.href);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/line/runs/${encodeURIComponent(runId)}/live`;
  base.search = new URLSearchParams({ source, speed: String(speed), start: String(start) }).toString();
  return new WebSocket(base);
}
