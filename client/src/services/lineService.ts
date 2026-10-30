import { api } from "./api";
import type { Lab, RecipeDefinition } from "./recipeCheck";

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
  tray_slots: { good: [number, number][]; bad: [number, number][] };
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
  vial_boxes: number;
  yolo_vial_boxes: number;
  vial_geometry_source: "yolo" | "station_profile_candidate";
  flags: string[];
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
  | { type: "tray_missed"; t: number; seq: number; decision: string | null }
  | { type: "schedule"; t: number; schedule: string[]; scheduled_vials: number }
  | { type: "reference"; t: number; recipe_id: string; seq: number; lab: Lab; from: string }
  | { type: "reference_failed"; t: number; recipe_id: string; seq: number; reason: string };

export type LiveMessage =
  | LineMeta
  | { type: "events"; events: LineEvent[]; catch_up?: boolean }
  | { type: "clock"; t: number; lag_s: number; speed: number; paused: boolean; cameras?: Record<string, number> }
  | { type: "preview"; key: string; cameras: string[]; run_id: string }
  | { type: "saved"; run_id: string }
  | { type: "end"; t: number }
  | { type: "error"; message: string };

export type LineSource = "recorded" | "video" | "camera";

export interface LineRun {
  id: string;
  title: string;
  sources: LineSource[];
  has_tray: boolean;
  /** The station's live cameras (run "live"); models: whether the server has the models loaded. */
  live?: boolean;
  models?: boolean;
  /** The station also offers mock cameras (recorded videos), started with simulate "mock". */
  mock?: boolean;
}

/** An inspection run whose recipe plan (the order copied from CSI's machine) the line can follow. */
export interface LineOrder {
  session_id: string;
  run_name: string;
  created_at: string;
  vials: number;
  recipes: string[];
}

export async function listLineOrders(): Promise<LineOrder[]> {
  return (await api.get<LineOrder[]>("/line/orders")).data;
}

export async function listLineRuns(): Promise<LineRun[]> {
  return (await api.get<LineRun[]>("/line/runs")).data;
}

export async function reviewEvents(runId: string, order?: string | null): Promise<LineEvent[]> {
  const params = order ? { order } : undefined;
  return (await api.get<{ events: LineEvent[] }>(`/line/runs/${encodeURIComponent(runId)}/events`, { params })).data.events;
}

export function videoUrl(runId: string, camera: "fill" | "tray"): string {
  return `${api.defaults.baseURL}/line/runs/${encodeURIComponent(runId)}/video/${camera}`;
}

export function previewUrl(key: string, camera: "fill" | "fill_mask" | "tray" | "tray_detection"): string {
  return `${api.defaults.baseURL}/line/preview/${encodeURIComponent(key)}/${camera}`;
}

export function liveSocket(runId: string, source: string, speed: number, start: number, order?: string | null,
  simulate?: string | null): WebSocket {
  const base = new URL(api.defaults.baseURL ?? "/api/v1", window.location.href);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/line/runs/${encodeURIComponent(runId)}/live`;
  base.search = new URLSearchParams({ source, speed: String(speed), start: String(start), ...(order ? { order } : {}),
    ...(simulate ? { simulate } : {}) }).toString();
  return new WebSocket(base);
}

/** The live station's recipe library with recipes created at the line, and the recipe armed in CREATE mode. */
export interface LineRecipes { recipes: RecipeDefinition[]; capture: string | null }

export async function listLineRecipes(): Promise<LineRecipes> {
  return (await api.get<LineRecipes>("/line/recipes")).data;
}

export async function createLineRecipe(name: string, ml: { t100_ml: number; t200_ml: number; t400_ml: number }) {
  return (await api.post<RecipeDefinition>("/line/recipes", { name, ...ml })).data;
}

/** CREATE mode: the next vial the live cameras decide sets this recipe's reference (null turns it off). */
export async function setLineCapture(recipeId: string | null): Promise<string | null> {
  return (await api.put<{ capture: string | null }>("/line/recipes/capture", { recipe_id: recipeId })).data.capture;
}
