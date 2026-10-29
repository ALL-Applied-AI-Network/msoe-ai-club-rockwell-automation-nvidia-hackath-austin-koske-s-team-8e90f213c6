import type { LineAction, LineEvent, LineMeta, LineSample, LineVial, TrayArrival } from "./lineService";
import type { Lab } from "./recipeCheck";

/** Everything the line view shows at video time t, from the events known by then. */
export interface LineView {
  t: number;
  meta: LineMeta | null;
  dwell: { start_t: number; samples: LineSample[] } | null;
  vials: LineVial[];
  actions: LineAction[];
  arrivals: TrayArrival[];
  missed: number[];
}

export function viewAt(events: LineEvent[], t: number): LineView {
  const view: LineView = { t, meta: null, dwell: null, vials: [], actions: [], arrivals: [], missed: [] };
  for (const event of events) {
    if (event.t > t) break;
    switch (event.type) {
      case "meta": view.meta = event; break;
      case "dwell_start": view.dwell = { start_t: event.t, samples: [] }; break;
      case "sample": view.dwell?.samples.push(event); break;
      case "dwell_end": view.dwell = null; break;
      case "vial": view.vials.push(event.vial); break;
      case "action": view.actions.push(event); break;
      case "tray_arrival": view.arrivals.push(event); break;
      case "tray_missed": view.missed.push(event.seq); break;
      case "schedule":
        if (view.meta) view.meta = { ...view.meta, schedule: event.schedule, scheduled_vials: event.scheduled_vials };
        break;
    }
  }
  return view;
}

export const MIN_FILL_PX = 300;

/** Phase of the vial under the nozzle, from the samples so far (no knowledge of the future). */
export function phase(view: LineView): string {
  const samples = view.dwell?.samples ?? [];
  if (!view.dwell) return "Carrier moving";
  if (!samples.length) return "Carrier stopped · taking empty-vial reference";
  const last = samples[samples.length - 1].liquid_px;
  const before = samples.slice(-5, -1).map((s) => s.liquid_px);
  if (last < MIN_FILL_PX) return "Waiting for fill";
  if (before.length && last > Math.max(...before) * 1.02) return "Filling";
  return "Fill complete · measuring until the carrier moves";
}

/** CIE Lab (D65) to an sRGB CSS colour. */
export function labToCss(lab: Lab | null | undefined): string {
  if (!lab) return "transparent";
  const [L, a, b] = lab;
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const inv = (f: number) => (f ** 3 > 0.008856 ? f ** 3 : (f - 16 / 116) / 7.787);
  const [x, y, z] = [0.95047 * inv(fx), 1.0 * inv(fy), 1.08883 * inv(fz)];
  const lin = [
    3.2406 * x - 1.5372 * y - 0.4986 * z,
    -0.9689 * x + 1.8758 * y + 0.0415 * z,
    0.0557 * x - 0.204 * y + 1.057 * z,
  ];
  const gamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  const [r, g, bl] = lin.map((c) => Math.round(Math.min(1, Math.max(0, gamma(c))) * 255));
  return `rgb(${r}, ${g}, ${bl})`;
}

export const VERDICT_TONE: Record<string, string> = { IN_SPEC: "signal", OUT_OF_SPEC: "failure", CANT_VERIFY: "warning" };
export const ACTION_TONE: Record<string, string> = {
  PASS: "signal", SCRAP: "failure", FLUSH: "cyan", FLUSH_RECOMMENDED: "cyan", STOP_CHECK_DOSING: "failure", HOLD: "warning",
};

export function slotName(index: number): string {
  return `R${Math.floor(index / 6) + 1}C${(index % 6) + 1}`;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
