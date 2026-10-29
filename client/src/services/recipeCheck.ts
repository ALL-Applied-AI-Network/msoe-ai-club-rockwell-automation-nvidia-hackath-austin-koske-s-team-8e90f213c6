import type { InspectionEvent } from "@/models/inspection";

export type Lab = [number, number, number];

export interface RecipeDefinition {
  id: string;
  name: string;
  t100_ml: number | null;
  t200_ml: number | null;
  t400_ml: number | null;
  order: string;
  note?: string | null;
  reference: { lab: Lab; from: string } | null;
}

export interface RecipeLibrary {
  source: string;
  band_de2000: number;
  direction_cos: number;
  endpoint_rule: string;
  recipes: RecipeDefinition[];
}

export type VialVerdict =
  | { kind: "unmeasured"; samples: number }
  | { kind: "no_reference"; lab: Lab; samples: number }
  | { kind: "match" | "differs"; lab: Lab; samples: number; deltaE: number; towardPrevious: boolean; deltaEPrevious: number | null };

const ENDPOINT_WINDOW_MS = 6000;

export async function loadRecipeLibrary(): Promise<RecipeLibrary> {
  const response = await fetch("/data/run5/recipe-library.json");
  if (!response.ok) throw new Error("Recipe library unavailable");
  return response.json();
}

export function recipeSummary(recipe: RecipeDefinition): string {
  const parts = [["clear", recipe.t100_ml], ["orange", recipe.t200_ml], ["blue", recipe.t400_ml]]
    .filter(([, ml]) => ml).map(([color, ml]) => `${ml} mL ${color}`);
  return parts.join(" · ");
}

/** CIEDE2000 colour difference (Sharma et al. 2005). */
export function deltaE2000(lab1: Lab, lab2: Lab): number {
  const [L1, a1, b1] = lab1, [L2, a2, b2] = lab2;
  const rad = Math.PI / 180;
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2);
  const Cbar = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hue = (b: number, a: number) => (a === 0 && b === 0 ? 0 : (Math.atan2(b, a) / rad + 360) % 360);
  const h1p = hue(b1, a1p), h2p = hue(b2, a2p);
  const dLp = L2 - L1, dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
  const Lbp = (L1 + L2) / 2, Cbp = (C1p + C2p) / 2;
  let hbp = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) > 180) hbp += h1p + h2p < 360 ? 360 : -360;
    hbp /= 2;
  }
  const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad)
    + 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.2 * Math.cos((4 * hbp - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hbp - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
  const Sc = 1 + 0.045 * Cbp, Sh = 1 + 0.015 * Cbp * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

function median(values: number[]): number {
  const sorted = [...values].sort((x, y) => x - y);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Vial endpoint: median Lab of QA-valid samples in the last 6 s of the vial's frames. */
export function vialEndpoint(events: InspectionEvent[]): { lab: Lab | null; samples: number } {
  const timed = events.filter((event) => event.source_timestamp_ms != null);
  if (!timed.length) return { lab: null, samples: 0 };
  const last = Math.max(...timed.map((event) => event.source_timestamp_ms as number));
  const labs = timed
    .filter((event) => (event.source_timestamp_ms as number) >= last - ENDPOINT_WINDOW_MS)
    .filter((event) => event.decision.evidence?.measurement_valid && event.decision.evidence.lab)
    .map((event) => event.decision.evidence!.lab as number[]);
  if (!labs.length) return { lab: null, samples: 0 };
  return { lab: [0, 1, 2].map((i) => median(labs.map((lab) => lab[i]))) as Lab, samples: labs.length };
}

export function judgeVial(events: InspectionEvent[], library: RecipeLibrary, recipeId: string, previousId: string | null): VialVerdict {
  const { lab, samples } = vialEndpoint(events);
  if (!lab) return { kind: "unmeasured", samples };
  const reference = library.recipes.find((recipe) => recipe.id === recipeId)?.reference?.lab;
  if (!reference) return { kind: "no_reference", lab, samples };
  const deltaE = deltaE2000(lab, reference);
  const previous = previousId && previousId !== recipeId
    ? library.recipes.find((recipe) => recipe.id === previousId)?.reference?.lab : undefined;
  let towardPrevious = false;
  let deltaEPrevious: number | null = null;
  if (previous) {
    deltaEPrevious = deltaE2000(lab, previous);
    const u = lab.map((v, i) => v - reference[i]), w = previous.map((v, i) => v - reference[i]);
    const dot = u.reduce((sum, v, i) => sum + v * w[i], 0);
    const cos = dot / (Math.hypot(...u) * Math.hypot(...w) + 1e-9);
    towardPrevious = cos >= library.direction_cos;
  }
  const differs = deltaE > library.band_de2000;
  return { kind: differs ? "differs" : "match", lab, samples, deltaE, towardPrevious: differs && towardPrevious, deltaEPrevious };
}
