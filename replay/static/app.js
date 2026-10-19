"use strict";
const el = (id) => document.getElementById(id);
const video = el("primary");
const peers = ["tank_level", "cap_seal"].map(el);
const phases = {
  visible_stream: "Visible stream · image detection",
  ingredient_pause: "Stream paused · same fill remains open",
  carrier_moving: "Carrier departing / moving",
  carrier_settling: "Carrier settling",
  waiting_for_fill: "Waiting for visible fill evidence",
  detection_unavailable: "Fill evidence unavailable",
  annotated_fill: "Filling · supplied annotation",
  collecting_completed_vial_window: "Collecting post-fill measurement",
  completed_vial_available: "Completed measurement available",
  unlabeled: "Unlabeled recording · telemetry only",
  before_first_annotation: "Before the first supplied annotation",
};
const statuses = {
  unavailable: "Unable to assess",
  review: "Review requested",
  within_reference_band: "Within experimental band",
};
let catalog,
  current,
  snapshot,
  busy = false,
  refreshPending = false,
  generation = 0,
  prepared = false,
  fillIndex = [],
  navigationReady = false,
  pendingSeekFrame = null,
  lastFrame = -1;
const time = (seconds) => {
  const s = Math.max(0, seconds || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(2).padStart(5, "0")}`;
};
const lab = (value) =>
  value ? value.map((v) => v.toFixed(1)).join(" / ") : "—";
const rgb = (value) =>
  value
    ? `rgb(${value.map((v) => Math.max(0, Math.min(255, Math.round(v)))).join(",")})`
    : "";
function message(text, error = false) {
  el("message").textContent = text;
  el("message").classList.toggle("error", error);
}
async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
function swatch(id, value) {
  el(id).style.background = rgb(value);
}
function labDisplay(l) {
  if (!l) return null;
  const fy = (l[0] + 16) / 116,
    fx = fy + l[1] / 500,
    fz = fy - l[2] / 200;
  const inv = (t) =>
    t > 6 / 29 ? t * t * t : 3 * (6 / 29) ** 2 * (t - 4 / 29);
  const [x, y, z] = [inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883];
  const linear = [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
  return linear.map(
    (t) =>
      255 *
      Math.max(
        0,
        Math.min(
          1,
          t <= 0.0031308
            ? 12.92 * t
            : 1.055 * Math.max(0, t) ** (1 / 2.4) - 0.055,
        ),
      ),
  );
}
function overlays() {
  const canvas = el("overlay"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!catalog || !el("overlays").checked) return;
  for (const [name, color, label] of [
    ["color_roi", "#a6e3c3", "COLOR"],
    ["holder_roi", "#f5c27b", "QA"],
    ["nozzle_roi", "#9bc7e5", "NOZZLE"],
  ]) {
    const [x0, y0, x1, y1] = catalog.config[name];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = color;
    ctx.font = "9px system-ui";
    ctx.fillText(label, x0, y0 - 5);
  }
}
function sync(force = false) {
  for (const peer of peers) {
    if (!peer.src || peer.readyState < 1) continue;
    peer.playbackRate = video.playbackRate;
    if (force || Math.abs(peer.currentTime - video.currentTime) > 0.12)
      peer.currentTime = Math.min(
        video.currentTime,
        Math.max(0, peer.duration - 0.02),
      );
    if (video.paused) peer.pause();
    else peer.play().catch(() => {});
  }
}
async function selectRun() {
  generation++;
  pendingSeekFrame = null;
  video.pause();
  snapshot = null;
  lastFrame = -1;
  prepared = false;
  fillIndex = [];
  navigationReady = false;
  current = catalog.runs.find(
    (item) => item.run_id === Number(el("run").value),
  );
  video.src = `/media/${current.run_id}/vial_fill.mp4`;
  for (const [i, name] of ["tank_level", "cap_seal"].entries()) {
    peers[i].pause();
    if (current.cameras[name])
      peers[i].src = `/media/${current.run_id}/${name}.mp4`;
    else peers[i].removeAttribute("src");
  }
  el("vial").replaceChildren(
    new Option(
      catalog.fill_source === "image" ? "Preparing fill index…" : current.annotations.length ? "Choose a vial" : "No annotations",
      "",
    ),
  );
  for (const annotation of catalog.fill_source === "annotations" ? current.annotations : []) {
    const option = new Option(
      `Vial ${annotation.vial_id} · Recipe ${annotation.recipe_id}`,
      String(annotation.start_frame / current.cameras.vial_fill.fps),
    );
    el("vial").add(option);
  }
  el("reference").disabled = true;
  el("completed-jump").disabled = true;
  el("mode-label").textContent = catalog.fill_source === "image" ? "● IMAGE FILL DETECTION" : current.annotations.length
    ? "● ANNOTATED REPLAY"
    : "● UNLABELED REPLAY";
  el("ledger").replaceChildren();
  render({
    frame_index: 0,
    video_s: 0,
    phase: catalog.fill_source === "image" ? "waiting_for_fill" : current.annotations.length ? "before_first_annotation" : "unlabeled",
    feature: null,
    context: null,
    current_completed: null,
    completed: [],
    coverage: { available: 0, valid: 0, assessed: 0 },
  });
  for (const id of ["events", "observations"])
    el(id).setAttribute("aria-disabled", "true");
  el("provenance").textContent =
    `Profile ${catalog.config_id} · ${current.sync_status} · ${current.clock_status} · Outputs: ${catalog.output_directory}`;
  message("Preparing recording…");
  await tick(true);
}
function seekToFrame(frame) {
  const camera = current.cameras.vial_fill;
  const target = Math.max(0, Math.min(camera.frames - 1, Math.round(frame)));
  if (video.readyState === 0) {
    pendingSeekFrame = target;
    message("Loading recording metadata before seeking…");
    return;
  }
  pendingSeekFrame = null;
  // Stay inside the requested frame despite browser microsecond rounding.
  video.currentTime = target === 0 ? 0 : (target + 0.25) / camera.fps;
  sync(true);
  tick(true);
}
function render(state) {
  snapshot = state;
  const feature = state.feature,
    context = state.context,
    completed = state.current_completed;
  el("frame").textContent = `FRAME ${state.frame_index.toLocaleString()}`;
  el("clock").textContent = `MP4 ${time(state.video_s)}`;
  el("wall").textContent =
    feature?.estimated_wall_elapsed_s != null
      ? `Est. wall elapsed ${time(feature.estimated_wall_elapsed_s)} · unreviewed`
      : "Wall time unavailable";
  const hasRecipe = context?.recipe_id != null;
  el("recipe").textContent = context
    ? hasRecipe ? `Recipe ${context.recipe_id}${context.vial_id == null ? " selected" : " / detected fill " + context.vial_id}` : `Detected fill ${context.vial_id ?? "—"} · recipe unavailable`
    : "Recipe context unavailable";
  el("phase").textContent = phases[state.phase] || state.phase;
  el("order").textContent = hasRecipe
    ? `Order ${context.dispense_order}${context.position_in_batch == null ? "" : " · position " + context.position_in_batch + " in annotated batch"}`
    : "No recipe inferred from color";
  el("volumes").replaceChildren();
  if (context)
    for (const [tank, volume] of Object.entries(context.volumes_ml)) {
      const box = document.createElement("div"),
        name = document.createElement("small"),
        amount = document.createElement("strong");
      name.textContent = tank;
      amount.textContent = `${volume} mL`;
      box.append(name, amount);
      el("volumes").append(box);
    }
  el("hold").textContent = hasRecipe && context.position_in_batch != null
    ? `Baseline only: first-vial hold ${context.position_in_batch === 1 ? "YES" : "NO"} · first-two hold ${context.position_in_batch <= 2 ? "YES" : "NO"}`
    : "Baseline hold policies need an explicit batch-start input";
  const source = context?.recipe_context;
  el("recipe-source").textContent = source
    ? `${source.provenance.kind} · ${source.recorded_by} · active from frame ${source.active_from_frame} · ${source.provenance.description}`
    : "No explicit recipe selection for this fill";
  const assessment = completed?.assessment;
  el("decision-card").className =
    `panel decision ${assessment?.status || "unavailable"}`;
  el("decision").textContent = assessment
    ? statuses[assessment.status]
    : state.phase === "unlabeled"
      ? "Awaiting reviewed annotations"
      : "Awaiting completed vial";
  el("explanation").textContent =
    assessment?.explanation ||
    (state.phase === "unlabeled"
      ? "Video and image telemetry are available. Import reviewed recipe/fill annotations to enable completed-vial checks."
      : "Current partial-fill color is not compared with a finished-recipe reference.");
  swatch("observed-swatch", completed?.rgb);
  swatch("expected-swatch", labDisplay(assessment?.expected_lab));
  el("observed-lab").textContent = lab(completed?.lab);
  el("expected-lab").textContent = lab(assessment?.expected_lab);
  el("delta").textContent =
    assessment?.delta_e2000 != null ? assessment.delta_e2000.toFixed(2) : "—";
  el("threshold").textContent =
    `Threshold ${catalog.config.review_threshold_de2000.toFixed(2)} · unvalidated`;
  el("qa").textContent = completed
    ? `${completed.measurement_valid ? "QA accepted" : "QA unavailable"} · ${completed.accepted_sample_frames.length} accepted samples${completed.quality_flags.length ? " · " + completed.quality_flags.join(", ") : ""}`
    : "Post-fill sampling window has not completed";
  el("reference").disabled = !completed?.measurement_valid || completed?.recipe_id == null;
  el("reference").disabled ||= !!completed?.context_quality_flags?.length;
  el("reference-identity").textContent = assessment?.reference_id
    ? `${assessment.reference_id} · established at frame ${assessment.reference_available_frame} · ${assessment.reference_source_description}`
    : "No eligible reference · a development reference does not certify clean product";
  el("completed-context").textContent = completed
    ? `Completed detected fill ${completed.vial_id} · intended recipe ${completed.recipe_id ?? "unknown"} · published at MP4 ${time(completed.available_video_s)}`
    : "A completed decision will identify its own fill and recipe here";
  el("live-lab").textContent = lab(feature?.lab);
  el("live-quality").textContent = feature
    ? feature.measurement_valid
      ? "ROI QA accepted"
      : feature.quality_flags.join(", ")
    : "No current sample";
  el("motion").textContent =
    `Carrier motion ${feature?.motion != null ? feature.motion.toFixed(2) : "—"}`;
  el("nozzle").textContent =
    `Stream contrast ${feature?.stream_contrast != null ? feature.stream_contrast.toFixed(2) : "—"} · episode ${feature?.fill_detection?.episode || 0}`;
  el("latency").textContent =
    `Feature processing ${feature ? feature.processing_ms.toFixed(1) + " ms" : "—"} · excludes replay transport`;
  el("coverage").textContent =
    `${state.coverage.valid}/${state.coverage.available} observations pass QA · ${state.coverage.assessed} assessed`;
  el("ledger").replaceChildren();
  for (const row of [...state.completed].reverse()) {
    const tr = document.createElement("tr");
    function cell(text) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
      return td;
    }
    cell(`${row.vial_id} / ${row.recipe_id == null ? "recipe unknown" : "R" + row.recipe_id}`);
    cell(time(row.available_video_s));
    const qa = cell(row.measurement_valid ? "Accepted" : "Unavailable");
    qa.title = row.quality_flags.join(", ");
    const color = cell("");
    const patch = document.createElement("span");
    patch.className = "mini-swatch";
    patch.style.background = rgb(row.rgb);
    color.append(patch);
    cell(
      row.assessment.delta_e2000 != null
        ? row.assessment.delta_e2000.toFixed(2)
        : "—",
    );
    const status = cell(""),
      badge = document.createElement("span");
    badge.className = `badge ${row.assessment.status}`;
    badge.textContent = statuses[row.assessment.status];
    status.append(badge);
    const ref = cell(row.assessment.reference_id || "—");
    ref.className = "mono";
    el("ledger").append(tr);
  }
  if (!state.completed.length) {
    const tr = document.createElement("tr"),
      td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "Play or seek past a completed-vial sampling window.";
    tr.append(td);
    el("ledger").append(tr);
  }
  el("events").href =
    `/api/events.jsonl?run=${current.run_id}&frame=${state.frame_index}`;
  el("observations").href =
    `/api/observations.csv?run=${current.run_id}&frame=${state.frame_index}`;
}
function sourceFrame() {
  if (!current) return 0;
  const fps = current.cameras.vial_fill.fps;
  return Math.max(
    0,
    Math.min(
      current.cameras.vial_fill.frames - 1,
      Math.floor(video.currentTime * fps),
    ),
  );
}
async function tick(force = false) {
  if (!current) return;
  if (busy) { if (force) refreshPending = true; return; }
  const fps = current.cameras.vial_fill.fps;
  const frame = sourceFrame();
  if (!force && prepared && frame === lastFrame) return;
  busy = true;
  const token = generation;
  try {
    const data = await request(
      `/api/state?run=${current.run_id}&frame=${frame}`,
    );
    if (token !== generation) return;
    if (data.status === "preparing") {
      message(
        `Preparing ROI telemetry… ${Math.floor(data.progress * 100)}%. Video playback is available.`,
      );
      return;
    }
    if (data.status === "error") throw new Error(data.error);
    prepared = true;
    if (!navigationReady) {
      const navigation = await request(`/api/fills?run=${current.run_id}`);
      if (token !== generation) return;
      fillIndex = navigation.fills;
      navigationReady = true;
      el("vial").replaceChildren(new Option(fillIndex.length ? "Choose a fill" : "No completed fills detected", ""));
      for (const fill of fillIndex) {
        el("vial").add(new Option(`Fill ${fill.vial_id}${fill.recipe_id == null ? " · image detected" : " · Recipe " + fill.recipe_id}`, String(fill.start_frame / fps)));
      }
    }
    for (const id of ["events", "observations"])
      el(id).setAttribute("aria-disabled", "false");
    lastFrame = frame;
    render(data);
    message(
      catalog.fill_source === "image"
        ? "Image-based fills · explicit intended-recipe input · provisional comparisons request operator review and never authorize release."
        : "Annotation-guided baseline. Reference comparisons require a matching reference established before the evaluated fill.",
    );
  } catch (error) {
    message(error.message, true);
  } finally {
    busy = false;
    if (refreshPending) { refreshPending = false; queueMicrotask(() => tick(true)); }
  }
}
el("play").onclick = () => {
  if (video.paused) video.play().catch((error) => message(error.message, true));
  else video.pause();
};
el("restart").onclick = () => {
  video.pause();
  seekToFrame(0);
};
el("speed").onchange = () => {
  video.playbackRate = Number(el("speed").value);
  sync(true);
};
el("vial").onchange = () => {
  el("completed-jump").disabled = el("vial").value === "";
  if (el("vial").value !== "") {
    seekToFrame(Number(el("vial").value) * current.cameras.vial_fill.fps);
  }
};
el("completed-jump").onclick = () => {
  const annotation = fillIndex.find(
    (item) =>
      Math.abs(
        item.start_frame / current.cameras.vial_fill.fps -
          Number(el("vial").value),
      ) < 0.001,
  );
  if (annotation) {
    video.pause();
    seekToFrame(
      annotation.available_frame,
    );
  }
};
el("run").onchange = () =>
  selectRun().catch((error) => message(error.message, true));
el("overlays").onchange = overlays;
async function recordRecipe(unknown = false) {
  if (!snapshot || !current || !prepared) return;
  video.pause();
  try {
    const body = {run_id: current.run_id, frame_index: sourceFrame(),
      recipe_id: unknown ? null : el("recipe-id").value,
      volumes_ml: unknown ? {} : {T100:Number(el("tank100").value),T200:Number(el("tank200").value),T400:Number(el("tank400").value)},
      dispense_order: unknown ? null : el("recipe-order").value,
      recorded_by: el("operator-name").value, description: el("recipe-note").value};
    await request("/api/recipe-plan", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    await tick(true);
    message("Intended recipe selection recorded with provenance. Context is evaluated at each fill's start.");
  } catch (error) { message(error.message, true); }
}
el("recipe-form").onsubmit = event => { event.preventDefault(); recordRecipe(); };
el("recipe-clear").onclick = () => recordRecipe(true);
video.addEventListener("play", () => {
  el("play").textContent = "Pause";
  sync(true);
});
video.addEventListener("loadedmetadata", () => {
  video.playbackRate = Number(el("speed").value);
  if (pendingSeekFrame !== null) seekToFrame(pendingSeekFrame);
  sync(true);
});
video.addEventListener("pause", () => {
  el("play").textContent = "Play";
  sync(true);
});
video.addEventListener("seeked", () => {
  sync(true);
  tick(true);
});
video.addEventListener("timeupdate", () => {
  sync();
  tick();
});
video.addEventListener("ended", () => {
  el("play").textContent = "Play";
  sync(true);
});
video.addEventListener("error", () =>
  message(
    "The browser could not decode this MP4. Check the recording files and try a Chromium browser.",
    true,
  ),
);
for (const peer of peers)
  peer.addEventListener("loadedmetadata", () => sync(true));
el("reference").onclick = async () => {
  if (!snapshot?.current_completed) return;
  video.pause();
  el("reference").disabled = true;
  try {
    await request("/api/references", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: current.run_id,
        vial_id: snapshot.current_completed.vial_id,
        frame_index: sourceFrame(),
      }),
    });
    await tick(true);
    message("Development reference saved. Only later fills can use it.");
  } catch (error) {
    message(error.message, true);
  } finally {
    el("reference").disabled = !snapshot?.current_completed?.measurement_valid || snapshot?.current_completed?.recipe_id == null || !!snapshot?.current_completed?.context_quality_flags?.length;
  }
};
async function start() {
  catalog = await request("/api/catalog");
  el("run").replaceChildren();
  for (const run of catalog.runs) {
    el("run").add(
      new Option(
        `Run ${run.run_id}${catalog.fill_source === "image" ? " · image detection" : run.annotations.length ? " · " + run.annotations.length + " annotations" : " · unlabeled"}`,
        String(run.run_id),
      ),
    );
  }
  const params = new URLSearchParams(location.search);
  const requestedRun = Number(params.get("run") || 5);
  el("run").value = String(catalog.runs.find(r => r.run_id === requestedRun)?.run_id || catalog.runs[0].run_id);
  overlays();
  await selectRun();
  if (params.has("frame") && /^\d+$/.test(params.get("frame"))) seekToFrame(Number(params.get("frame")));
}
setInterval(() => tick().catch((error) => message(error.message, true)), 500);
start().catch((error) => message(error.message, true));
