"use strict";
const el = id => document.getElementById(id);
const video = el("video");
let session, snapshot, selectedCase = null, generation = 0, controller = null;
let dialogContext = null, toastTimer = null, refreshTimer = null, evidenceKey = null;
const time = seconds => {
  const n = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(n / 60).toString().padStart(2,"0")}:${(n % 60).toFixed(1).padStart(4,"0")}`;
};
const words = value => String(value || "").replaceAll("_", " ");
const label = (id, value) => { el(id).textContent = value; };
const show = (id, visible) => { el(id).hidden = !visible; };

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The workspace could not save this request.");
  return data;
}
function announce(value) {
  label("toast", value); show("toast", true);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => show("toast", false), 3500);
}
function rememberActor(value) {
  el("operator-name").value = value;
  try { localStorage.setItem("first-vial-operator", value); } catch (_) { /* local identity is optional */ }
}
function paintSwatch(id, rgb) {
  const ctx = el(id).getContext("2d");
  ctx.clearRect(0,0,28,28);
  ctx.fillStyle = rgb ? `rgb(${rgb.map(n=>Math.round(n)).join(",")})` : "#e8eee8";
  ctx.fillRect(0,0,28,28);
}
function samplePreview() {
  if (!snapshot || !el("sample-details").open) return;
  const row = snapshot.sample, kind = el("evidence-kind").value;
  const key = `${row.source_frame}/${kind}`;
  if (key === evidenceKey) return;
  evidenceKey = key;
  const image = el("evidence-image"); image.hidden = true;
  label("evidence-clock", `Sample frame ${row.source_frame} · ${time(row.seconds)}`);
  image.onload = () => { if (evidenceKey === key && el("sample-details").open) image.hidden = false; };
  image.onerror = () => { if (evidenceKey === key) { image.hidden = true; label("evidence-clock", "Exact source image is unavailable. Keep the evidence unresolved."); } };
  image.src = `/api/evidence?frame=${row.source_frame}&at_s=${row.seconds}&kind=${encodeURIComponent(kind)}`;
}
function status(kicker, title, reason, tone = "neutral") {
  label("assessment-kicker", kicker); label("assessment-title", title); label("assessment-reason", reason);
  el("assessment-status").className = `assessment-status ${tone}`;
}
function renderAssessment(current) {
  for (const id of ["start-vial","complete-vial","ack-review","record-action","close-review","request-review","case-context","comparison","historical-note","record-tools"]) show(id, false);
  if (!current) {
    label("case-badge", "No record"); el("case-badge").className = "badge neutral";
    status("CONTEXT NEEDED", "Start with the intended recipe", "Create a vial record to connect this image with an operator-supplied recipe and physical vial.");
    show("start-vial", true); return;
  }
  show("case-context", true); show("record-tools", true);
  label("case-recipe", current.recipe_id || "Unknown");
  label("case-vial", [current.batch_id, current.physical_vial_id || "Unlinked vial"].filter(Boolean).join(" / "));
  label("case-completion", current.lifecycle === "complete" ? `Operator confirmed · ${time(current.endpoint.seconds)}` : current.lifecycle === "closed_uncertain" ? "Closed as uncertain" : "Not confirmed");
  label("case-badge", current.lifecycle === "complete" ? "Endpoint captured" : current.lifecycle === "closed_uncertain" ? "Uncertain" : "In progress");
  el("case-badge").className = "badge neutral";
  const review = current.review_state, a = current.assessment;
  if (review === "requested") {
    status("OPERATOR FOLLOW-UP", "Review requested", current.review_reason, "amber");
    show("ack-review", true); label("case-badge", "Needs response"); el("case-badge").className = "badge amber";
  } else if (review === "acknowledged") {
    const responded = current.actions.some(action => action.event_seq > current.review_cycle_start);
    status("REVIEW ACKNOWLEDGED", responded ? "Operator response recorded" : "Record the response", responded ? "Keep the follow-up open until the response is documented and the vial record is complete or closed as uncertain." : "Acknowledgement is saved. Record what the operator actually did next.", "amber");
    show("record-action", true); show("close-review", responded && current.lifecycle !== "open");
  } else if (review === "closed") {
    status("FOLLOW-UP CLOSED", "Response documented", "The operator record is closed. Independent product-quality verification remains separate.");
    show("request-review", true);
  } else if (current.lifecycle === "open") {
    status("COMPLETION NEEDED", "Observe the current fill", "Confirm that dispensing has finished before capturing an endpoint comparison. An ingredient pause is not fill completion.");
    show("request-review", true);
  } else if (a.diagnostic_status === "deviation") {
    status("DEVELOPMENT COMPARISON", "Appearance differs", "The sample differs from an earlier development reference. Recipe assessment remains unavailable: the reference and model geometry need review.", "amber");
    show("request-review", true);
  } else if (a.diagnostic_status === "within_band") {
    status("DEVELOPMENT COMPARISON", "Within the development band", "Appearance agrees with the selected development reference. This does not establish product acceptance or close an open review.");
    show("request-review", true);
  } else {
    status("RECIPE ASSESSMENT UNAVAILABLE", "More evidence needed", a.reason);
    show("request-review", true);
  }
  show("complete-vial", current.lifecycle === "open");
  if (current.endpoint) {
    show("comparison", true);
    paintSwatch("observed-swatch", current.endpoint.rgb); paintSwatch("reference-swatch", a.reference?.rgb);
    label("observed-value", current.endpoint.measurement_valid ? "Sample checks pass" : "Unavailable");
    label("reference-value", a.reference ? "Development only" : "Not available");
    label("delta-value", a.delta_e2000 === null ? "—" : a.delta_e2000.toFixed(2));
    label("delta-band", a.threshold_de2000.toFixed(1));
  }
  const hasReference = snapshot.references.some(r => r.source_case_id === current.case_id);
  el("save-reference").disabled = !current.endpoint?.measurement_valid || !current.recipe_id || !current.identity_confirmed || hasReference || !current.editable;
  el("abandon-vial").disabled = current.lifecycle !== "open" || !current.editable;
  if (!current.editable) {
    show("historical-note", true);
    for (const button of el("primary-actions").querySelectorAll("button")) button.disabled = true;
  } else {
    for (const button of el("primary-actions").querySelectorAll("button")) button.disabled = false;
  }
}
function renderLedger(cases) {
  const body = el("ledger"); body.replaceChildren();
  if (!cases.length) {
    const row = document.createElement("tr"), cell = document.createElement("td");
    cell.colSpan = 6; cell.className = "empty-row"; cell.textContent = "No vial records yet. Explore a fill neighborhood, then create a record.";
    row.append(cell); body.append(row); return;
  }
  for (const record of [...cases].reverse()) {
    const row = document.createElement("tr");
    const values = [
      [record.physical_vial_id || "Unlinked vial", record.batch_id || "No batch entered"],
      [record.recipe_id || "Unknown", record.identity_confirmed ? "Manual context" : "Identity unconfirmed"],
      [record.endpoint ? (record.endpoint.measurement_valid ? "Checks pass" : "Unavailable") : "Not captured", record.endpoint ? time(record.endpoint.seconds) : "Completion not confirmed"],
      ["Unavailable", record.assessment.diagnostic_status ? `Development: ${words(record.assessment.diagnostic_status)}` : "No qualified comparison"],
      [record.review_state === "none" ? "No review requested" : words(record.review_state), record.actions.length ? `${record.actions.length} human action${record.actions.length === 1 ? "" : "s"}` : "No action recorded"]
    ];
    for (const [heading, sub] of values) {
      const cell = document.createElement("td"), strong = document.createElement("strong"), small = document.createElement("small");
      strong.textContent = heading; small.textContent = sub; cell.append(strong, small); row.append(cell);
    }
    const cell = document.createElement("td"), button = document.createElement("button");
    button.className = "button secondary"; button.textContent = "View"; button.setAttribute("aria-label", `View ${record.physical_vial_id || "vial record"}`);
    button.addEventListener("click", () => { selectedCase = record.case_id; refresh(); el("assessment-status").scrollIntoView({behavior:"smooth",block:"center"}); });
    cell.append(button); row.append(cell); body.append(row);
  }
}
function renderActivity(events) {
  const list = el("activity"); list.replaceChildren();
  const current = snapshot.current;
  const rows = events.filter(e => !current || e.case_id === current.case_id).slice(-5).reverse();
  label("activity-count", `${rows.length} recent event${rows.length === 1 ? "" : "s"}`);
  if (!rows.length) {
    const li = document.createElement("li"); li.className = "empty-activity";
    li.textContent = "Acknowledgements and actual operator actions will appear here."; list.append(li); return;
  }
  const titles = {vial_started:"Vial context entered",fill_completed:"Fill completion confirmed",vial_abandoned:"Record closed as uncertain",review_requested:"Manual review requested",review_acknowledged:"Review acknowledged",operator_action_recorded:"Human action recorded",follow_up_closed:"Follow-up closed",development_reference_saved:"Development reference saved"};
  for (const item of rows) {
    const li = document.createElement("li"), title = document.createElement("strong"), detail = document.createElement("small");
    title.textContent = titles[item.kind] || words(item.kind);
    detail.textContent = `${item.actor} · source ${time(item.seconds)}${item.payload.action ? ` · ${words(item.payload.action)}` : ""}`;
    li.append(title,detail); list.append(li);
  }
}
function render() {
  const current = snapshot.current, row = snapshot.sample, counts = snapshot.counts;
  label("video-clock", `${time(video.currentTime)} / ${time(snapshot.source.duration_s)}`);
  label("metric-recipe", current?.recipe_id || "Not entered");
  label("metric-batch", current?.batch_id ? `Batch ${current.batch_id}` : "Add context from the operator");
  label("metric-vial", current?.physical_vial_id || "Not linked");
  label("metric-identity", current?.identity_confirmed ? "Operator-associated identity" : "Visual IDs are separate");
  label("metric-qa", row.measurement_valid ? "Sample checks pass" : "Unavailable");
  label("metric-sample", `Sample ${time(row.seconds)} · ${row.mask_pixels.toLocaleString()} pixels`);
  label("metric-followup", counts.open_reviews ? `${counts.open_reviews} open review${counts.open_reviews === 1 ? "" : "s"}` : "No open reviews");
  label("metric-action", `${counts.actions} operator action${counts.actions === 1 ? "" : "s"} recorded`);
  label("nav-count", String(counts.vials));
  el("new-vial").disabled = snapshot.cases.some(c => c.lifecycle === "open");
  label("sample-status", row.measurement_valid ? "Sampling checks pass · geometry unreviewed" : "Image sample unavailable");
  label("sample-caption", row.measurement_valid ? `Source sample ${time(row.seconds)} · no recipe approval` : words(row.quality_flags[0] || "No usable sample at this source point"));
  el("sample-dot").className = `signal${row.measurement_valid ? " valid" : ""}`;
  label("sample-phase", `Image: ${words(row.image_phase)}`);
  renderAssessment(current); renderLedger(snapshot.cases); renderActivity(snapshot.events); samplePreview();
  const query = `at_s=${encodeURIComponent(video.currentTime)}`;
  el("export-header").href = el("export-json").href = `/api/export.json?${query}`;
  el("export-csv").href = `/api/export.csv?${query}`;
  label("provenance", JSON.stringify({session_id:snapshot.session_id,mode:snapshot.source_mode,source:session.source.source_name,
    source_identity:session.source.identity,sample_frame:row.source_frame,native_pts:row.pts,time_base:row.time_base,
    geometry:"unreviewed model proposals",reference_mode:"same-recording development",hardware_commands:0,
    metered_progress:"not available",independent_outcomes:"not available"},null,2));
}
async function refresh() {
  if (!session) return;
  const token = ++generation, at = video.currentTime || 0;
  if (controller) controller.abort(); controller = new AbortController();
  try {
    const query = `at_s=${encodeURIComponent(at)}${selectedCase ? `&case_id=${encodeURIComponent(selectedCase)}` : ""}`;
    const data = await request(`/api/state?${query}`, {signal:controller.signal});
    if (token !== generation) return false;
    snapshot = data; show("page-error",false); render(); return true;
  } catch (error) {
    if (error.name !== "AbortError" && token === generation) {
      label("page-error", `Evidence unavailable: ${error.message}`); show("page-error",true);
    }
    return false;
  }
}
function scheduleRefresh() { clearTimeout(refreshTimer); refreshTimer = setTimeout(refresh,80); }
async function pauseAndCapture() {
  video.pause(); clearTimeout(refreshTimer);
  const at = video.currentTime, captured = await refresh();
  if (!captured || !snapshot || Math.abs(video.currentTime-at) > .001 || !el("page-error").hidden) throw new Error("Wait for a valid source snapshot before recording an action.");
  return {source_frame:snapshot.sample.source_frame,case_id:snapshot.current?.case_id,
    expected_revision:snapshot.current?.revision,request_id:crypto.randomUUID()};
}
function clearFormError(dialog) { const e = dialog.querySelector(".form-error"); e.hidden = true; e.textContent = ""; }
function formError(dialog, error) { const e = dialog.querySelector(".form-error"); e.textContent = error.message; e.hidden = false; }
async function openStart() {
  try {
    dialogContext = await pauseAndCapture(); dialogContext.command = "start";
    const dialog = el("start-dialog"); el("start-form").reset(); clearFormError(dialog);
    el("start-actor").value = el("operator-name").value; dialog.showModal();
  } catch(error) { label("page-error",error.message); show("page-error",true); }
}
const descriptions = {
  complete:{title:"Confirm fill completion",description:"Confirm that dispensing for this vial has finished. The workspace captures the last three source samples at this point; failed checks stay unavailable.",button:"Confirm & capture endpoint",confirmation:"I confirm dispensing has finished for this vial.",note:false},
  request_review:{title:"Request manual review",description:"Record why this vial needs attention. The request remains open until a human response is documented.",button:"Save review request",note:true},
  acknowledge:{title:"Acknowledge the review",description:"Record that you have seen this request. Acknowledgement leaves the review open and sends no equipment command.",button:"Save acknowledgement",note:false},
  record_action:{title:"Record the operator response",description:"Record what a person actually did in this recorded demonstration. This form neither performs an action nor changes the line.",button:"Save performed action",confirmation:"I confirm the described human action was performed in this demonstration.",note:true},
  close_review:{title:"Close the follow-up record",description:"Document the follow-up outcome. Closing this journal entry does not release product or establish independent quality truth.",button:"Close follow-up record",note:true},
  save_reference:{title:"Save a development reference",description:"Use this endpoint for practice comparisons with later records of the same intended recipe. It remains unreviewed and cannot approve product or become a training label.",button:"Save development reference",note:true},
  abandon:{title:"Close incomplete record",description:"Preserve this attempt as uncertain. It will not be treated as a completed fill or an acceptable reference.",button:"Close as uncertain",note:true}
};
async function openCommand(command) {
  try {
    const context = await pauseAndCapture();
    if (!snapshot.current || !snapshot.current.editable) throw new Error("Select an editable vial record at its latest source point.");
    dialogContext = {...context,command};
    const definition = descriptions[command], dialog = el("command-dialog");
    el("command-form").reset(); clearFormError(dialog);
    label("command-title",definition.title); label("command-description",definition.description); label("command-save",definition.button);
    el("command-actor").value = el("operator-name").value;
    show("command-note-label",definition.note); el("command-note").required = definition.note;
    show("confirmation-label",Boolean(definition.confirmation)); el("command-confirmed").required = Boolean(definition.confirmation);
    if (definition.confirmation) label("confirmation-text",definition.confirmation);
    show("action-fields",command === "record_action"); show("procedure-label",false); el("procedure-id").required = false;
    dialog.showModal();
  } catch(error) { label("page-error",error.message); show("page-error",true); }
}
async function submit(dialog, body) {
  const button = dialog.querySelector('button[type="submit"]'); button.disabled = true; clearFormError(dialog);
  try {
    const result = await request("/api/command",{method:"POST",headers:{"Content-Type":"application/json","X-Operator-CSRF":session.csrf_token},body:JSON.stringify({...dialogContext,...body})});
    selectedCase = result.case_id; rememberActor(body.actor); dialog.close();
    await refresh(); announce(body.command === "record_action" ? "Human action recorded" : "Operator record saved");
  } catch(error) { formError(dialog,error); }
  finally { button.disabled = false; }
}
el("start-form").addEventListener("submit",event=>{
  event.preventDefault();
  submit(el("start-dialog"),{command:"start",actor:el("start-actor").value,recipe_id:el("start-recipe").value,
    batch_id:el("start-batch").value,physical_vial_id:el("start-physical").value,
    identity_confirmed:el("start-identity").checked,note:el("start-note").value});
});
el("command-form").addEventListener("submit",event=>{
  event.preventDefault();
  submit(el("command-dialog"),{command:dialogContext.command,actor:el("command-actor").value,note:el("command-note").value,
    confirmed:el("command-confirmed").checked,performed:el("command-confirmed").checked,
    action:el("action-type").value,reported_station_route:el("station-route").value,procedure_id:el("procedure-id").value});
});
for (const button of document.querySelectorAll("[data-close]")) button.addEventListener("click",()=>el(button.dataset.close).close());
el("new-vial").addEventListener("click",openStart); el("start-vial").addEventListener("click",openStart);
for (const [id,command] of Object.entries({"complete-vial":"complete","request-review":"request_review","ack-review":"acknowledge","record-action":"record_action","close-review":"close_review","save-reference":"save_reference","abandon-vial":"abandon"})) el(id).addEventListener("click",()=>openCommand(command));
el("action-type").addEventListener("change",()=>{ const needed = el("action-type").value === "approved_procedure"; show("procedure-label",needed); el("procedure-id").required = needed; });
el("operator-name").addEventListener("change",()=>rememberActor(el("operator-name").value));
el("sample-jump").addEventListener("change",()=>{if(el("sample-jump").value !== ""){video.pause();selectedCase = null;video.currentTime = Number(el("sample-jump").value);}});
el("speed").addEventListener("change",()=>{video.playbackRate = Number(el("speed").value);});
el("sample-details").addEventListener("toggle",()=>{if(el("sample-details").open)samplePreview();else{evidenceKey=null;el("evidence-image").removeAttribute("src");el("evidence-image").hidden=true;}});
el("evidence-kind").addEventListener("change",()=>{evidenceKey=null;samplePreview();});
video.addEventListener("timeupdate",scheduleRefresh); video.addEventListener("seeked",refresh);
video.addEventListener("seeking",()=>{generation++;evidenceKey=null;el("evidence-image").hidden=true;});
video.addEventListener("error",()=>{label("page-error","The saved recording could not be played. Keep source evidence unavailable and check the local file.");show("page-error",true);});
async function boot() {
  try {
    session = await request("/api/session");
    label("source-name",`Run ${session.source.run_id ?? "—"} · Vial fill`);
    for (const item of session.source.navigation) {
      const option=document.createElement("option"); option.value=item.seconds;
      option.textContent=`${item.label} · ${time(item.seconds)}`;el("sample-jump").append(option);
    }
    try { el("operator-name").value=localStorage.getItem("first-vial-operator") || ""; } catch (_) { /* no persistence available */ }
    await refresh();
  } catch(error) {label("page-error",error.message);show("page-error",true);}
}
boot();
