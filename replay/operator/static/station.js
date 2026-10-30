"use strict";
const $=id=>document.getElementById(id),words=v=>String(v||"").replaceAll("_"," ");
let state=null,csrf=null,selected=null,dialogContext=null,imageKey=null,imageGeneration=0,reviewKey=null,localError=null,pending=false;
let videoStream=null,lastDeliveryMode=null,cvKey=null,usingVideo=false;
const label=(id,value)=>{$(id).textContent=value;};
const visible=(id,value)=>{$(id).hidden=!value;};
const current=()=>state?.cases.find(c=>c.case_id===state.current_id);
const reviewCase=()=>state?.cases.find(c=>c.case_id===selected);
const recipeLabel=value=>state?.source.kind==="isaac"&&value==="synthetic-10-10-10-order-0-1-2"?"10 / 10 / 10 mL":value||"Unknown";
const clock=s=>`${(s.pts*Number(s.time_base.split("/")[0])/Number(s.time_base.split("/")[1]||1)).toFixed(1)} s · sample ${s.sequence}`;
const qaNames={vial_track_not_stable:"Vial track is still settling",liquid_missing_or_ambiguous_inside_vial:"No unambiguous liquid found inside the vial",insufficient_final_liquid_pixels:"Too few usable liquid pixels",insufficient_unclipped_pixels:"Too few unclipped pixels",insufficient_coherent_colored_liquid:"Liquid color region is not coherent enough",carrier_motion:"Motion check failed",no_prior_frame_for_motion_check:"Waiting for a preceding frame"};
const qaReason=row=>row?.measurement.quality_flags.map(flag=>qaNames[flag]||words(flag)).join("; ");
async function request(path,options={}){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw new Error(data.error||"Request failed");return data;}
function apply(data){if(!state||data.version>=state.version){state=data;render();}}
function showLive(row){
  const playback=state.playback,videoMode=Boolean(playback&&state.source.delivery_mode==="prerender"&&$("view").value==="source"),video=$("source-video");
  if(videoMode!==usingVideo){usingVideo=videoMode;imageGeneration++;imageKey=null;}
  visible("cv-preview",Boolean(playback));visible("source-video",videoMode);
  $("camera-streams").classList.toggle("has-playback",Boolean(playback));
  if(videoMode){
    if(!video.getAttribute("src"))video.src=playback.video_url;
    const newStream=videoStream!==playback.stream_id;videoStream=playback.stream_id;
    if(video.readyState>=1&&(newStream||Math.abs(video.currentTime-playback.seconds)>.3))video.currentTime=Math.min(playback.seconds,video.duration||playback.duration_s);
    video.playbackRate=1;
    if(playback.paused||state.source.status==="error"||state.source.status==="switching")video.pause();
    else if(video.paused)video.play().catch(e=>{if(e.name!=="AbortError")localError="Video playback unavailable: "+e.message;});
    visible("live-image",false);visible("camera-empty",false);label("image-clock",`Pre-rendered Isaac video · ${video.currentTime.toFixed(1)} s · 1× / 30 fps`);
  }else video.pause();
  if(playback){
    const lag=playback.analysis_lag_s;label("cv-lag",lag===null?"Waiting for this loop's analysis":`${lag.toFixed(2)} s CV lag`);
    if(row&&$("cv-preview").open&&cvKey!==row.image_key){cvKey=row.image_key;$("cv-image").src=`/api/image?key=${encodeURIComponent(row.image_key)}&kind=overlay`;label("cv-clock",`Analyzed: ${clock(row.source)} · ${playback.analysis_lag_s===null?"previous loop":playback.analysis_lag_s.toFixed(2)+" s behind video"}`);label("cv-mask-status",row.measurement.geometry.mask_pixels?`${row.measurement.geometry.mask_pixels.toLocaleString()} mask pixels · ${row.measurement.measurement_valid?"sampling usable":"sampling checks failed"}`:"No usable liquid mask in this sample");}
  }
  if(videoMode)return;
  if(!row){imageGeneration++;visible("live-image",false);visible("camera-empty",true);return;}
  const key=row.image_key+"/"+$("view").value;if(key===imageKey)return;imageKey=key;
  const generation=++imageGeneration,image=new Image();
  image.onload=()=>{if(generation!==imageGeneration)return;$("live-image").src=image.src;$("live-image").dataset.key=row.image_key;visible("live-image",true);visible("camera-empty",false);label("image-clock",`${row.source.source_id} · ${clock(row.source)}`);};
  image.onerror=()=>{if(generation===imageGeneration){visible("live-image",false);label("image-clock","Source image unavailable; waiting for the next frame");imageKey=null;}};
  image.src=`/api/image?key=${encodeURIComponent(row.image_key)}&kind=${encodeURIComponent($("view").value)}`;
}
function renderQueue(){
  const reviews=state.cases.filter(c=>c.review!=="none").reverse();
  if(!selected&&reviews.length)selected=(reviews.find(c=>c.review!=="closed")||reviews[0]).case_id;
  const queue=$("queue");queue.replaceChildren();
  for(const c of reviews){const b=document.createElement("button"),strong=document.createElement("strong"),small=document.createElement("small"),status=document.createElement("span");
    b.className="queue-item"+(c.case_id===selected?" selected":"");b.dataset.case=c.case_id;
    strong.textContent=c.physical_vial_id||c.case_id;small.textContent=c.review_reason;status.textContent=words(c.review);
    b.append(strong,small,status);b.addEventListener("click",()=>{selected=c.case_id;renderQueue();});queue.append(b);}
  if(!reviews.length){const p=document.createElement("p");p.className="queue-empty";p.textContent="No review requests.";queue.append(p);}
  label("queue-summary",`${state.open_reviews} open review${state.open_reviews===1?"":"s"}`);
  const c=reviewCase();visible("review-detail",Boolean(c));visible("review-empty",!c);
  if(!c)return;
  label("review-title",c.physical_vial_id||c.case_id);label("review-reason",c.review_reason);label("review-state",words(c.review));
  const evidence=c.evidence,key=evidence?c.case_id+"/"+c.review_id:null;
  if(evidence){
    label("review-clock",clock(evidence.source));label("review-identity",`${evidence.source.source_id} · ${recipeLabel(c.recipe_id)}${c.physical_vial_id?` · vial ${c.physical_vial_id}`:" · physical vial unconfirmed"}`);
    label("review-measurement",evidence.diagnostic.delta_e2000===null?evidence.measurement.quality_flags.join(", ")||"No eligible color comparison":`Captured color difference ${evidence.diagnostic.delta_e2000.toFixed(2)} dE · development comparison`);
    if(key!==reviewKey){reviewKey=key;$("review-image").src=`/api/image?case_id=${encodeURIComponent(c.case_id)}&kind=overlay`;$("review-image").dataset.case=c.case_id;}
  }
  visible("acknowledge",c.review==="requested");visible("record-action",c.review==="acknowledged");
  visible("close-review",c.review==="acknowledged"&&c.actions.some(a=>a.cycle===c.review_cycle));
  const history=$("action-history");history.replaceChildren();
  for(const a of c.actions){const p=document.createElement("p");p.textContent=`${a.actor} · ${words(a.action)}: ${a.note}`;history.append(p);}
  if(c.resolution){const p=document.createElement("p");p.textContent=`Follow-up: ${c.resolution.note} · product release remains unauthorized`;history.append(p);}
}
function render(){
  const c=current(),row=state.latest,source=state.source,live=source.status==="live",d=row?.diagnostic;
  if(lastDeliveryMode!==source.delivery_mode){lastDeliveryMode=source.delivery_mode;imageKey=null;imageGeneration++;if(source.delivery_mode==="prerender")$("view").value="source";}
  visible("source-mode-label",source.kind==="isaac");if(!pending)$("source-mode").value=source.requested_mode||"isaac";
  $("source-mode").querySelector('option[value="prerender"]').disabled=!(source.available_modes||[]).includes("prerender");
  label("source-name",source.kind==="isaac"?(source.delivery_mode==="prerender"?"Pre-rendered Isaac · live CV":"Live Isaac rendering"):"Local frame input");label("source-pill",source.kind==="isaac"?"SIMULATED SOURCE":"FRAME INPUT");
  label("source-status",words(source.status));label("source-performance",row?`${source.delivery_mode==="prerender"?"30 fps video · ":""}${source.ingest_fps.toFixed(1)} CV samples/s · ${row.inference_ms.toFixed(0)} ms inference`:"Loading source and models");
  label("recipe",recipeLabel(c?.recipe_id||state.context.recipe_id));label("batch",c?.batch_id?`Batch ${c.batch_id}`:c?.context_origin==="simulated_job_configuration"?"Simulated job · order 0 → 1 → 2":"Operator context");
  label("vial",c?.physical_vial_id||c?.case_id||"Waiting");label("identity",c?.identity_confirmed?"Operator-associated physical vial":"Physical vial unconfirmed");
  label("sample-quality",!live?"Unavailable":row.measurement.measurement_valid?"Sampling usable":"Unavailable");
  label("sample-summary",row?`${row.measurement.geometry.mask_pixels.toLocaleString()} sampling pixels · geometry unreviewed`:"Quality checks required");
  label("review-count",`${state.open_reviews} open review${state.open_reviews===1?"":"s"}`);label("nav-reviews",String(state.open_reviews));label("frame-count",`${state.counts.frames.toLocaleString()} samples received`);
  label("live-badge",live?"Live":words(source.status));$("live-image").parentElement.classList.toggle("stale",!live);
  label("image-age",source.age_s===null?"No sample yet":`${source.age_s.toFixed(1)} s since last processed sample`);
  label("qa-status",!live?"Fresh evidence unavailable":row.measurement.measurement_valid?"Sampling checks pass · geometry unreviewed":"Measurement unavailable");
  label("qa-reason",!live?source.error||"Waiting for fresh incoming frames":qaReason(row)||"Model sampling pixels are source-aligned");
  $("qa-dot").className="signal"+(live&&row?.measurement.measurement_valid?" valid":"");
  label("source-phase",row?.source.telemetry?`Simulated ingredient ${row.source.telemetry.phase_key}`:"Process progress unknown");
  let title="Waiting for fresh frames",reason=source.error||"The station processes incoming frames automatically.",amber=false;
  if(live&&row){
    if(d.warning){title="Appearance differs";reason="Confirmed development deviation: captured evidence is retained for this observation. "+(row.measurement.measurement_valid?"Inspect the review queue.":"This frame also fails sampling checks; the earlier captured alert remains available.");amber=true;}
    else if(!row.measurement.measurement_valid){title="Measurement unavailable";reason=qaReason(row)||"Sampling checks failed for this frame. Review its image and evidence.";}
    else if(d.delta_e2000!==null&&d.delta_e2000>d.threshold_de2000){title="Outside development band";reason=`Current reading exceeds ${d.threshold_de2000.toFixed(1)} dE. Confirmation pending (${d.consecutive}/3 consecutive valid samples); gaps and failed checks reset confirmation.`;amber=true;}
    else if(d.delta_e2000!==null){title="Within development band";reason="Current appearance agrees with the configured development reference. Recipe acceptance remains unavailable.";}
    else{title="Liquid observation available";reason=d.reason+". Intended recipe acceptance remains unavailable.";}
  }
  label("assessment-title",title);label("assessment-reason",reason);label("assessment-kicker",d?.warning?"REVIEW REQUESTED":amber?"CONFIRMATION PENDING":"CURRENT EVIDENCE");
  label("assessment-badge",d?.warning?"Needs review":amber?"Pending":"Advisory");$("assessment").className="assessment-status "+(amber?"amber":"neutral");
  label("delta",live&&d?.delta_e2000!==null&&d?.delta_e2000!==undefined?`${d.delta_e2000.toFixed(2)} dE`:"—");
  label("comparison-status",!live?"Evidence unavailable":d?.delta_e2000!==null&&d?.delta_e2000!==undefined?"Development reference":"No eligible comparison");
  $("context").disabled=!c;$("new-vial").disabled=!c||!live;$("request-review").disabled=!c||!live||["requested","acknowledged"].includes(c.review);$("complete").disabled=!c||!live||c.fill_complete;
  label("request-review",state.playback&&$("view").value==="source"?"Review latest CV sample":"Request review of this frame");
  renderQueue();showLive(row);
  const ledger=$("ledger");ledger.replaceChildren();
  for(const item of [...state.cases].reverse().slice(0,30)){const tr=document.createElement("tr");for(const value of [item.physical_vial_id||item.case_id,recipeLabel(item.recipe_id),`${item.qa_pass}/${item.frames}`,item.comparisons?`${item.comparisons} samples · peak ${item.max_delta_e2000.toFixed(2)} dE`:"Unavailable",item.review==="none"?"No review requested":words(item.review)]){const td=document.createElement("td");td.textContent=value;tr.append(td);}ledger.append(tr);}
  const list=$("activity");list.replaceChildren();
  for(const e of state.events.filter(e=>["review_requested","review_acknowledged","operator_action_recorded","follow_up_closed","context_recorded","fill_completion_recorded"].includes(e.kind)).slice(-5).reverse()){
    const li=document.createElement("li"),strong=document.createElement("strong"),small=document.createElement("small");strong.textContent=words(e.kind);small.textContent=`${e.actor||"Vision pipeline"} · ${e.payload.case_id||""}`;li.append(strong,small);list.append(li);}
  if(!list.children.length){const li=document.createElement("li");li.className="empty-activity";li.textContent="Operator responses will appear here.";list.append(li);}
  visible("pause-source",source.kind==="isaac");label("pause-source",source.paused?"Resume simulated test feed":"Pause simulated test feed");
  label("source-description",source.kind==="isaac"?"Continuous Isaac camera frames run through the frozen vision pipeline. Test-feed controls affect the simulator source only.":"This source accepts local frame input. Recipe and vial context stay separate from the image feed; no phase or meter signal is inferred.");
  label("provenance",JSON.stringify({session_id:state.session_id,source,frame:row?.source,geometry:state.geometry_status,reference:state.reference,recipe_acceptance:"unavailable",hardware_commands:0},null,2));
  label("footer-source",source.kind==="isaac"?"Simulated source · no equipment control":"Local frame source · no equipment control");
  const error=source.error||localError;visible("error",Boolean(error));if(error)label("error",error);
}
function actor(){const name=$("actor").value.trim();if(!name)throw new Error("Enter your operator name first.");try{localStorage.setItem("first-vial-station-actor",name);}catch(_){}return name;}
function bind(c,command){return {case_id:c.case_id,expected_revision:c.revision,command,request_id:crypto.randomUUID()};}
async function submit(body,dialog){
  if(pending)return;pending=true;
  try{const who=actor();const result=await request("/api/command",{method:"POST",headers:{"Content-Type":"application/json","X-Operator-CSRF":csrf},body:JSON.stringify({...body,actor:who})});localError=null;if(dialog)dialog.close();if(result.case_id&&body.command==="request_review")selected=result.case_id;apply(await request("/api/state"));}
  catch(e){if(dialog){const error=dialog.querySelector(".form-error");error.textContent=e.message;error.hidden=false;}else{localError=e.message;render();}}
  finally{pending=false;}
}
const definitions={request_review:["Request review","Describe what needs a human decision. The frame captured when you opened this form stays fixed while the feed continues.",true,false],complete:["Confirm dispensing finished","Confirm that the vial's complete dispensing operation has ended. Ingredient pauses are separate. Unusable endpoint samples remain unavailable.",false,true],new_vial:["Create a new vial record","Mark the current source point as a new operator-observed vial. Its physical identity remains unconfirmed until you enter it.",true,false],action:["Record the operator action","Record what the person actually did. This entry does not perform the action or send an equipment command.",true,true],close:["Close the follow-up","Document the response and remaining uncertainty. This closes the review record and does not authorize product release.",true,false]};
async function openAction(command,c){
  try{actor();if(!c)throw new Error("Wait for an observation or select a review.");dialogContext=bind(c,command);const dialog=$("action-dialog");$("action-form").reset();dialog.querySelector(".form-error").hidden=true;
    visible("manual-capture",false);
    if(command==="request_review"){
      const result=await request("/api/command",{method:"POST",headers:{"Content-Type":"application/json","X-Operator-CSRF":csrf},body:JSON.stringify({...bind(c,"capture"),actor:actor(),image_key:state.latest.image_key})});
      dialogContext.capture_id=result.capture_id;$("manual-capture-image").src=`/api/image?capture_id=${encodeURIComponent(result.capture_id)}&kind=overlay`;label("manual-capture-clock",clock(result.evidence.source));visible("manual-capture",true);
    }
    const [title,description,note,confirm]=definitions[command];label("dialog-title",title);label("dialog-description",description);visible("note-label",note);$("action-note").required=note;visible("confirm-label",confirm);$("action-confirm").required=confirm;visible("action-fields",command==="action");visible("procedure-label",false);$("procedure-id").required=false;label("confirm-text",command==="complete"?"I confirm dispensing for this vial has finished.":"I confirm the described human action was performed.");dialog.showModal();
  }catch(e){localError=e.message;render();}
}
$("context").addEventListener("click",()=>{try{actor();const c=current();if(!c)return;dialogContext=bind(c,"context");const d=$("context-dialog");$("context-form").reset();d.querySelector(".form-error").hidden=true;$("context-recipe").value=c.recipe_id;$("context-batch").value=c.batch_id;$("context-vial").value=c.physical_vial_id;$("context-identity").checked=c.identity_confirmed;d.showModal();}catch(e){localError=e.message;render();}});
$("context-form").addEventListener("submit",event=>{event.preventDefault();submit({...dialogContext,recipe_id:$("context-recipe").value,batch_id:$("context-batch").value,physical_vial_id:$("context-vial").value,identity_confirmed:$("context-identity").checked,apply_next:$("context-next").checked,note:$("context-note").value},$("context-dialog"));});
for(const [id,command] of Object.entries({"request-review":"request_review","complete":"complete","new-vial":"new_vial"}))$(id).addEventListener("click",()=>openAction(command,current()));
$("acknowledge").addEventListener("click",()=>{const c=reviewCase();if(c)submit(bind(c,"acknowledge"));});
$("record-action").addEventListener("click",()=>openAction("action",reviewCase()));$("close-review").addEventListener("click",()=>openAction("close",reviewCase()));
$("action-form").addEventListener("submit",event=>{event.preventDefault();submit({...dialogContext,note:$("action-note").value,confirmed:$("action-confirm").checked,action:$("action-type").value,procedure_id:$("procedure-id").value},$("action-dialog"));});
$("action-type").addEventListener("change",()=>{const needed=$("action-type").value==="approved_procedure";visible("procedure-label",needed);$("procedure-id").required=needed;});
for(const button of document.querySelectorAll("[data-close]"))button.addEventListener("click",()=>$(button.dataset.close).close());
$("view").addEventListener("change",()=>{imageKey=null;showLive(state.latest);});
$("pause-source").addEventListener("click",()=>submit({command:"source_pause",paused:!state.source.paused,request_id:crypto.randomUUID()}));
$("source-mode").addEventListener("change",()=>submit({command:"source_mode",mode:$("source-mode").value,request_id:crypto.randomUUID()}));
$("cv-preview").addEventListener("toggle",()=>{cvKey=null;if(state)showLive(state.latest);});
$("source-video").addEventListener("loadedmetadata",()=>{videoStream=null;if(state)showLive(state.latest);});
async function poll(){try{apply(await request("/api/state"));}catch(e){label("error","Live evidence unavailable: "+e.message);visible("error",true);$("live-image").parentElement.classList.add("stale");label("live-badge","Disconnected");$("source-video").pause();}finally{setTimeout(poll,200);}}
async function boot(){try{const session=await request("/api/session");csrf=session.csrf_token;try{$("actor").value=localStorage.getItem("first-vial-station-actor")||"";}catch(_){}apply(session.state);poll();}catch(e){label("error",e.message);visible("error",true);}}
boot();
