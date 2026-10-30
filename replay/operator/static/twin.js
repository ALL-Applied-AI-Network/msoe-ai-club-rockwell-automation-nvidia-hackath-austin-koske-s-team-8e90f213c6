"use strict";
const $=id=>document.getElementById(id);
let state=null,csrf=null,pending=false,dialogCommand=null,shownKey=null,imageGeneration=0,chartEpisode=null,localError=null;
const points=[];
const words=value=>String(value||"").replaceAll("_"," ");
const text=(id,value)=>{$(id).textContent=value;};
const labels={loading:["Starting local simulation","Loading the actual frozen models and native Isaac renderer."],ready_reference:["Capture the clean reference","Render a nominal fill and measure it using the actual vision models. This is a development reference."],reference_running:["Measuring the reference fill","Isaac is rendering new camera frames. Each frame waits for YOLO, the liquid detector, SAM2 and QA."],ready_test:["Inspect the test vial","The next declared scenario starts with a different residue in the same assumed 2 mL shared path. The intended recipe stays the same."],test_running:["Inspect the current fill","Watch actual model measurements against the prior nominal reference. Unavailable evidence breaks consecutive deviations."],awaiting_response:["Operator response needed","Inspect the evidence, acknowledge the review, then request the hypothetical simulator-only prime-to-drain."],flushing:["Simulated drain / prime in progress","The compartment model sends 8 mL to the drain and retains the resulting path inventory. This is not a CSI-approved procedure."],ready_recovery:["Inspect what happens next","Run a new vial using the exact retained path state after the simulated intervention."],recovery_running:["Measuring the recovery fill","The models are observing newly rendered recovery frames. The previous review remains open."],recovery_complete:["Document the follow-up","Review the recovery evidence and record the outcome. No comparison or no warning alone cannot establish acceptance."],complete:["Response documented","The simulated follow-up is closed. Review the measured episodes below or start another walkthrough."],stopping:["Stopping the simulator","Waiting for the native renderer to shut down."],stopped:["Simulation stopped","Evidence and operator history remain available for inspection and export."],failed:["Simulation unavailable","Inspect the error and retained logs. No successful result is inferred from missing evidence."]};
async function request(path,options={}){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw new Error(data.error||"Request failed");return data;}
function drawChart(row){
  if(!row)return;
  if(chartEpisode!==row.source.episode_id){chartEpisode=row.source.episode_id;points.length=0;}
  if(!points.some(p=>p.frame===row.source.frame_index))points.push({frame:row.source.frame_index,time:row.source.sim_time_s,value:row.diagnostic.delta_e2000});
  const c=$("chart"),ctx=c.getContext("2d"),w=c.width,h=c.height,max=Math.max(10,...points.map(p=>p.value||0))*1.15;
  ctx.clearRect(0,0,w,h);const y=value=>h-23-value/max*(h-35),x=time=>25+time/7.2*(w-45);
  ctx.strokeStyle="#dce6dd";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(25,h-23);ctx.lineTo(w-10,h-23);ctx.stroke();
  ctx.setLineDash([5,5]);ctx.strokeStyle="#c48d46";ctx.beginPath();ctx.moveTo(25,y(5));ctx.lineTo(w-10,y(5));ctx.stroke();ctx.setLineDash([]);
  ctx.font="11px Segoe UI";ctx.fillStyle="#8a9a8d";ctx.fillText("5",5,y(5)+4);ctx.fillText("0 s",25,h-5);ctx.fillText("7.2 s",w-45,h-5);
  let connected=false;ctx.strokeStyle="#147c72";ctx.lineWidth=2.5;ctx.beginPath();
  for(const p of points){if(p.value===null){connected=false;continue;}if(connected)ctx.lineTo(x(p.time),y(p.value));else ctx.moveTo(x(p.time),y(p.value));connected=true;}ctx.stroke();
}
function liveImage(row){
  if(!row){imageGeneration++;$("live-image").hidden=true;$("camera-empty").hidden=false;shownKey=null;
    $("camera-empty").querySelector("strong").textContent=state.connected?"Ready for a walkthrough":"Preparing the station";
    $("camera-empty").querySelector("p").textContent=state.connected?"Enter your operator name and capture a clean reference to begin.":"Loading frozen models and the native Isaac renderer.";
    return;}
  const key=row.image_key+"/"+$("view").value;if(key===shownKey)return;shownKey=key;
  const generation=++imageGeneration,img=new Image();
  img.onload=()=>{if(generation!==imageGeneration)return;$("live-image").src=img.src;$("live-image").hidden=false;$("camera-empty").hidden=true;$("live-image").dataset.frame=row.image_key;text("image-clock",`${row.source.episode_id} · sample ${row.source.frame_index} · simulation ${row.source.sim_time_s.toFixed(1)} s`);};
  img.onerror=()=>{if(generation===imageGeneration){$("live-image").hidden=true;text("image-clock","Source image unavailable; wait for the next matched sample");}};
  img.src=`/api/image?key=${encodeURIComponent(row.image_key)}&kind=${encodeURIComponent($("view").value)}`;
}
function render(){
  const row=state.latest,warning=state.review==="requested"||state.review==="acknowledged";
  const delayed=Boolean(row&&state.step.endsWith("_running")&&state.frame_age_s>15);
  text("connection-status",state.connected?"Live native Isaac Sim / local GPU":state.step==="loading"?"Starting native Isaac Sim…":"Simulator disconnected");
  text("step-badge",words(state.step));text("sample-quality",delayed?"Evidence delayed":row?(row.measurement_valid?"Sample checks pass":"Unavailable"):"Waiting for camera");
  text("sample-clock",row?`Simulation ${row.source.sim_time_s.toFixed(1)} s · ${row.mask_pixels.toLocaleString()} pixels`:"New frames will appear here");
  const score=delayed?null:row?.diagnostic.delta_e2000;
  text("score",score===null||score===undefined?"—":`${score.toFixed(2)} dE`);
  text("comparison-note",row?.role==="reference"?"Building rendered development reference":score===null||score===undefined?"No eligible comparison at this sample":"Development band 5.0 · not recipe approval");
  text("review-status",state.review==="none"?"No review requested":words(state.review));
  text("episode-label",row?words(row.role)+" fill":"Waiting for operator");
  text("latency",row?`Inference ${row.inference_ms.toFixed(0)} ms · 5 Hz simulation clock`:"Local rendering + GPU inference");
  text("frame-status",state.step.endsWith("_running")?"Rendering new frames":"Waiting for operator");
  text("qa-status",row?(row.measurement_valid?"Sample checks pass · geometry unreviewed":"Sample unavailable"):"Model geometry remains unreviewed");
  text("qa-reason",row?row.quality_flags.join(", ")||"Actual model sampling pixels; no private labels used":"Private composition and geometric labels never enter inference.");
  $("qa-dot").className="signal"+(row?.measurement_valid&&!delayed?" valid":"");
  let [title,reason]=labels[state.step]||[words(state.step),""];
  if(state.review==="requested"&&(state.step==="test_running"||state.step==="awaiting_response")){
    const actual=state.events.some(e=>e.kind==="review_requested"&&e.payload.origin==="actual_model_color_diagnostic");
    title=actual?"Appearance deviation: review requested":"Planned human review requested";
    reason=actual?"Three consecutive qualified model measurements exceeded the development band. Acknowledge and inspect before requesting a simulated response.":"This scenario requires a human inspection. A model warning was not observed; do not count this as a detection.";
  }
  text("title",title);text("reason",reason);text("kicker",warning?"OPERATOR FOLLOW-UP / SIMULATION":"HUMAN-DIRECTED WALKTHROUGH");
  $("assessment").className="assessment-status "+(warning?"amber":"neutral");
  const buttons={reference:state.step==="ready_reference",test:state.step==="ready_test",acknowledge:state.review==="requested",flush:state.step==="awaiting_response"&&state.review==="acknowledged",recovery:state.step==="ready_recovery",close:state.step==="recovery_complete"&&state.review==="acknowledged",reset:state.step==="complete"};
  for(const [id,visible] of Object.entries(buttons)){$(id).hidden=!visible;$(id).disabled=pending||!state.connected;}
  const stages=["reference","test","response","recovery"],position=["ready_reference","reference_running"].includes(state.step)?0:["ready_test","test_running"].includes(state.step)?1:["awaiting_response","flushing"].includes(state.step)?2:3;
  stages.forEach((v,i)=>$("stage-"+v).className=i<position||state.step==="complete"?"done":i===position?"current":"");
  const table=$("episodes");table.replaceChildren();
  for(const e of state.episodes){const tr=document.createElement("tr");for(const value of [e.episode_id,`${e.qa_pass}/${e.frames}`,e.comparisons,e.max_delta_e2000===null?"—":e.max_delta_e2000.toFixed(2),e.role==="reference"?"Development reference":e.warning?"Review requested":e.comparisons?"No warning observed":"Unavailable"]){const td=document.createElement("td");td.textContent=value;tr.append(td);}table.append(tr);}
  const activity=$("activity");activity.replaceChildren();
  for(const e of [...state.events].reverse().slice(0,7)){const li=document.createElement("li"),strong=document.createElement("strong"),small=document.createElement("small");strong.textContent=words(e.kind);small.textContent=`${e.actor||"System"}${e.source?` · ${e.source.episode_id} / ${e.source.sim_time_s.toFixed(1)} s`:""}`;li.append(strong,small);activity.append(li);}
  text("provenance",JSON.stringify({session:state.session_id,mode:state.mode,source:row?.source,model_inference:state.model_inference,operational_recipe_assessment:"unavailable",reference:"rendered development only",hardware_commands:0,journal:state.journal_path},null,2));
  const error=state.error||localError;$("error").hidden=!error;if(error)text("error",error);
  drawChart(row);liveImage(row);
}
async function send(command,extra={}){
  if(pending)return;const actor=$("actor").value.trim();if(!actor)throw new Error("Enter your operator name first.");
  localError=null;pending=true;render();
  try{await request("/api/command",{method:"POST",headers:{"Content-Type":"application/json","X-Operator-CSRF":csrf},body:JSON.stringify({command,actor,expected_step:state.step,review_id:state.review_id,request_id:crypto.randomUUID(),...extra})});state=await request("/api/state");render();}
  finally{pending=false;render();}
}
async function clickCommand(command){
  if(command==="flush"||command==="close"){
    dialogCommand=command;$("action-form").reset();$("action-error").hidden=true;
    text("dialog-title",command==="flush"?"Request a simulated drain / prime":"Document the simulated follow-up");
    text("dialog-description",command==="flush"?"Send 8 mL of ingredient 0 through the retained simulated path to the drain. This hypothesis is not a physical operating instruction. Models will measure a new recovery fill afterward.":"Describe the observed recovery evidence and remaining uncertainty. Closing this review does not authorize product release.");
    $("action-dialog").showModal();return;
  }
  try{await send(command);}catch(e){localError=e.message;render();}
}
for(const b of document.querySelectorAll("[data-command]"))b.addEventListener("click",()=>clickCommand(b.dataset.command));
$("cancel-action").addEventListener("click",()=>$("action-dialog").close());
$("action-form").addEventListener("submit",async event=>{event.preventDefault();$("save-action").disabled=true;try{await send(dialogCommand,{note:$("action-note").value,confirmed:$("action-confirm").checked});$("action-dialog").close();}catch(e){text("action-error",e.message);$("action-error").hidden=false;}finally{$("save-action").disabled=false;}});
$("view").addEventListener("change",()=>{shownKey=null;liveImage(state.latest);});
async function poll(){try{state=await request("/api/state");render();}catch(e){text("error","Live evidence unavailable: "+e.message);$("error").hidden=false;$("live-image").hidden=true;}finally{setTimeout(poll,400);}}
async function boot(){try{const session=await request("/api/session");csrf=session.csrf_token;state=session.state;render();poll();}catch(e){text("error",e.message);$("error").hidden=false;}}
boot();
