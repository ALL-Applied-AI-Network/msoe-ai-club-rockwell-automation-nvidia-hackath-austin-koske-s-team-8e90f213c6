'use strict';
const packet=JSON.parse(document.getElementById('packet').textContent),$=id=>document.getElementById(id);
$('queue').querySelector('[value="priority"]').textContent='Priority: '+packet.tasks.filter(t=>t.priority).length+' targets';
$('queue').querySelector('[value="all"]').textContent='All: '+packet.tasks.length+' targets';
const demonstration=new URLSearchParams(location.search).get('demonstration')==='1';
const storageKey='dataset1-slot:'+packet.packet_id+':'+demonstration,taskById=new Map(packet.tasks.map(t=>[t.task_id,t]));
const sceneById=new Map(packet.scenes.map(s=>[s.sample_id,s]));
let history=[],latest={},current=null,image=null,dirty=false,loadToken=0,ready=false;
function say(message){$('message').textContent=message;}
function validate(events){const state={};events.forEach((e,i)=>{const t=taskById.get(e.task_id);if(!t||e.seq!==i+1||e.source_sha256!==t.source_sha256||e.slot!==t.slot||JSON.stringify(e.box_xyxy)!==JSON.stringify(t.box_xyxy))throw Error('History/source/target mismatch');if(!packet.labels.includes(e.label)||!['confirmed','incorrect','uncertain'].includes(e.mapping)||!['visible','occluded','uncertain'].includes(e.visibility)||!e.reviewer.trim()||!e.timestamp)throw Error('Invalid attributed decision');if(e.visibility!=='visible'&&e.label!=='uncertain')throw Error('Occluded/uncertain visibility requires uncertain class');state[e.task_id]=e;});return state;}
function eligibleTasks(){return packet.tasks.filter(t=>$('queue').value==='all'||t.priority);}
function progress(){const tasks=eligibleTasks();$('progress').textContent=tasks.filter(t=>latest[t.task_id]).length+'/'+tasks.length+' saved'+(demonstration?' · DEMONSTRATION':'');}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify({history,reviewer:$('reviewer').value}));}catch(e){say('Browser storage unavailable; download your JSON now.');}}
function exportReview(){return {schema:'dataset1-slot-human-review/v1',packet_id:packet.packet_id,training_eligible:false,demonstration,eligible_for_team_review:!demonstration,team_approved:false,history};}
async function load(task){const token=++loadToken;ready=false;current=task;image=null;dirty=false;$('task').value=task.task_id;
const source=sceneById.get(task.sample_id),saved=latest[task.task_id];$('identity').textContent='Run '+task.run_id+' / '+task.slot+' / '+source.filename;
$('box').textContent='Source crop: ['+task.box_xyxy.join(', ')+'] · 640 × 480';
for(const field of ['mapping','visibility','label','notes'])$(field).value=saved?saved[field]:'';
$('draft').textContent='Machine draft hidden.';$('show').disabled=!saved;$('save').disabled=true;
const im=new Image();await new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=()=>reject(Error('Source image failed to load'));im.src=source.image_data;});if(token!==loadToken)return;
if(im.naturalWidth!==640||im.naturalHeight!==480)throw Error('Source dimensions differ');image=im;ready=true;
const c=$('scene').getContext('2d');c.clearRect(0,0,640,480);c.drawImage(im,0,0);const [l,t,r,b]=task.box_xyxy;c.strokeStyle='#00ffff';c.lineWidth=2;c.strokeRect(l,t,r-l,b-t);c.fillStyle='#142b40';c.fillRect(l,t-20,43,18);c.fillStyle='white';c.font='13px sans-serif';c.fillText(task.slot,l+3,t-6);
const zoom=$('target').getContext('2d');zoom.imageSmoothingEnabled=false;zoom.clearRect(0,0,288,288);zoom.drawImage(im,l,t,r-l,b-t,0,0,288,288);$('save').disabled=false;progress();}
function navigate(task){if(dirty&&!confirm('Discard unsaved changes?'))return false;load(task).catch(e=>say(e.message));return true;}
function refreshQueue(){const tasks=eligibleTasks();$('task').replaceChildren(...tasks.map(t=>new Option((latest[t.task_id]?'✓ ':'')+t.task_id,t.task_id)));return tasks;}
$('save').onclick=()=>{try{if(!ready)throw Error('Wait for the source image');const event={seq:history.length+1,task_id:current.task_id,source_sha256:current.source_sha256,slot:current.slot,box_xyxy:current.box_xyxy,reviewer:$('reviewer').value.trim(),timestamp:new Date().toISOString(),mapping:$('mapping').value,visibility:$('visibility').value,label:$('label').value,notes:$('notes').value};const proposed=[...history,event];latest=validate(proposed);history=proposed;dirty=false;persist();$('show').disabled=false;refreshQueue();$('task').value=current.task_id;progress();say('Human decision appended. Download your JSON regularly.');}catch(e){say(e.message);}};
$('show').onclick=()=>{if(!latest[current.task_id])return;const draft=current.draft;$('draft').textContent='UNREVIEWED MACHINE DRAFT: '+draft.proposed_class+'\nScores (not calibrated accuracy): '+JSON.stringify(draft.class_scores)+'\nMapping and occlusion remain human review questions.';};
$('next').onclick=()=>{const tasks=eligibleTasks(),i=tasks.findIndex(t=>t.task_id===current.task_id);if(i+1<tasks.length)navigate(tasks[i+1]);};
$('prev').onclick=()=>{const tasks=eligibleTasks(),i=tasks.findIndex(t=>t.task_id===current.task_id);if(i>0)navigate(tasks[i-1]);};
$('task').onchange=()=>{if(!navigate(taskById.get($('task').value)))$('task').value=current.task_id;};
$('queue').onchange=()=>{const tasks=refreshQueue();navigate(tasks[0]);};
for(const field of ['mapping','visibility','label','notes'])$(field).addEventListener('input',()=>{dirty=true;});
$('visibility').onchange=()=>{if($('visibility').value!=='visible')$('label').value='uncertain';dirty=true;};
$('reviewer').onchange=persist;
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(exportReview(),null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='fixture-review-'+($('reviewer').value.trim().replace(/[^a-z0-9_-]/gi,'_')||'review')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function importReview(value){if(value.schema!=='dataset1-slot-human-review/v1'||value.packet_id!==packet.packet_id||value.training_eligible!==false||value.demonstration!==demonstration||value.eligible_for_team_review!==!demonstration||!Array.isArray(value.history))throw Error('Packet or demonstration mismatch');const state=validate(value.history);for(let i=0;i<Math.min(history.length,value.history.length);i++)if(JSON.stringify(history[i])!==JSON.stringify(value.history[i]))throw Error('Divergent histories: preserve both files and resolve separately');if(value.history.length<history.length)throw Error('Older history cannot overwrite current decisions');history=value.history;latest=state;persist();refreshQueue();return load(current);}
$('import').onchange=async e=>{try{if(dirty)throw Error('Save or navigate away from unsaved changes before importing');await importReview(JSON.parse(await e.target.files[0].text()));say('Matching history imported.');}catch(error){say(error.message);}e.target.value='';};
try{const stored=JSON.parse(localStorage.getItem(storageKey)||'null');if(stored){latest=validate(stored.history);history=stored.history;$('reviewer').value=stored.reviewer||'';}}catch(e){say('Stored history was not restored: '+e.message);}
const tasks=refreshQueue();load(tasks[0]).catch(e=>say(e.message));if(demonstration)say('DEMONSTRATION: exports are ineligible for real scoring.');
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
window.slotReview={exportReview,importReview,packet};
