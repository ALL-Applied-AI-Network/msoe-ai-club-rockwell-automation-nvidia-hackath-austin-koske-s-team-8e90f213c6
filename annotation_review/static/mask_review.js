'use strict';
const packet=JSON.parse(document.getElementById('packet').textContent);
const demonstration=new URLSearchParams(location.search).get('demonstration')==='1';
const $=id=>document.getElementById(id), tasks=packet.tasks, W=packet.width,H=packet.height;
const canvas=$('image'),ctx=canvas.getContext('2d'),buffer=document.createElement('canvas');buffer.width=W;buffer.height=H;
const bctx=buffer.getContext('2d',{willReadFrequently:true});
let index=0,mask=new Uint8Array(W*H),polygon=[],undos=[],overlay=false,stroke=false,currentImage=null,dirty=false;
let history=[],frameStates={},onsets={},loadedDraft=null,loadToken=0;
const key='visible-pool-review:'+packet.packet_id+(demonstration?':demonstration':'');
const identity=t=>({sample_id:t.sample_id,image_sha256:t.image_sha256,source_rgb_sha256:t.source_rgb_sha256,source_video_sha256:t.source_video_sha256,frame_index:t.frame_index,pts:t.pts,time_base:t.time_base});
function say(s){$('message').textContent=s;}
function encode(m){let runs=[],value=0,count=0;for(const p of m){if(p===value)count++;else{runs.push(count);count=1;value=p;}}runs.push(count);return {width:W,height:H,counts:runs};}
function decode(r){if(!r||r.width!==W||r.height!==H||!Array.isArray(r.counts)||r.counts.length>W*H+1)throw Error('Invalid mask dimensions');let m=new Uint8Array(W*H),pos=0,v=0;for(const n of r.counts){if(!Number.isSafeInteger(n)||n<0||pos+n>m.length)throw Error('Invalid mask run');if(v)m.fill(1,pos,pos+n);pos+=n;v=1-v;}if(pos!==m.length)throw Error('Incomplete mask');return m;}
function validateEvent(e,states){
 if(!e||!Number.isSafeInteger(e.seq)||e.seq<1||typeof e.reviewer!=='string'||!e.reviewer.trim()||!Number.isFinite(Date.parse(e.time)))throw Error('Missing event identity');
 if(e.kind==='frame'){
  const t=tasks.find(t=>t.sample_id===e.source?.sample_id);if(!t||JSON.stringify(e.source)!==JSON.stringify(identity(t)))throw Error('Source identity mismatch');
  if(!['empty','pool','uncertain'].includes(e.status)||typeof e.notes!=='string'||e.training_eligible!==false)throw Error('Invalid review decision');
  const m=decode(e.mask),area=m.reduce((a,b)=>a+b,0);if(e.status==='empty'&&area||e.status==='pool'&&!area)throw Error('Status and mask disagree');
  if(e.status==='uncertain'&&!e.notes.trim())throw Error('Explain uncertain decisions');
  if(e.draft_model_sha256!==null&&!t.drafts.some(d=>d.model_sha256===e.draft_model_sha256))throw Error('Draft model identity mismatch');
 }else if(e.kind==='onset'){
  const a=tasks.find(t=>t.sample_id===e.last_empty),b=tasks.find(t=>t.sample_id===e.first_pool);
  if(!a||!b||a.case_id!==e.case_id||b.case_id!==e.case_id||a.frame_index>=b.frame_index||states[a.sample_id]?.status!=='empty'||states[b.sample_id]?.status!=='pool')throw Error('Onset requires reviewed, ordered endpoints in the same clip');
 }else throw Error('Unknown event kind');
}
function replay(events){let states={},intervals={};events.forEach((e,i)=>{if(e.seq!==i+1)throw Error('History sequence differs');validateEvent(e,states);if(e.kind==='frame')states[e.source.sample_id]=e;else intervals[e.case_id]=e;});return {states,intervals};}
function persist(){try{localStorage.setItem(key,JSON.stringify({packet_id:packet.packet_id,history}));localStorage.setItem(key+':reviewer',$('reviewer').value);}catch(e){say('Browser storage unavailable. Download your JSON before closing.');}}
function addEvent(e){const full={...e,seq:history.length+1,reviewer:$('reviewer').value.trim(),time:new Date().toISOString()};validateEvent(full,frameStates);history.push(full);const r=replay(history);frameStates=r.states;onsets=r.intervals;persist();update();}
function draftImage(url){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(Error('Embedded image failed to load'));i.src=url;});}
function bounds(){return $('zoom').checked?[240,230,400,440]:[0,0,W,H];}
function draw(){if(!currentImage)return;const [x,y,r,b]=bounds();ctx.clearRect(0,0,W,H);ctx.drawImage(currentImage,x,y,r-x,b-y,0,0,W,H);if(overlay){const im=bctx.createImageData(W,H);for(let i=0;i<mask.length;i++)if(mask[i]){im.data[i*4]=60;im.data[i*4+1]=255;im.data[i*4+2]=150;im.data[i*4+3]=105;}bctx.putImageData(im,0,0);ctx.drawImage(buffer,x,y,r-x,b-y,0,0,W,H);}if(polygon.length){ctx.strokeStyle='#ffe475';ctx.lineWidth=2;ctx.beginPath();polygon.forEach(([px,py],i)=>{const a=(px-x)/(r-x)*W,c=(py-y)/(b-y)*H;i?ctx.lineTo(a,c):ctx.moveTo(a,c);});ctx.stroke();for(const [px,py] of polygon){ctx.fillStyle='#ffe475';ctx.fillRect((px-x)/(r-x)*W-3,(py-y)/(b-y)*H-3,6,6);}}}
function checkpoint(){undos.push(mask.slice());if(undos.length>15)undos.shift();}
function update(){
 const t=tasks[index],s=frameStates[t.sample_id];$('progress').textContent=Object.keys(frameStates).length+' / '+tasks.length+' reviewed';$('saved').textContent=s?'Last saved: '+s.status+' by '+s.reviewer:'No human decision saved.';
 const empty=tasks.filter(v=>v.case_id===t.case_id&&frameStates[v.sample_id]?.status==='empty'),pool=tasks.filter(v=>v.case_id===t.case_id&&frameStates[v.sample_id]?.status==='pool');
 for(const [id,list]of [['last-empty',empty],['first-pool',pool]]){const old=$(id).value;$(id).replaceChildren(new Option('Choose endpoint',''),...list.map(v=>new Option('Frame '+v.frame_index,v.sample_id)));$(id).value=old;}
 const o=onsets[t.case_id];$('onset').textContent=o?(onsetValid(o)?'Saved interval: ('+tasks.find(t=>t.sample_id===o.last_empty).frame_index+', '+tasks.find(t=>t.sample_id===o.first_pool).frame_index+']':'Saved interval needs review: endpoint decision changed.'):'No interval saved.';
 $('prev').disabled=index===0;$('next').disabled=index===tasks.length-1;
}
function onsetValid(o){return frameStates[o.last_empty]?.status==='empty'&&frameStates[o.first_pool]?.status==='pool';}
async function load(i){
 const token=++loadToken;index=i;const t=tasks[i],s=frameStates[t.sample_id];polygon=[];undos=[];loadedDraft=s?.draft_model_sha256??null;mask=s?decode(s.mask):new Uint8Array(W*H);overlay=!!s;dirty=false;
 $('status').value=s?.status??'unreviewed';$('notes').value=s?.notes??'';$('case').value=t.case_id;
 $('frame').replaceChildren(...tasks.filter(v=>v.case_id===t.case_id).map(v=>new Option('Frame '+v.frame_index,v.sample_id)));$('frame').value=t.sample_id;
 $('identity').textContent=t.case_id+' · native frame '+t.frame_index+' · PTS '+t.pts+' × '+t.time_base+' · image '+t.image_sha256.slice(0,12);
 $('draft-info').textContent='Draft hidden. Show it only after inspecting the raw image.';
 const clip='clips/'+t.case_id+'.mp4';if(!$('clip').getAttribute('src')?.endsWith(clip)){$('clip').src=clip;}
 currentImage=null;update();const im=await draftImage(t.image_data);if(token===loadToken){currentImage=im;draw();}
}
function navigate(i){if(dirty&&!confirm('Leave this frame without saving your changes?'))return;load(i).catch(e=>say(e.message));}
function point(e){const box=canvas.getBoundingClientRect(),[x,y,r,b]=bounds();return [Math.max(0,Math.min(W-1,x+(e.clientX-box.left)/box.width*(r-x))),Math.max(0,Math.min(H-1,y+(e.clientY-box.top)/box.height*(b-y)))];}
function brush(p){const radius=Math.max(1,Math.min(30,Number($('brush').value)||3)),[x,y]=p,v=$('tool').value==='erase'?0:1;for(let b=Math.max(0,Math.floor(y-radius));b<Math.min(H,Math.ceil(y+radius+1));b++)for(let a=Math.max(0,Math.floor(x-radius));a<Math.min(W,Math.ceil(x+radius+1));a++)if((a-x)**2+(b-y)**2<=radius**2)mask[b*W+a]=v;overlay=true;dirty=true;draw();}
canvas.addEventListener('pointerdown',e=>{e.preventDefault();if($('tool').value==='polygon'){polygon.push(point(e));dirty=true;draw();}else{checkpoint();stroke=true;canvas.setPointerCapture(e.pointerId);brush(point(e));}});
canvas.addEventListener('pointermove',e=>{if(stroke)brush(point(e));});canvas.addEventListener('pointerup',()=>stroke=false);canvas.addEventListener('pointercancel',()=>stroke=false);
$('close').onclick=()=>{if(polygon.length<3){say('Add at least three polygon vertices.');return;}checkpoint();bctx.clearRect(0,0,W,H);bctx.fillStyle='white';bctx.beginPath();polygon.forEach(([x,y],i)=>i?bctx.lineTo(x,y):bctx.moveTo(x,y));bctx.closePath();bctx.fill();const d=bctx.getImageData(0,0,W,H).data;for(let i=0;i<mask.length;i++)if(d[i*4+3]>=128)mask[i]=1;polygon=[];overlay=true;dirty=true;draw();};
$('undo').onclick=()=>{if(polygon.length)polygon.pop();else if(undos.length)mask=undos.pop();dirty=true;draw();};
$('clear').onclick=()=>{checkpoint();mask.fill(0);polygon=[];overlay=true;dirty=true;draw();};$('raw').onclick=()=>{overlay=!overlay;draw();};$('zoom').onchange=draw;
$('draft').onclick=async()=>{try{const token=loadToken,d=tasks[index].drafts.find(d=>d.mode===packet.primary_mode),im=await draftImage(d.mask_data);if(token!==loadToken)return;checkpoint();bctx.clearRect(0,0,W,H);bctx.drawImage(im,0,0);const pixels=bctx.getImageData(0,0,W,H).data;mask=Uint8Array.from({length:W*H},(_,i)=>pixels[i*4]>=128?1:0);loadedDraft=d.model_sha256;overlay=true;dirty=true;draw();$('draft-info').textContent='Unreviewed '+d.mode+' draft · presence score '+d.presence_probability.toFixed(3)+' (not accuracy). Correct it before saving.';}catch(e){say(e.message);}};
$('save').onclick=()=>{try{if(polygon.length)throw Error('Close or undo the unfinished polygon first.');const status=$('status').value;if(status==='empty')mask.fill(0);addEvent({kind:'frame',source:identity(tasks[index]),status,notes:$('notes').value,mask:encode(mask),draft_model_sha256:loadedDraft,training_eligible:false});dirty=false;overlay=true;draw();say('Human decision saved. Download your JSON to keep a portable copy.');}catch(e){say(e.message);}};
$('save-onset').onclick=()=>{try{addEvent({kind:'onset',case_id:tasks[index].case_id,last_empty:$('last-empty').value,first_pool:$('first-pool').value});say('Reviewed interval saved. It does not establish an exact native onset unless endpoints are adjacent.');}catch(e){say(e.message);}};
function exportReview(){return {schema:'visible-pool-human-review/v1',packet_id:packet.packet_id,source_video_sha256:packet.source_video_sha256,source_run:4,training_eligible:false,demonstration,eligible_for_team_review:!demonstration,team_approved:false,history,latest_decisions:frameStates,onset_intervals:Object.fromEntries(Object.entries(onsets).map(([k,o])=>[k,{...o,valid:onsetValid(o),convention:'(last definitely absent, first definitely present]',exact_native_first_present:tasks.find(t=>t.sample_id===o.first_pool).frame_index===tasks.find(t=>t.sample_id===o.last_empty).frame_index+1}]))};}
function importReview(v){if(v.schema!=='visible-pool-human-review/v1'||v.packet_id!==packet.packet_id||v.source_video_sha256!==packet.source_video_sha256||v.training_eligible!==false||v.demonstration!==demonstration||!Array.isArray(v.history))throw Error('Packet/source or demonstration mismatch');const r=replay(v.history);for(let i=0;i<Math.min(history.length,v.history.length);i++)if(JSON.stringify(history[i])!==JSON.stringify(v.history[i]))throw Error('Divergent reviewer histories: keep both exports and resolve separately.');if(v.history.length<history.length)throw Error('Import is older than current history.');history=v.history;frameStates=r.states;onsets=r.intervals;persist();return load(index);}
$('download').onclick=()=>{const blob=new Blob([JSON.stringify(exportReview(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='visible-pool-'+($('reviewer').value.trim().replace(/[^a-z0-9_-]/gi,'_')||'review')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('import').onchange=async e=>{try{await importReview(JSON.parse(await e.target.files[0].text()));say('History imported with matching packet/source identity.');}catch(err){say(err.message);}e.target.value='';};
$('prev').onclick=()=>navigate(index-1);$('next').onclick=()=>navigate(index+1);$('case').onchange=()=>navigate(tasks.findIndex(t=>t.case_id===$('case').value));$('frame').onchange=()=>navigate(tasks.findIndex(t=>t.sample_id===$('frame').value));$('notes').oninput=()=>dirty=true;$('status').onchange=()=>dirty=true;$('reviewer').onchange=persist;
$('case').replaceChildren(...packet.cases.map(c=>new Option(c.case_id,c.case_id)));
try{const saved=localStorage.getItem(key);if(saved){const v=JSON.parse(saved);if(v.packet_id!==packet.packet_id)throw Error('Stored packet differs');const r=replay(v.history);history=v.history;frameStates=r.states;onsets=r.intervals;}$('reviewer').value=localStorage.getItem(key+':reviewer')||'';}catch(e){say('Could not restore browser history: '+e.message);}
load(0).catch(e=>say(e.message));
if(demonstration)say('DEMONSTRATION: saved decisions are ineligible as team labels.');
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
