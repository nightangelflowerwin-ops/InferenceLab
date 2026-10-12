const $ = id => document.getElementById(id);
let token, state, polling = false, jobsSignature = '', preparedBatch = null;
const names = {pending:'Waiting', leased:'Running', done:'Completed', dead:'Failed'};
function node(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
async function post(path, body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Job-Token':token},body:JSON.stringify(body)});
  const data=await response.json(); if(!response.ok)throw Error(data.error); return data;
}
function render(data) {
  state=data;
  $('pending').textContent=data.summary.pending_jobs; $('running').textContent=data.summary.leased_jobs;
  $('completed').textContent=data.summary.done_jobs; $('failed').textContent=data.summary.dead_jobs;
  $('workers').replaceChildren(...data.workers.map(w=>{const box=node('div',undefined,'worker');box.append(node('strong',w.alive?'● Available process':'○ Replacing process'),node('span',`${w.id} · PID ${w.pid}`));return box;}));
  const signature=JSON.stringify(data.jobs);
  if(signature!==jobsSignature) {$('jobs').replaceChildren(...data.jobs.map(job=>{
    const row=node('tr'), file=node('td',job.name);file.append(node('small',job.id));row.append(file);
    const status=node('td'); status.append(node('span',job.result?.error?'Report error':names[job.state],'badge '+job.state));row.append(status,node('td',String(job.attempts)),node('td',job.result?.words===undefined?'-':String(job.result.words)));
    const report=node('td');if(job.result){const detail=node('details');detail.append(node('summary','View report'),node('pre',JSON.stringify(job.result,null,2)));report.append(detail);}else{report.textContent='Awaiting result';}row.append(report);return row;
  })); jobsSignature=signature;}
  $('empty').hidden=!!data.jobs.length;
  $('crash').disabled=!data.summary.leased_jobs;
  $('export').disabled=!data.jobs.some(j=>j.result);
}
async function refresh(){if(polling)return;polling=true;try{const r=await fetch('/api/state');if(!r.ok)throw Error('Disconnected');render(await r.json());$('connection').textContent='● Live · local';}catch(e){$('connection').textContent='○ Connection lost';}finally{polling=false;}}
$('files').addEventListener('change',()=>{preparedBatch=null;const files=[...$('files').files];const bytes=files.reduce((a,f)=>a+f.size,0);const valid=files.length>0&&files.length<=100&&bytes<=10*1024*1024&&files.every(f=>f.size<=1024*1024&&f.name.toLowerCase().endsWith('.txt'));$('submit').disabled=!valid;$('selection').textContent=valid?`${files.length} file(s) selected · ${(bytes/1024).toFixed(1)} KiB`:'Choose 1–100 .txt files, up to 1 MiB each and 10 MiB total.';});
$('submit').addEventListener('click',async()=>{const button=$('submit');button.disabled=true;let added=0;try{const batch=preparedBatch||[];if(!preparedBatch){for(const file of $('files').files){const buffer=await file.arrayBuffer();const text=new TextDecoder('utf-8',{fatal:true}).decode(buffer);batch.push({name:file.name,text,id:crypto.randomUUID()});}preparedBatch=batch;}for(const job of batch){const result=await post('/api/jobs',job);if(result.inserted)added++;}$('notice').textContent=`Added ${added} text jobs. Workers are processing them.`;preparedBatch=null;$('files').value='';$('selection').textContent='Up to 100 files and 10 MiB per batch.';}catch(e){$('notice').textContent=`Added ${added} jobs before stopping: ${e.message}`;button.disabled=false;}await refresh();});
for(const [id,path,message] of [['demo','/api/demo','Recovery job queued. Crash a busy worker while it runs.'],['crash','/api/crash','Worker stopped. Watch the attempt count increase after its lease expires.']]){$(id).addEventListener('click',async()=>{try{const result=await post(path);$('notice').textContent=message+(result.job?' Job: '+result.job:'');}catch(e){$('notice').textContent=e.message;}await refresh();});}
$('export').addEventListener('click',async()=>{try{const r=await fetch('/api/reports');if(!r.ok)throw Error('Download failed');const reports=await r.json();const url=URL.createObjectURL(new Blob([JSON.stringify(reports,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download='relay-reports.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){$('notice').textContent=e.message;}});
(async()=>{try{const r=await fetch('/api/session');token=(await r.json()).token;await refresh();setInterval(refresh,800);}catch(e){$('connection').textContent='○ Could not connect';}})();
