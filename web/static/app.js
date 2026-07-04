// ===== Tiện ích dùng chung cho toàn dashboard =====

// Đánh dấu menu đang active
(function markNav(){
  const path = location.pathname;
  const map = {'/':'dashboard','/events':'events','/trips':'trips','/waiting':'waiting',
               '/reconcile':'reconcile','/alerts':'alerts','/reports':'reports','/config':'config'};
  const key = map[path];
  document.querySelectorAll('.nav a').forEach(a=>{
    if(a.dataset.nav === key) a.classList.add('active');
  });
})();

// Định dạng
function esc(s){ return (s??'').toString().replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function badge(v, cls){ return `<span class="badge ${cls||v}">${esc(v)}</span>`; }

// Toast
function toast(title, msg, sev){
  let box = document.getElementById('toasts');
  if(!box){ box = document.createElement('div'); box.id='toasts'; document.body.appendChild(box); }
  const el = document.createElement('div');
  el.className = 'toast ' + (sev||'INFO');
  el.innerHTML = `<div class="t-title">${esc(title)}</div><div class="t-msg">${esc(msg)}</div>`;
  box.appendChild(el);
  setTimeout(()=>{ el.style.opacity='0'; setTimeout(()=>el.remove(),300); }, 5000);
}

// WebSocket realtime với auto-reconnect
class RealtimeBus {
  constructor(){ this.handlers = []; this.connect(); }
  on(fn){ this.handlers.push(fn); }
  connect(){
    const proto = location.protocol==='https:'?'wss':'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws.onopen = ()=> this.setStatus(true);
    this.ws.onclose = ()=>{ this.setStatus(false); setTimeout(()=>this.connect(), 2000); };
    this.ws.onerror = ()=> this.setStatus(false);
    this.ws.onmessage = (e)=>{
      let d; try{ d = JSON.parse(e.data); }catch(_){ return; }
      this.handlers.forEach(h=>h(d));
    };
  }
  setStatus(on){
    const dot = document.getElementById('wsDot');
    const txt = document.getElementById('wsText');
    if(dot) dot.className = 'dot ' + (on?'on':'off');
    if(txt) txt.textContent = on ? 'Đã kết nối realtime' : 'Mất kết nối, đang thử lại…';
  }
}

// Modal xem ảnh/clip
function showMedia(url){
  let bg = document.getElementById('mediaModal');
  if(!bg){
    bg = document.createElement('div'); bg.id='mediaModal'; bg.className='modal-bg';
    bg.innerHTML = `<div class="modal"><div class="modal-head"><b>Bằng chứng</b>
      <button class="sm" onclick="document.getElementById('mediaModal').classList.remove('show')">✕ Đóng</button></div>
      <div id="mediaBody"></div></div>`;
    document.body.appendChild(bg);
    bg.addEventListener('click', e=>{ if(e.target===bg) bg.classList.remove('show'); });
  }
  const body = document.getElementById('mediaBody');
  if(url && url.match(/\.(mp4|avi|mov|mkv)$/i)){
    body.innerHTML = `<video src="${esc(url)}" controls autoplay></video>`;
  } else {
    body.innerHTML = `<img src="${esc(url)}">`;
  }
  bg.classList.add('show');
}

async function api(path, opts){
  const r = await fetch(path, opts);
  return r.json();
}
