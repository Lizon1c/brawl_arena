"""Local map editor for brawl-arena: pure-stdlib web UI.

Run:  .venv/Scripts/python editor.py
Then open http://127.0.0.1:8787 in a browser.

Saves user maps as full-map ASCII files to brawl_arena/maps_custom/
('<mode>_<name>.txt' plus a rendered preview PNG). brawl_arena.maps scans
that directory at import time, so saved maps join the random sampling pool
of Game automatically (invalid maps are skipped with a warning).

No external resources: single-page HTML + vanilla JS canvas, served by
http.server.
"""
from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from brawl_arena.core import MODES
from brawl_arena.maps import (CUSTOM_MAPS_DIR, MAPS, _CHAR_TO_TILE, _MODE_SIZE,
                              _check_map, _expand, parse_ascii_map)
from brawl_arena.render import render_map_preview

HOST, PORT = "127.0.0.1", 8787

_TILE_TO_CHAR = {v: k for k, v in _CHAR_TO_TILE.items()}

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>brawl-arena map editor</title>
<style>
body{background:#16181d;color:#dde;font-family:sans-serif;margin:16px}
h1{font-size:18px} .row{display:flex;gap:18px;align-items:flex-start}
#pal{display:flex;flex-direction:column;gap:6px}
.sw{display:flex;align-items:center;gap:8px;padding:4px 8px;border:2px solid #333;
    cursor:pointer;border-radius:4px;background:#20242a;user-select:none}
.sw.sel{border-color:#6cf} .sw.off{opacity:.25;pointer-events:none}
.chip{width:20px;height:20px;border-radius:3px;display:inline-block}
canvas{border:1px solid #444;image-rendering:pixelated;cursor:crosshair}
select,input,button{background:#20242a;color:#dde;border:1px solid #444;
    padding:4px 8px;border-radius:4px}
#status{margin-top:8px;white-space:pre-wrap;color:#8d9}
#status.err{color:#f88}
label{margin-right:6px}
</style></head><body>
<h1>brawl-arena map editor</h1>
<div>
<label>mode <select id="mode"></select></label>
<label>symmetry <select id="sym">
<option value="none">none</option><option value="tb">top-bottom</option>
<option value="lr">left-right</option>
<option value="center">center (180°)</option>
<option value="quad" selected>quad mirror</option></select></label>
<label>name <input id="name" value="mymap"></label>
<button id="save">save</button>
<button id="new">new</button>
<label>load builtin <select id="blt"></select></label>
<button id="load">load</button>
<button id="undo">undo (Ctrl+Z)</button>
<span id="status"></span>
</div>
<div class="row" style="margin-top:12px">
<div id="pal"></div>
<canvas id="cv"></canvas>
</div>
<script>
const SIZES = {gem_grab:[21,15], brawl_ball:[21,15], knockout:[17,13],
               showdown:[25,19]};
const PAL = [
 {ch:'.', label:'empty', c:'#20242a'},
 {ch:'#', label:'wall', c:'#7a7a82'},
 {ch:'c', label:'crate', c:'#96643c'},
 {ch:'b', label:'bush', c:'#24602c'},
 {ch:'f', label:'fence', c:'#583822'},
 {ch:'g', label:'goal', c:'#e8c858', modes:['brawl_ball']},
 {ch:'s', label:'spawn', c:'#f0f0f0', modes:['brawl_ball','knockout','showdown']},
 {ch:'m', label:'gem mine (fixed centre)', c:'#c83cdc', modes:['gem_grab']},
];
const CELL = 26;
let mode = 'gem_grab', cur = '#', sym = 'quad';
let W = 21, H = 15, grid = [];
let builtins = {};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const statusEl = document.getElementById('status');
function status(msg, err){ statusEl.textContent = msg;
  statusEl.className = err ? 'err' : ''; }

function freshGrid(){
  const [w,h] = SIZES[mode]; W = w; H = h;
  grid = [];
  for(let y=0;y<h;y++){ grid.push(new Array(w).fill('.')); }
  for(let x=0;x<w;x++){ grid[0][x]='#'; grid[h-1][x]='#'; }
  for(let y=0;y<h;y++){ grid[y][0]='#'; grid[y][w-1]='#'; }
  if(mode==='brawl_ball'){
    const cy = h>>1;
    for(let y=cy-1;y<=cy+1;y++){ grid[y][0]='g'; grid[y][w-1]='g'; }
  }
  if(mode==='showdown'){
    const pts = [[2,2],[w-3,2],[2,h-3],[w-3,h-3],[w>>1,2],[w>>1,h-3]];
    for(const [x,y] of pts) grid[y][x]='s';
  }
  cv.width = W*CELL; cv.height = H*CELL;
  draw();
}
function mirrors(x,y){
  if(sym==='none') return [[x,y]];
  if(sym==='tb') return [[x,y],[x,H-1-y]];
  if(sym==='lr') return [[x,y],[W-1-x,y]];
  if(sym==='center') return [[x,y],[W-1-x,H-1-y]];
  return [[x,y],[x,H-1-y],[W-1-x,y],[W-1-x,H-1-y]];
}
function paintAt(cx,cy,ch){
  for(const [x,y] of mirrors(cx,cy)) grid[y][x]=ch;
  draw();
}
function drawCell(x,y){
  const ch = grid[y][x], px=x*CELL, py=y*CELL;
  const pal = PAL.find(p=>p.ch===ch);
  ctx.fillStyle = pal ? pal.c : '#20242a';
  ctx.fillRect(px,py,CELL,CELL);
  if(ch==='#'){ ctx.fillStyle='#565660';
    ctx.fillRect(px+4,py+4,4,4); ctx.fillRect(px+CELL-8,py+4,4,4);
    ctx.fillRect(px+4,py+CELL-8,4,4); ctx.fillRect(px+CELL-8,py+CELL-8,4,4); }
  if(ch==='c'){ ctx.fillStyle='#6e4626';
    ctx.fillRect(px,py+CELL/3,CELL,2); ctx.fillRect(px,py+2*CELL/3,CELL,2); }
  if(ch==='b'){ ctx.fillStyle='#3d8a46';
    for(let i=0;i<5;i++){ const rx=(x*7+i*5+y*3)%CELL, ry=(y*11+i*7+x*2)%CELL;
      ctx.fillRect(px+rx,py+ry,3,3); } }
  if(ch==='f'){ ctx.fillStyle='#2c1c12';
    for(let i=1;i<4;i+=2) ctx.fillRect(px+i*CELL/4,py,3,CELL); }
  if(ch==='g'){ ctx.fillStyle='#342e4c';
    ctx.fillRect(px,py+CELL/3,CELL,CELL/3); }
  if(ch==='s'){ ctx.fillStyle='#fff'; ctx.beginPath();
    ctx.arc(px+CELL/2,py+CELL/2,CELL/4,0,7); ctx.fill(); }
  ctx.strokeStyle='#00000033'; ctx.strokeRect(px+.5,py+.5,CELL-1,CELL-1);
}
function draw(){
  for(let y=0;y<H;y++) for(let x=0;x<W;x++) drawCell(x,y);
  if(mode==='gem_grab'){ const px=(W>>1)*CELL, py=(H>>1)*CELL;
    ctx.fillStyle='#c83cdc'; ctx.beginPath();
    ctx.arc(px+CELL/2,py+CELL/2,CELL/3,0,7); ctx.fill(); }
}
function cellFromEvent(e){
  const r = cv.getBoundingClientRect();
  return [Math.floor((e.clientX-r.left)/CELL), Math.floor((e.clientY-r.top)/CELL)];
}
let painting = false;
const undoStack = [];
function pushUndo(){
  undoStack.push(grid.map(r=>r.join('')));
  if(undoStack.length > 200) undoStack.shift();
}
function undo(){
  const rows = undoStack.pop();
  if(!rows){ status('nothing to undo'); return; }
  grid = rows.map(r=>r.split(''));
  draw(); status('undo');
}
cv.addEventListener('mousedown', e=>{
  painting = true;
  pushUndo();
  const [x,y] = cellFromEvent(e);
  paintAt(x, y, e.button===2 ? '.' : cur);
});
cv.addEventListener('mousemove', e=>{
  if(!painting) return;
  const [x,y] = cellFromEvent(e);
  paintAt(x, y, (e.buttons&2) ? '.' : cur);
});
window.addEventListener('mouseup', ()=> painting=false);
document.getElementById('undo').onclick = undo;
window.addEventListener('keydown', e=>{
  if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='z'){ e.preventDefault(); undo(); }
});
cv.addEventListener('contextmenu', e=> e.preventDefault());

function buildPalette(){
  const pal = document.getElementById('pal'); pal.innerHTML='';
  for(const p of PAL){
    const d = document.createElement('div');
    d.className = 'sw' + (p.ch===cur?' sel':'');
    if(p.modes && !p.modes.includes(mode)) d.classList.add('off');
    d.innerHTML = '<span class="chip" style="background:'+p.c+'"></span>'+p.label;
    d.onclick = ()=>{
      if(p.ch==='m'){ status('gem mine is fixed at the map centre'); return; }
      cur = p.ch; buildPalette();
    };
    pal.appendChild(d);
  }
}
document.getElementById('mode').onchange = e=>{ mode=e.target.value;
  pushUndo(); buildPalette(); freshGrid(); };
document.getElementById('sym').onchange = e=>{ sym=e.target.value; };
document.getElementById('new').onclick = ()=>{ pushUndo(); freshGrid(); status('new map'); };
document.getElementById('save').onclick = async ()=>{
  const body = {mode, name:document.getElementById('name').value,
                rows:grid.map(r=>r.join(''))};
  const res = await fetch('/api/save', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await res.json();
  status(data.message, !data.ok);
};
document.getElementById('load').onclick = ()=>{
  const key = document.getElementById('blt').value;
  if(!key) return;
  const [m, ...rest] = key.split('/');
  pushUndo();
  mode = m; document.getElementById('mode').value = m;
  const rows = builtins[m][rest.join('/')];
  const [w,h] = SIZES[mode]; W=w; H=h;
  grid = rows.map(r=>r.split(''));
  cv.width = W*CELL; cv.height = H*CELL;
  buildPalette(); draw();
  status('loaded builtin ' + key);
};
(async function init(){
  const ms = document.getElementById('mode');
  for(const m of Object.keys(SIZES)){
    const o = document.createElement('option'); o.value=m; o.textContent=m;
    ms.appendChild(o);
  }
  const res = await fetch('/api/builtin'); builtins = await res.json();
  const bs = document.getElementById('blt');
  for(const m of Object.keys(builtins))
    for(const n of Object.keys(builtins[m])){
      const o = document.createElement('option');
      o.value = m+'/'+n; o.textContent = m+'/'+n; bs.appendChild(o);
    }
  buildPalette(); freshGrid();
})();
</script></body></html>"""


def _builtin_json() -> dict:
    out = {}
    for mode, maps in MAPS.items():
        out[mode] = {}
        for name, quadrant in maps.items():
            tiles = _expand(quadrant)
            out[mode][name] = ["".join(_TILE_TO_CHAR[int(v)] for v in row)
                               for row in tiles]
    return out


def save_map(mode: str, name: str, rows: list[str]) -> str:
    """Validate and persist a user map; returns the saved file path."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    name = re.sub(r"[^a-z0-9_-]", "", name.lower())[:40]
    if not name:
        raise ValueError("empty/invalid map name")
    text = "\n".join(rows) + "\n"
    tiles, spawns = parse_ascii_map(text)
    _check_map(mode, name, tiles, spawns=spawns or None)
    os.makedirs(CUSTOM_MAPS_DIR, exist_ok=True)
    base = os.path.join(CUSTOM_MAPS_DIR, f"{mode}_{name}")
    with open(base + ".txt", "w", encoding="ascii") as f:
        f.write(text)
    img = render_map_preview(tiles, scale=14)
    for x, y in spawns:   # white dot for spawn markers
        cy, cx = (y + 0.5) * 14, (x + 0.5) * 14
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = (240, 240, 240)
    Image.fromarray(img).save(base + ".png")
    return base + ".txt"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/builtin":
            self._send(200, json.dumps(_builtin_json()).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            path = save_map(payload["mode"], payload["name"], payload["rows"])
            out = {"ok": True, "message": f"saved to {path}"}
            self._send(200, json.dumps(out).encode(), "application/json")
        except Exception as e:
            out = {"ok": False, "message": f"invalid map: {e}"}
            self._send(400, json.dumps(out).encode(), "application/json")

    def log_message(self, *args):
        pass


def main():
    os.makedirs(CUSTOM_MAPS_DIR, exist_ok=True)
    print(f"map editor on http://{HOST}:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
