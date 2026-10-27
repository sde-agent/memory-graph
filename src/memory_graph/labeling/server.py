"""Local (127.0.0.1-only) labeling web UI — simple, fast, one task at a time.

Serves PRE-RENDERED static thumbnails (prepared by labeling/prepare.py) so there is no
on-demand decode and no per-image Photos DB scan — the earlier slowness is gone. The UI
guides you through three short tasks IN SEQUENCE with a clear target and progress bar;
each screen is ONE photo (or one pair) and ONE click. You label a few HUNDRED, not
thousands, and can stop/resume anytime.

Run: `memory-graph label` -> open http://127.0.0.1:8737/
"""

from __future__ import annotations

import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from memory_graph.config import load_config
from memory_graph.labeling.prepare import label_set_rows, prepare_label_set, thumb_path
from memory_graph.labeling.store import LabelStore

HOST = "127.0.0.1"
PORT = 8737

# Per-task targets (a few hundred clicks total — NOT the whole archive).
TARGETS = {"base_rate": 100, "quadrant": 150, "pairwise": 100}
TASK_ORDER = ["base_rate", "quadrant", "pairwise"]
TASK_TITLE = {
    "base_rate": "Would you want the app to send you THIS, unprompted, as your one memory today?",
    "quadrant": "What feeling does this evoke?",
    "pairwise": "Which would you rather be shown?",
}

_QUADRANTS = [
    (1, "Nostalgic / Reflective"),
    (2, "Visceral / Awe"),
    (3, "Tender / Connected"),
    (4, "Solitary / Calm"),
    (5, "Provocative / Tense"),
]


class _State:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.rows = label_set_rows()
        self.ids = [r["id"] for r in self.rows]
        self.base = LabelStore("base_rate")
        self.quad = LabelStore("quadrant")
        self.pair = LabelStore("pairwise")
        self.rng = random.Random(self.cfg.dispatch.seed)

    def store(self, kind: str) -> LabelStore:
        return {"base_rate": self.base, "quadrant": self.quad, "pairwise": self.pair}[kind]

    def progress(self) -> dict:
        return {k: min(self.store(k).count(), TARGETS[k]) for k in TASK_ORDER}

    def next_single(self, kind: str) -> str | None:
        done = self.store(kind).labeled_keys()
        pool = [i for i in self.ids if i not in done]
        return self.rng.choice(pool) if pool else None

    def next_pair(self) -> tuple[str, str] | None:
        if len(self.ids) < 2:
            return None
        return tuple(self.rng.sample(self.ids, 2))  # type: ignore[return-value]


class _Handler(BaseHTTPRequestHandler):
    state: _State = None

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/thumb":
            self._serve_thumb(parse_qs(parsed.query).get("id", [""])[0])
            return
        if parsed.path == "/api/state":
            s = self.state
            self._json({
                "progress": s.progress(),
                "targets": TARGETS,
                "order": TASK_ORDER,
                "titles": TASK_TITLE,
                "quadrants": [{"q": q, "name": n} for q, n in _QUADRANTS],
                "set_size": len(s.ids),
            })
            return
        if parsed.path == "/api/next":
            self._next(parse_qs(parsed.query).get("kind", ["base_rate"])[0])
            return
        self.send_error(404)

    def _serve_thumb(self, pid: str):
        path = thumb_path(pid)
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _next(self, kind: str):
        s = self.state
        if s.progress().get(kind, 0) >= TARGETS.get(kind, 0):
            self._json({"complete": True})
            return
        if kind == "pairwise":
            pair = s.next_pair()
            if not pair:
                self._json({"complete": True})
                return
            self._json({"kind": kind, "a": pair[0], "b": pair[1]})
            return
        pid = s.next_single(kind)
        if pid is None:
            self._json({"complete": True})
            return
        self._json({"kind": kind, "id": pid})

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        p = json.loads(self.rfile.read(length) or b"{}")
        s = self.state
        if parsed.path == "/api/label/base_rate":
            s.base.append({"id": p["id"], "worthy": bool(p["worthy"])})
        elif parsed.path == "/api/label/quadrant":
            s.quad.append({"id": p["id"], "quadrant": int(p["quadrant"])})
        elif parsed.path == "/api/label/pairwise":
            s.pair.append({"id": f"{p['a']}|{p['b']}", "a": p["a"], "b": p["b"],
                           "winner": p["winner"]})
        else:
            self.send_error(404)
            return
        self._json({"ok": True})


def serve(host: str = HOST, port: int = PORT) -> None:
    print("Preparing the labeling set (one-time thumbnail render)...")
    summary = prepare_label_set()
    print(f"  set: {summary['set_size']} photos "
          f"(rendered {summary['rendered']}, reused {summary['reused']}, "
          f"failed {summary['failed']})")
    _Handler.state = _State()
    if not _Handler.state.ids:
        print("No photos available to label — run `sample-run` or `analyze-hdd` first.")
        return
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"\n  Open  http://{host}:{port}/  in your browser.")
    total = sum(TARGETS.values())
    print(f"  ~{total} quick clicks total; stop and resume anytime. Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Memory Graph — Labeling</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 820px;
         margin: 2rem auto; color: #1a1a1a; padding: 0 1rem; }
  h1 { font-size: 1.1rem; margin-bottom: .25rem; }
  .q { font-size: 1.3rem; margin: 1rem 0; text-align: center; }
  .bar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; margin:.5rem 0; }
  .bar > div { height: 100%; background: #2a7; transition: width .2s; }
  .steps { display:flex; gap:.5rem; font-size:.8rem; color:#999; margin-bottom:1rem; }
  .steps .on { color:#2a7; font-weight:600; }
  .photo { text-align:center; }
  .photo img { max-width:100%; max-height:58vh; border-radius:10px; box-shadow:0 2px 16px #0003; }
  .pair { display:flex; gap:1rem; } .pair > div { flex:1; }
  .pair img { max-height:48vh; }
  .btns { margin-top:1rem; text-align:center; }
  .btns button { font-size:1rem; padding:.7rem 1.2rem; margin:.3rem; cursor:pointer;
                 border-radius:8px; border:1px solid #ccc; background:#fafafa; }
  .btns button:hover { background:#2a7; color:#fff; border-color:#2a7; }
  .skip { color:#aaa; font-size:.85rem; margin-top:1rem; text-align:center; cursor:pointer; }
  .hint { text-align:center; color:#b36; font-size:.9rem; margin:-.5rem 0 1rem; }
</style></head>
<body>
<h1>Memory Graph — teach it your taste</h1>
<div class="steps" id="steps"></div>
<div class="bar"><div id="prog" style="width:0%"></div></div>
<div class="q" id="question"></div>
<p class="hint" id="hint"></p>
<div id="card"></div>
<p class="skip" id="skip" onclick="skip()">skip this one &rsaquo;</p>
<script>
let S=null, task='base_rate', cur=null;
function pct(k){ return Math.round(100*Math.min(S.progress[k],S.targets[k])/S.targets[k]); }
async function state(){ S = await (await fetch('/api/state')).json(); render_steps(); }
function render_steps(){
  document.getElementById('steps').innerHTML = S.order.map(k =>
    `<span class="${k===task?'on':''}">${k.replace('_',' ')} `+
    `${Math.min(S.progress[k],S.targets[k])}/${S.targets[k]}</span>`).join(' · ');
}
function thumb(id){ return `<img src="/thumb?id=${encodeURIComponent(id)}">`; }
async function load(){
  await state();
  // advance to the first unfinished task
  for(const k of S.order){ if(S.progress[k] < S.targets[k]){ task=k; break; } task=k; }
  render_steps();
  document.getElementById('prog').style.width = pct(task)+'%';
  document.getElementById('question').textContent = S.titles[task];
  document.getElementById('hint').textContent = (task==='base_rate')
    ? 'Be strict: not "nice to see again" but "would I want THIS pushed to me today?" '
      + 'Most photos should be No.'
    : '';
  const n = await (await fetch('/api/next?kind='+task)).json();
  const card = document.getElementById('card');
  if(n.complete){
    // move to next task or finish
    const idx = S.order.indexOf(task);
    if(idx < S.order.length-1){ task = S.order[idx+1]; return load(); }
    card.innerHTML = '<p style="text-align:center">All done — thank you! '+
      'You can close this tab.</p>';
    document.getElementById('skip').style.display='none';
    document.getElementById('question').textContent='Finished 🎉';
    return;
  }
  cur = n;
  if(task==='base_rate'){
    card.innerHTML = `<div class="photo">${thumb(n.id)}</div>`+
      `<div class="btns"><button onclick="send({worthy:true})">Yes, send me this</button>`+
      `<button onclick="send({worthy:false})">No, skip it</button></div>`;
  } else if(task==='quadrant'){
    const qb = S.quadrants.map(q=>
      `<button onclick="send({quadrant:${q.q}})">${q.name}</button>`).join('');
    card.innerHTML = `<div class="photo">${thumb(n.id)}</div><div class="btns">${qb}</div>`;
  } else {
    card.innerHTML = `<div class="pair">`+
      `<div class="photo">${thumb(n.a)}<div class="btns">`+
      `<button onclick="send({winner:'a'})">This one</button></div></div>`+
      `<div class="photo">${thumb(n.b)}<div class="btns">`+
      `<button onclick="send({winner:'b'})">This one</button></div></div></div>`;
  }
}
async function send(extra){
  const body = Object.assign({}, cur, extra);
  await fetch('/api/label/'+task, {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  load();
}
function skip(){ load(); }
load();
</script>
</body></html>
"""
