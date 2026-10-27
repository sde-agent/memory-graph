"""A local web page for the blinded holdout labelling (evaluation.md Parts U and V).

One screen shows the photo, the question, Yes/No, and (only after Yes) the five feelings.
Answering No saves at once. Answers go to the round's append-only file. It listens on
127.0.0.1 only, so photos never leave the Mac. The page never says whether a photo came from
the engine's top 3 or from the random controls.
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memory_graph.evaluation import holdout as H

HOST = "127.0.0.1"
PORT = 8742

# Plain-language wording. The meaning is the same as the tuning question (Part U).
QUESTION = ("Imagine this photo arrives on your phone today as your one memory. "
            "Would you be glad to see it?")
QUADRANTS = [
    {"q": 1, "name": "Nostalgic",
     "desc": "warm and familiar, a memory of home or a moment you want to keep"},
    {"q": 2, "name": "Excited",
     "desc": "big and bold, the feeling of something vast or full of energy"},
    {"q": 3, "name": "Close",
     "desc": "tender and personal, connected to people you care about"},
    {"q": 4, "name": "Quiet",
     "desc": "still and solitary, calm, with plenty of empty space"},
    {"q": 5, "name": "Tense",
     "desc": "unsettling or a little mysterious, something off-balance"},
]

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")
_LOCK = threading.Lock()


def thumb_name(photo_id: str) -> str:
    return photo_id.replace(":", "_") + ".jpg"


def _answered_ids(label_file) -> set[str]:
    return H._read_jsonl_ids(label_file)


def state(set_obj: dict, label_file=None) -> dict:
    """What the page needs: progress, the next photo, the question, the feelings. Blind."""
    label_file = label_file or H.LABEL_FILE
    items = set_obj["items"]
    done = _answered_ids(label_file)
    pending = [it for it in items
               if it["id"] not in done and (H.THUMB_DIR / thumb_name(it["id"])).is_file()]
    nxt = None
    if pending:
        nxt = {"id": pending[0]["id"], "image": "/thumb/" + thumb_name(pending[0]["id"])}
    return {"question": QUESTION, "quadrants": QUADRANTS, "total": len(items),
            "done": len([i for i in items if i["id"] in done]), "next": nxt}


def record(set_obj: dict, body: dict, label_file=None, multi: bool = False
           ) -> tuple[bool, str]:
    """Validate and append one answer.

    Single-choice (rounds 1 and 2, Part V): No needs no feeling; Yes needs one from 1 to 5.
    Multi (round 3, Part W): every photo needs yes or no, and at least one feeling from 1 to 5.
    """
    label_file = label_file or H.LABEL_FILE
    pid = body.get("id")
    worthy = body.get("worthy")
    valid_ids = {it["id"] for it in set_obj["items"]}
    if pid not in valid_ids:
        return False, "unknown photo"
    if not isinstance(worthy, bool):
        return False, "answer yes or no first"
    if multi:
        qs = body.get("quadrants")
        if not (isinstance(qs, list) and qs and len(set(qs)) == len(qs)
                and all(isinstance(q, int) and q in {1, 2, 3, 4, 5} for q in qs)):
            return False, "pick at least one feeling"
        record_out = {"id": pid, "worthy": worthy, "quadrants": sorted(qs)}
    else:
        quad = body.get("quadrant")
        if worthy and quad not in {1, 2, 3, 4, 5}:
            return False, "choose a feeling from 1 to 5"
        record_out = {"id": pid, "worthy": worthy, "quadrant": quad if worthy else None}
    with _LOCK:
        if pid in _answered_ids(label_file):
            return False, "already answered"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with label_file.open("a") as fh:
            fh.write(json.dumps(record_out) + "\n")
    return True, "saved"


# Round 3 (Part W): yes or no, then every feeling that applies. More than one is allowed.
PAGE_MULTI = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory check</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f6f5f2;
         color: #222; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  .how { background: #fff; border-radius: 10px; padding: 14px 16px; margin: 14px 0;
         line-height: 1.5; font-size: 15px; }
  .how ol { margin: 6px 0 0 18px; padding: 0; }
  .progress { font-size: 14px; color: #555; margin: 8px 0; }
  .bar { height: 8px; background: #e2e0da; border-radius: 4px; overflow: hidden; }
  .bar div { height: 100%; background: #4a7c59; }
  img.photo { width: 100%; max-height: 62vh; object-fit: contain; background: #111;
              border-radius: 10px; margin: 12px 0; }
  .question { font-size: 18px; font-weight: 600; margin: 10px 0; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
  button { font-size: 16px; padding: 12px 18px; border-radius: 8px; border: 2px solid #bbb;
           background: #fff; cursor: pointer; }
  button:hover { border-color: #4a7c59; }
  button.on { border-color: #4a7c59; background: #e7f1e9; font-weight: 600; }
  .feel { flex: 1 1 210px; text-align: left; }
  .feel small { display: block; color: #555; font-weight: normal; margin-top: 3px;
                font-size: 13px; }
  button.save { background: #4a7c59; color: #fff; border-color: #4a7c59; font-weight: 600;
                opacity: 0.4; }
  button.save.ready { opacity: 1; }
  #msg { min-height: 22px; color: #a33; font-size: 14px; }
  .keys { font-size: 13px; color: #666; margin-top: 12px; }
  .done { font-size: 18px; padding: 30px 0; text-align: center; }
</style></head>
<body><main>
  <h1>Memory check</h1>
  <div class="how">
    <strong>What to do, for each photo:</strong>
    <ol>
      <li>Look at the photo.</li>
      <li>Would you be glad to see this again today? Press <b>Yes</b> or <b>No</b>.</li>
      <li>Pick <b>every</b> feeling the photo gives you. More than one is fine.
          Pick at least one.</li>
      <li>Press <b>Save and next</b>.</li>
    </ol>
    <div class="keys">Keyboard: <b>Y</b> or <b>N</b>, then <b>1</b> to <b>5</b> to pick or unpick
      a feeling, then <b>Enter</b> to save.</div>
  </div>
  <div class="progress" id="progress"></div>
  <div class="bar"><div id="fill" style="width:0%"></div></div>
  <div id="card">
    <img class="photo" id="photo" alt="photo to rate">
    <div class="question" id="question"></div>
    <div class="row">
      <button id="yes">Yes, I'd be glad to see it</button>
      <button id="no">No</button>
    </div>
    <div class="question" style="font-size:16px">
      Which feelings does it give you? Pick all that apply.</div>
    <div class="row" id="feelings"></div>
    <div class="row"><button class="save" id="save">Save and next</button></div>
    <div id="msg"></div>
  </div>
  <div class="done" id="finished" style="display:none"></div>
</main>
<script>
let current = null, worthy = null, picked = new Set(), busy = false;
const $ = id => document.getElementById(id);
function refresh() {
  $('yes').classList.toggle('on', worthy === true);
  $('no').classList.toggle('on', worthy === false);
  document.querySelectorAll('#feelings button').forEach(b =>
    b.classList.toggle('on', picked.has(+b.dataset.q)));
  const ready = worthy !== null && picked.size > 0;
  $('save').classList.toggle('ready', ready);
}
function setWorthy(v) { worthy = v; $('msg').textContent = ''; refresh(); }
function togglePick(q) {
  if (picked.has(q)) picked.delete(q); else picked.add(q);
  $('msg').textContent = ''; refresh();
}
async function save() {
  if (busy || !current) return;
  if (worthy === null) { $('msg').textContent = 'First press Yes or No.'; return; }
  if (picked.size === 0) { $('msg').textContent = 'Pick at least one feeling.'; return; }
  busy = true;
  const r = await fetch('/api/answer', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: current.id, worthy: worthy, quadrants: [...picked]})});
  const out = await r.json();
  if (!out.ok) { $('msg').textContent = out.message; busy = false; return; }
  load();
}
async function load() {
  const r = await fetch('/api/state'); const s = await r.json();
  $('progress').textContent = `${s.done} of ${s.total} done`;
  $('fill').style.width = (s.total ? 100 * s.done / s.total : 0) + '%';
  if (!s.next) {
    $('card').style.display = 'none';
    $('finished').style.display = 'block';
    $('finished').textContent = s.done >= s.total
      ? 'All done. Thank you. You can close this page.'
      : 'Nothing left to show here yet.';
    return;
  }
  current = s.next; busy = false; worthy = null; picked = new Set();
  $('photo').src = s.next.image + '?t=' + Date.now();
  $('question').textContent = s.question;
  const box = $('feelings'); box.innerHTML = '';
  s.quadrants.forEach(q => {
    const b = document.createElement('button');
    b.className = 'feel'; b.dataset.q = q.q;
    b.innerHTML = `${q.q}. ${q.name}<small>${q.desc}</small>`;
    b.onclick = () => togglePick(q.q);
    box.appendChild(b);
  });
  $('msg').textContent = '';
  refresh();
}
$('yes').onclick = () => setWorthy(true);
$('no').onclick = () => setWorthy(false);
$('save').onclick = save;
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'y') setWorthy(true);
  else if (k === 'n') setWorthy(false);
  else if ('12345'.includes(k)) togglePick(parseInt(k, 10));
  else if (k === 'enter') save();
});
load();
</script></body></html>
"""


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory check</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f6f5f2;
         color: #222; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 20px 40px; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  .how { background: #fff; border-radius: 10px; padding: 14px 16px; margin: 14px 0;
         line-height: 1.5; font-size: 15px; }
  .how ol { margin: 6px 0 0 18px; padding: 0; }
  .progress { font-size: 14px; color: #555; margin: 8px 0; }
  .bar { height: 8px; background: #e2e0da; border-radius: 4px; overflow: hidden; }
  .bar div { height: 100%; background: #4a7c59; }
  img.photo { width: 100%; max-height: 62vh; object-fit: contain; background: #111;
              border-radius: 10px; margin: 12px 0; }
  .question { font-size: 18px; font-weight: 600; margin: 10px 0; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
  button { font-size: 16px; padding: 12px 18px; border-radius: 8px; border: 2px solid #bbb;
           background: #fff; cursor: pointer; }
  button:hover { border-color: #4a7c59; }
  button.on { border-color: #4a7c59; background: #e7f1e9; font-weight: 600; }
  .feel { flex: 1 1 210px; text-align: left; }
  .feel small { display: block; color: #555; font-weight: normal; margin-top: 3px;
                font-size: 13px; }
  #feelBlock { display: none; }
  #msg { min-height: 22px; color: #a33; font-size: 14px; }
  .keys { font-size: 13px; color: #666; margin-top: 12px; }
  .done { font-size: 18px; padding: 30px 0; text-align: center; }
</style></head>
<body><main>
  <h1>Memory check</h1>
  <div class="how">
    <strong>What to do, for each photo:</strong>
    <ol>
      <li>Look at the photo.</li>
      <li>Ask yourself: would I be glad to see this again today? Press <b>Yes</b> or <b>No</b>.</li>
      <li>If you pressed <b>No</b>, that saves it and shows the next photo.</li>
      <li>If you pressed <b>Yes</b>, pick the <b>feeling</b> the photo gives you. That saves it
          and shows the next one.</li>
    </ol>
    <div class="keys">Keyboard: <b>Y</b> or <b>N</b>. After Y, press <b>1</b> to <b>5</b>.</div>
  </div>
  <div class="progress" id="progress"></div>
  <div class="bar"><div id="fill" style="width:0%"></div></div>
  <div id="card">
    <img class="photo" id="photo" alt="photo to rate">
    <div class="question" id="question"></div>
    <div class="row">
      <button id="yes">Yes, I'd be glad to see it</button>
      <button id="no">No</button>
    </div>
    <div id="feelBlock">
      <div class="question" style="font-size:16px">Which feeling does it give you?</div>
      <div class="row" id="feelings"></div>
    </div>
    <div id="msg"></div>
  </div>
  <div class="done" id="finished" style="display:none"></div>
</main>
<script>
let current = null, busy = false;
const $ = id => document.getElementById(id);
function setYes(on) {
  $('yes').classList.toggle('on', on);
  $('no').classList.remove('on');
  $('feelBlock').style.display = on ? 'block' : 'none';
  $('msg').textContent = '';
}
async function send(worthy, quadrant) {
  if (busy || !current) return;
  busy = true;
  const r = await fetch('/api/answer', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: current.id, worthy: worthy, quadrant: quadrant})});
  const out = await r.json();
  if (!out.ok) { $('msg').textContent = out.message; busy = false; return; }
  load();
}
async function load() {
  const r = await fetch('/api/state'); const s = await r.json();
  $('progress').textContent = `${s.done} of ${s.total} done`;
  $('fill').style.width = (s.total ? 100 * s.done / s.total : 0) + '%';
  if (!s.next) {
    $('card').style.display = 'none';
    $('finished').style.display = 'block';
    $('finished').textContent = s.done >= s.total
      ? 'All done. Thank you. You can close this page.'
      : 'Nothing left to show here yet.';
    return;
  }
  current = s.next; busy = false;
  $('photo').src = s.next.image + '?t=' + Date.now();
  $('question').textContent = s.question;
  $('yes').classList.remove('on'); $('no').classList.remove('on');
  $('feelBlock').style.display = 'none';
  const box = $('feelings'); box.innerHTML = '';
  s.quadrants.forEach(q => {
    const b = document.createElement('button');
    b.className = 'feel';
    b.innerHTML = `${q.q}. ${q.name}<small>${q.desc}</small>`;
    b.onclick = () => send(true, q.q);
    box.appendChild(b);
  });
  $('msg').textContent = '';
}
$('yes').onclick = () => setYes(true);
$('no').onclick = () => send(false, null);
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'y') setYes(true);
  else if (k === 'n') send(false, null);
  else if ('12345'.includes(k)) {
    if ($('feelBlock').style.display === 'block') send(true, parseInt(k, 10));
    else $('msg').textContent = 'First press Y (Yes).';
  }
});
load();
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    set_obj: dict = {}
    label_file = None
    multi = False

    def log_message(self, *args):   # keep the terminal quiet
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):   # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            page = PAGE_MULTI if self.multi else PAGE
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(state(self.set_obj, self.label_file))
        elif path.startswith("/thumb/"):
            name = path[len("/thumb/"):]
            if not _SAFE_NAME.match(name):
                return self._send(404, b"not found", "text/plain")
            f = (H.THUMB_DIR / name).resolve()
            if f.parent != H.THUMB_DIR.resolve() or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, f.read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):   # noqa: N802
        if self.path.split("?", 1)[0] != "/api/answer":
            return self._send(404, b"not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"ok": False, "message": "bad request"}, 400)
        ok, message = record(self.set_obj, body, self.label_file, multi=self.multi)
        self._json({"ok": ok, "message": message}, 200 if ok else 400)


def serve(set_obj: dict, label_file, *, multi: bool = False, port: int = PORT,
          open_browser: bool = True) -> None:
    """Run the labelling page on 127.0.0.1 until Ctrl-C. multi=True is the Part W page."""
    _Handler.set_obj = set_obj
    _Handler.label_file = label_file
    _Handler.multi = multi
    httpd = ThreadingHTTPServer((HOST, port), _Handler)
    url = f"http://{HOST}:{port}/"
    print(f"Memory check is running at {url}")
    print("Keep this Terminal window open while you answer. Press Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your answers so far are saved.")
    finally:
        httpd.server_close()
