"""Loopback-only photo viewer for the MCP dispatch (ZERO-EGRESS enforcement).

Hard rule: photo PIXELS never leave this Mac. A hosted LLM (Claude/ChatGPT) therefore
may never receive image bytes — showing a photo to a cloud model IS sending it. So the
MCP `daily_dispatch` returns only TEXT (grounded facts) plus a `http://127.0.0.1:<port>`
link. This module serves that link: it renders the dispatched photo in the owner's
browser, from the local `deriv_path` (preview/derivative — never a master). The pixels
transit only localhost; Anthropic/OpenAI only ever see text.

Bound explicitly to 127.0.0.1 (never 0.0.0.0) so it is not reachable off the machine.
"""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from memory_graph import db as dbmod

HOST = "127.0.0.1"
PORT = 8738


def _render_jpeg(deriv_path: str, max_px: int = 1400) -> bytes | None:
    import os

    from PIL import Image

    if not deriv_path or not os.path.isfile(deriv_path):
        return None
    try:
        img = Image.open(deriv_path).convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        pid = parse_qs(parsed.query).get("id", [""])[0]
        if parsed.path not in ("/photo", "/view") or not pid:
            self.send_error(404)
            return

        conn = dbmod.connect()
        try:
            row = dbmod.get_photo(conn, pid)
        finally:
            conn.close()
        # Never serve a private (Hidden-album) photo, even locally via a guessed id.
        if not row or row.get("private"):
            self.send_error(404)
            return

        if parsed.path == "/photo":
            data = _render_jpeg(row.get("deriv_path") or "")
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # /view: a minimal HTML page framing the photo (nice when opened in a browser).
        html = (
            f"<!doctype html><meta charset=utf-8>"
            f"<body style='margin:0;background:#111;display:flex;align-items:center;"
            f"justify-content:center;height:100vh'>"
            f"<img src='/photo?id={pid}' style='max-width:100%;max-height:100vh'>"
            f"</body>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


_server: ThreadingHTTPServer | None = None


def ensure_running(port: int = PORT) -> str:
    """Start the loopback viewer in a background thread if not already up; return base URL.

    Idempotent and safe to call from the MCP server at startup.
    """
    global _server
    if _server is None:
        try:
            _server = ThreadingHTTPServer((HOST, port), _Handler)
            thread = threading.Thread(target=_server.serve_forever, daemon=True)
            thread.start()
        except OSError:
            # Port already in use — a viewer is almost certainly already running (e.g. a
            # previous MCP session). Reuse it rather than crashing MCP startup.
            pass
    return f"http://{HOST}:{port}"


def view_url(photo_id: str, port: int = PORT) -> str:
    from urllib.parse import quote

    return f"http://{HOST}:{port}/view?id={quote(photo_id)}"
