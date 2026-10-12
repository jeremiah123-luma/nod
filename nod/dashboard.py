"""The Nod Command Center: a local dashboard in your browser.

- Standard library only (no new dependencies). Listens on 127.0.0.1 ONLY, so nothing else on your
  network can reach it.
- It shows numbers and a wireframe of your face (landmark points). It never sends camera pictures.
- Anything that changes something (pause, recenter, settings...) needs a secret token that is only
  handed to the page we serve, and a JSON content type. A random website in another tab can't
  read the token (browser same-origin rule) and can't guess it.
"""
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nod import settings

UI_DIR = Path(__file__).resolve().parent / "ui"
STATIC = {
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
COMMANDS = {"pause", "resume", "toggle_pause", "recenter", "calibrate", "toggle_dwell", "toggle_gaze", "voice_toggle"}
MAX_BODY = 16 * 1024
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class Dashboard:
    def __init__(self, get_cfg, on_command, on_settings, meta_extra=None, host="127.0.0.1", port=8765, config_path=None):
        self.get_cfg = get_cfg                  # () -> current config dict
        self.on_command = on_command            # (action) -> dict with extra info, runs on a server thread
        self.on_settings = on_settings          # (new_cfg) -> None, applies live
        self.meta_extra = meta_extra or (lambda: {})
        self.host, self.port = host, port
        self.config_path = config_path          # where settings are saved (None = the real config.json)
        self.token = secrets.token_urlsafe(24)
        self._cond = threading.Condition()
        self._seq = 0
        self._payload = b"{}"
        self._server = None
        self._thread = None

    # ---- called by the main loop ----
    def publish(self, state):
        """Latest state for every open dashboard tab. Cheap: just stores JSON and wakes the senders."""
        data = json.dumps(state, separators=(",", ":")).encode()
        with self._cond:
            self._payload = data
            self._seq += 1
            self._cond.notify_all()

    # ---- lifecycle ----
    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    def start(self):
        handler = _make_handler(self)
        last_error = None
        for port in ([0] if self.port == 0 else range(self.port, self.port + 10)):   # if busy, try the next few
            try:
                self._server = ThreadingHTTPServer((self.host, port), handler)
                self.port = self._server.server_address[1]     # the real port (matters when port=0)
                break
            except OSError as e:
                last_error = e
        if self._server is None:
            raise OSError(f"could not open a dashboard port: {last_error}")
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.05),
                                        daemon=True, name="nod-dashboard")
        self._thread.start()
        return self.url

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._cond:
            self._seq += 1
            self._cond.notify_all()

    # ---- used by the request handler ----
    def wait_for_update(self, last_seq, timeout):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._payload

    def meta(self):
        cfg = self.get_cfg()
        out = {"settings": settings.schema_for_ui(cfg), "hotkeys": cfg["hotkeys"], "version": "1"}
        out.update(self.meta_extra())
        return out


def _make_handler(dash):
    allowed_hosts = None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "NodDashboard"

        def log_message(self, *args):    # keep the terminal quiet
            pass

        # ---- helpers ----
        def _host_ok(self):
            # Blocks "DNS rebinding": a web page tricking your browser into talking to us under another name.
            host = (self.headers.get("Host") or "").lower()
            return host in {f"127.0.0.1:{dash.port}", f"localhost:{dash.port}"}

        def _token_ok(self, query=None):
            supplied = self.headers.get("X-Nod-Token") or (query or {}).get("t", [""])[0]
            return secrets.compare_digest(supplied, dash.token)

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        # ---- GET ----
        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = url.path

            if path in ("/", "/index.html"):
                html = (UI_DIR / "index.html").read_text().replace("__NOD_TOKEN__", dash.token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if path in STATIC:
                name, ctype = STATIC[path]
                f = UI_DIR / name
                if not f.exists():
                    return self._json(404, {"error": "not found"})
                return self._send(200, f.read_bytes(), ctype)
            if path == "/api/meta":
                if not self._token_ok(query):
                    return self._json(401, {"error": "bad token"})
                return self._json(200, dash.meta())
            if path == "/events":
                if not self._token_ok(query):
                    return self._json(401, {"error": "bad token"})
                return self._stream()
            return self._json(404, {"error": "not found"})

        def _stream(self):
            """Server-sent events: the page keeps one connection open and we push each new state."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            seq = -1
            try:
                while dash._server is not None:
                    # Wakes on every new state, or every 2 s with the last one: a heartbeat, so the page can tell
                    # "Nod is busy" (e.g. calibrating) apart from "Nod has stopped".
                    seq, payload = dash.wait_for_update(seq, 2)
                    if dash._server is None:
                        break
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True

        # ---- POST ----
        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            if not self._token_ok():
                return self._json(401, {"error": "bad token"})
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json(415, {"error": "send JSON"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "bad length"})
            if length <= 0 or length > MAX_BODY:
                return self._json(413, {"error": "bad body size"})
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                return self._json(400, {"error": "bad JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "expected an object"})

            path = urlparse(self.path).path
            if path == "/api/command":
                action = body.get("action")
                if action not in COMMANDS:
                    return self._json(400, {"error": f"unknown action: {action}"})
                return self._json(200, {"ok": True, **(dash.on_command(action) or {})})
            if path == "/api/settings":
                try:
                    new_cfg = settings.apply_changes(dash.get_cfg(), body.get("changes"))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                try:
                    settings.save_config(new_cfg, dash.config_path)
                    dash.on_settings(new_cfg)
                except Exception as e:      # e.g. disk problem, or the controller rejected the combination
                    return self._json(500, {"error": f"could not apply: {e}"})
                return self._json(200, {"ok": True, "settings": settings.schema_for_ui(new_cfg)})
            return self._json(404, {"error": "not found"})

        def do_OPTIONS(self):
            # No CORS: we never allow other websites to call us. Say so plainly.
            self._json(403, {"error": "cross-site requests are not allowed"})

    return Handler
