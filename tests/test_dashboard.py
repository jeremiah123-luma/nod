"""The dashboard server: it must work, and it must be hard to abuse."""
import http.client
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from nod.config import load_config
from nod.dashboard import Dashboard


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg_path = os.path.join(self.tmp.name, "config.json")
        self.commands, self.applied = [], []
        self.dash = Dashboard(lambda: self.cfg, lambda a: (self.commands.append(a) or {"paused": True}),
                              lambda c: self.applied.append(c), port=0, config_path=self.cfg_path)
        self.dash.start()
        self.port = self.dash.port

    def tearDown(self):
        self.dash.stop()
        self.tmp.cleanup()

    # tiny HTTP helper: returns (status, headers, body)
    def req(self, method, path, body=None, headers=None, host=None, token=True, ctype="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            h["X-Nod-Token"] = self.dash.token
        if body is not None:
            h["Content-Type"] = ctype
            body = json.dumps(body) if not isinstance(body, (bytes, str)) else body
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, dict(r.getheaders()), data

    def post_json(self, path, obj, **kw):
        s, _, d = self.req("POST", path, obj, **kw)
        return s, json.loads(d)


class BasicTests(ServerCase):
    def test_serves_page_with_token_injected(self):
        s, h, body = self.req("GET", "/", token=False)
        self.assertEqual(s, 200)
        self.assertIn(self.dash.token.encode(), body)
        self.assertNotIn(b"__NOD_TOKEN__", body)
        self.assertIn("text/html", h["Content-Type"])

    def test_binds_to_localhost_only(self):
        self.assertEqual(self.dash._server.server_address[0], "127.0.0.1")

    def test_security_headers_present(self):
        _, h, _ = self.req("GET", "/", token=False)
        self.assertIn("default-src 'self'", h["Content-Security-Policy"])
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        self.assertEqual(h["Cache-Control"], "no-store")

    def test_meta_lists_settings_and_hotkeys(self):
        s, _, d = self.req("GET", "/api/meta")
        meta = json.loads(d)
        self.assertEqual(s, 200)
        self.assertTrue(any(x["key"] == "click.threshold" for x in meta["settings"]))
        self.assertEqual(meta["hotkeys"]["pause"], self.cfg["hotkeys"]["pause"])

    def test_unknown_path_404(self):
        self.assertEqual(self.req("GET", "/etc/passwd")[0], 404)
        self.assertEqual(self.req("GET", "/../config.json")[0], 404)

    def test_no_path_traversal_through_static_files(self):
        for p in ("/ui/../config.json", "/%2e%2e/config.json", "/manifest.webmanifest/../../config.json"):
            self.assertIn(self.req("GET", p)[0], (404,))


class SecurityTests(ServerCase):
    def test_wrong_host_header_is_refused_even_with_token(self):      # DNS rebinding defence
        for host in ("evil.example.com", f"evil.example.com:{self.port}", "127.0.0.1.evil.com", "0.0.0.0:%d" % self.port):
            self.assertEqual(self.req("GET", "/", host=host)[0], 403, host)
            self.assertEqual(self.req("POST", "/api/command", {"action": "pause"}, host=host)[0], 403, host)
        self.assertEqual(self.commands, [])

    def test_localhost_name_is_allowed(self):
        self.assertEqual(self.req("GET", "/", host=f"localhost:{self.port}")[0], 200)

    def test_commands_need_the_token(self):
        self.assertEqual(self.req("POST", "/api/command", {"action": "toggle_pause"}, token=False)[0], 401)
        self.assertEqual(self.req("POST", "/api/command", {"action": "toggle_pause"}, token=False,
                                  headers={"X-Nod-Token": "guess"})[0], 401)
        self.assertEqual(self.commands, [])

    def test_settings_need_the_token(self):
        self.assertEqual(self.req("POST", "/api/settings", {"changes": {"pointer.gain_x": 100}}, token=False)[0], 401)
        self.assertFalse(os.path.exists(self.cfg_path))
        self.assertEqual(self.applied, [])

    def test_read_endpoints_need_the_token(self):
        self.assertEqual(self.req("GET", "/api/meta", token=False)[0], 401)
        self.assertEqual(self.req("GET", "/events?t=wrong", token=False)[0], 401)

    def test_posts_must_be_json(self):
        s, _, _ = self.req("POST", "/api/command", '{"action":"pause"}', ctype="text/plain")
        self.assertEqual(s, 415)                       # a cross-site <form> can only send text/plain or form data
        s, _, _ = self.req("POST", "/api/command", "action=pause", ctype="application/x-www-form-urlencoded")
        self.assertEqual(s, 415)
        self.assertEqual(self.commands, [])

    def test_cors_preflight_is_refused_and_no_cors_headers_are_ever_sent(self):
        s, h, _ = self.req("OPTIONS", "/api/command", headers={"Origin": "https://evil.example",
                                                              "Access-Control-Request-Method": "POST"})
        self.assertEqual(s, 403)
        _, h2, _ = self.req("GET", "/", token=False, headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", {k.title() for k in h2} | set(h2))

    def test_oversized_and_malformed_bodies_rejected(self):
        self.assertEqual(self.req("POST", "/api/command", "x" * 20000)[0], 413)
        self.assertEqual(self.req("POST", "/api/command", "{not json")[0], 400)
        self.assertEqual(self.req("POST", "/api/command", "[1,2]")[0], 400)

    def test_token_is_long_and_random(self):
        self.assertGreaterEqual(len(self.dash.token), 30)
        other = Dashboard(lambda: self.cfg, lambda a: None, lambda c: None, port=0)
        self.assertNotEqual(self.dash.token, other.token)


class CommandTests(ServerCase):
    def test_known_commands_run(self):
        for a in ("pause", "resume", "toggle_pause", "recenter", "calibrate", "toggle_dwell", "toggle_gaze", "voice_toggle"):
            s, d = self.post_json("/api/command", {"action": a})
            self.assertEqual(s, 200, a)
            self.assertTrue(d["ok"])
        self.assertEqual(len(self.commands), 8)

    def test_unknown_or_dangerous_commands_are_refused(self):
        for a in ("quit", "exit", "rm -rf", "", None, 5, "PAUSE"):
            s, _ = self.post_json("/api/command", {"action": a})
            self.assertEqual(s, 400, a)
        self.assertEqual(self.commands, [])


class SettingsEndpointTests(ServerCase):
    def test_valid_change_is_saved_and_applied(self):
        s, d = self.post_json("/api/settings", {"changes": {"pointer.gain_x": 111, "click.hold_ms": 333}})
        self.assertEqual(s, 200)
        saved = json.load(open(self.cfg_path))
        self.assertEqual(saved["pointer"]["gain_x"], 111)
        self.assertEqual(self.applied[-1]["click"]["hold_ms"], 333)
        self.assertEqual(saved["hotkeys"], self.cfg["hotkeys"])           # everything else untouched

    def test_invalid_changes_save_nothing(self):
        for changes in ({"pointer.gain_x": 99999}, {"hotkeys.pause": "x"}, {"camera.width": 10}, {"a.b.c": 1},
                        {"pointer.gain_x": "x"}, {}, None, {"pointer.invert_x": "yes"}):
            s, d = self.post_json("/api/settings", {"changes": changes})
            self.assertEqual(s, 400, changes)
            self.assertIn("error", d)
        self.assertFalse(os.path.exists(self.cfg_path))
        self.assertEqual(self.applied, [])

    def test_real_config_json_is_never_touched_by_these_tests(self):
        real = load_config()
        self.post_json("/api/settings", {"changes": {"pointer.gain_x": 123}})
        self.assertNotEqual(load_config()["pointer"]["gain_x"], 123)
        self.assertEqual(load_config(), real)


class StreamTests(ServerCase):
    def test_events_stream_pushes_published_state(self):
        self.dash.publish({"hello": 1})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", f"/events?t={self.dash.token}", headers={"Host": f"127.0.0.1:{self.port}"})
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIn("text/event-stream", r.getheader("Content-Type"))
        first = r.fp.readline().decode()
        self.assertEqual(first.strip(), 'data: {"hello":1}')
        r.fp.readline()
        threading.Timer(0.2, lambda: self.dash.publish({"hello": 2})).start()
        self.assertEqual(r.fp.readline().decode().strip(), 'data: {"hello":2}')
        conn.close()

    def test_stop_ends_streams_and_frees_the_port(self):
        self.dash.stop()
        with self.assertRaises(Exception):
            self.req("GET", "/")


if __name__ == "__main__":
    unittest.main()
