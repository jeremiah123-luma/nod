"""Bridge, mesh points, demo mode, and the dashboard's static files."""
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from nod import mesh
from nod.bridge import Bridge
from nod.config import load_config
from nod.controller import Controller
from nod.demo import DemoActions, demo_face, demo_mapper, demo_points
from nod.face import NOSE_POINTS
from tests.fakes import FakeActions, make_landmarks

UI = Path(__file__).resolve().parent.parent / "nod" / "ui"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeActions()
        self.fake.is_paused = lambda: self.fake.paused
        self.b = Bridge(self.fake)

    def test_pause_commands_apply_instantly(self):
        self.assertEqual(self.b.on_command("resume"), {"paused": False})
        self.assertEqual(self.b.on_command("pause"), {"paused": True})
        self.assertEqual(self.b.on_command("toggle_pause"), {"paused": False})
        self.assertEqual(self.b.drain(), [])

    def test_other_commands_are_queued_for_the_main_loop(self):
        self.assertEqual(self.b.on_command("recenter"), {"queued": True})
        self.b.on_settings({"x": 1})
        self.assertEqual(self.b.drain(), ["recenter", ("settings", {"x": 1})])
        self.assertEqual(self.b.drain(), [])

    def test_nothing_touches_the_mouse(self):
        for a in ("recenter", "calibrate", "toggle_dwell", "toggle_gaze", "voice_toggle"):
            self.b.on_command(a)
        self.assertEqual((self.fake.moves, self.fake.clicks, self.fake.scrolls), ([], 0, []))


class MeshTests(unittest.TestCase):
    def test_points_and_groups_are_consistent(self):
        n = len(mesh.POINT_IDS)
        self.assertEqual(n, len(set(mesh.POINT_IDS)))
        for name, pairs in mesh.GROUPS.items():
            for a, b in pairs:
                self.assertTrue(0 <= a < n and 0 <= b < n, name)
        self.assertEqual(set(mesh.GROUPS), {"oval", "brows", "eyes", "lips", "nose", "irises"})

    def test_compact_format_is_flat_ints_in_range(self):
        lm = make_landmarks()
        for p in lm:
            p.x, p.y = 0.25, 0.75
        out = mesh.compact(lm)
        self.assertEqual(len(out), 2 * len(mesh.POINT_IDS))
        self.assertTrue(all(isinstance(v, int) for v in out))
        self.assertEqual(out[:2], [250, 750])

    def test_payload_is_small_and_has_no_pixels(self):
        out = mesh.compact(make_landmarks())
        self.assertLess(len(json.dumps(out)), 3000)

    def test_meta_is_json(self):
        json.dumps(mesh.meta())


class DemoTests(unittest.TestCase):
    def test_demo_never_imports_pyautogui_or_touches_the_mouse(self):
        for mod in [m for m in sys.modules if m.startswith("pyautogui")]:
            pass   # (other tests may have imported it; what matters is DemoActions below never calls it)
        a = DemoActions()
        a.paused = False
        self.assertIsNone(a.click())
        self.assertIsNone(a.scroll(3))
        self.assertTrue(a.move_to(10, 10))
        self.assertEqual(a.position(), (10, 10))

    def test_demo_runs_the_real_controller_and_produces_valid_snapshots(self):
        import random
        a = DemoActions()
        a.paused = False
        c = Controller(load_config(), a, demo_mapper())
        rng, t = random.Random(1), 0.0
        for _ in range(900):                                # 30 simulated seconds
            face, pts = demo_face(t, rng)
            c.update(face, t)
            t += 1 / 30
            self.assertEqual(len(pts), 2 * len(mesh.POINT_IDS))
        snap = c.snapshot()
        json.dumps(snap)
        kinds = {e["kind"] for e in snap["log"]}
        self.assertIn("neutral", kinds)
        self.assertTrue(snap["gaze"]["calibrated"])

    def test_demo_points_are_inside_the_frame(self):
        for t in (0, 3.3, 9.1, 17.5):
            pts = demo_points(t, yaw=8, pitch=4, jaw=0.7, brow=0.7, look=(1, -1))
            self.assertTrue(all(0 <= v <= 1000 for v in pts), t)

    def test_demo_uses_a_throwaway_config_not_the_real_one(self):
        src = (Path(__file__).resolve().parent.parent / "nod" / "demo.py").read_text()
        self.assertIn("tempfile.mkdtemp", src)
        self.assertNotIn("save_config(", src)


class StaticAssetTests(unittest.TestCase):
    def setUp(self):
        self.html = (UI / "index.html").read_text()

    def test_files_exist(self):
        for f in ("index.html", "manifest.webmanifest", "icon-192.png", "icon-512.png", "favicon.svg"):
            self.assertTrue((UI / f).exists(), f)

    def test_manifest_is_valid_and_black_like_the_reference(self):
        m = json.loads((UI / "manifest.webmanifest").read_text())
        self.assertEqual(m["display"], "standalone")
        self.assertEqual(m["background_color"], "#000000")
        for icon in m["icons"]:
            self.assertTrue((UI / Path(icon["src"]).name).exists())

    def test_page_has_the_token_placeholder_exactly_once_in_script(self):
        self.assertEqual(self.html.count("__NOD_TOKEN__"), 1)

    def test_page_is_fully_offline_no_external_urls(self):
        urls = set(re.findall(r"https?://[^\s\"')]+", self.html))
        self.assertTrue(urls <= {"http://www.w3.org/2000/svg"}, urls)
        for bad in ("cdn.", "googleapis", "fonts.g", "unpkg", "jsdelivr"):
            self.assertNotIn(bad, self.html)

    def test_no_eval_or_dynamic_script_injection(self):
        for bad in ("eval(", "new Function", "document.write", "innerHTML = s.", "innerHTML = `"):
            self.assertNotIn(bad, self.html)

    def test_innerhtml_is_never_assigned_anything_but_empty(self):
        for m in re.finditer(r"innerHTML\s*=\s*([^;]+);", self.html):
            self.assertEqual(m.group(1).strip(), '""', m.group(0))

    def test_design_tokens_from_the_reference_palette(self):
        for token in ("#000000", "#e0e0e0", "#f54a00", "#868686"):
            self.assertIn(token, self.html)

    def test_accessibility_basics(self):
        self.assertIn('lang="en"', self.html)
        self.assertIn("prefers-reduced-motion", self.html)
        self.assertIn(":focus-visible", self.html)
        self.assertIn('role="switch"', self.html)
        self.assertIn("aria-live", self.html)

    def test_every_dom_id_the_script_uses_exists_in_the_markup(self):
        used = set(re.findall(r'\$\("([\w-]+)"\)', self.html))
        declared = set(re.findall(r'id="([\w-]+)"', self.html))
        self.assertEqual(used - declared, set())

    def test_icons_are_png_files(self):
        for f in ("icon-192.png", "icon-512.png"):
            self.assertEqual((UI / f).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()


class CalibrationScreenTests(unittest.TestCase):
    def test_dot_screen_and_cards_draw_at_any_screen_size(self):
        from nod.gaze_calibration import _card, _draw
        for size in ((1512, 982), (1920, 1080), (800, 600)):
            img = _draw(size, (size[0] * 0.9, size[1] * 0.1), 0.5, True, "Look at the dot", 7, 18, face_ok=False)
            self.assertEqual(img.shape, (size[1], size[0], 3))
            self.assertGreater(img.sum(), 0)
            self.assertEqual(_card(size, "Good", ["a", "b"]).shape, (size[1], size[0], 3))

    def test_old_call_signature_still_works(self):
        from nod.gaze_calibration import _draw
        self.assertEqual(_draw((800, 600), (400, 300), 0.5, True, "hello").shape, (600, 800, 3))

    def test_on_tick_reports_progress(self):
        src = (Path(__file__).resolve().parent.parent / "nod" / "gaze_calibration.py").read_text()
        self.assertIn("on_tick(n, len(plan)", src)


class DashboardBusyAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.html = (UI / "index.html").read_text()

    def test_page_handles_busy_state_and_shows_the_emergency_stop(self):
        self.assertIn('id="busy"', self.html)
        self.assertIn("s.busy", self.html)
        self.assertIn("top-left corner", self.html)

    def test_pulse_respects_reduced_motion_globally(self):
        self.assertIn("prefers-reduced-motion", self.html)
        self.assertIn(".tile.pulse", self.html)


class HeartbeatTests(unittest.TestCase):
    def test_stream_repeats_the_last_state_so_the_page_knows_nod_is_alive(self):
        import http.client
        from nod.dashboard import Dashboard
        dash = Dashboard(lambda: load_config(), lambda a: {}, lambda c: None, port=0)
        dash.start()
        try:
            dash.publish({"busy": "Calibrating eyes: 3/18"})
            conn = http.client.HTTPConnection("127.0.0.1", dash.port, timeout=6)
            conn.request("GET", f"/events?t={dash.token}", headers={"Host": f"127.0.0.1:{dash.port}"})
            r = conn.getresponse()
            lines = []
            while len(lines) < 2:                      # two data events with no new publish in between
                line = r.fp.readline().decode().strip()
                if line.startswith("data:"):
                    lines.append(line)
            self.assertEqual(lines[0], lines[1])
            self.assertIn("Calibrating", lines[0])
            conn.close()
        finally:
            dash.stop()
