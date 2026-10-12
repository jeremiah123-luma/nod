"""Settings schema, live reload, activity log, dashboard snapshot."""
import copy
import json
import os
import tempfile
import unittest

from nod import settings
from nod.config import load_config
from nod.controller import Controller
from tests.fakes import FakeActions, face
from tests.test_m4 import calibrated_mapper

FPS = 30


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_every_schema_key_exists_in_config_and_current_value_is_in_range(self):
        for s in settings.SCHEMA:
            value = settings.get_path(self.cfg, s["key"])
            settings.validate(s["key"], value)          # defaults must themselves be valid

    def test_schema_entries_are_complete(self):
        for s in settings.SCHEMA:
            for k in ("key", "group", "label", "kind", "help"):
                self.assertIn(k, s, s["key"])
            if s["kind"] in ("float", "int"):
                self.assertLess(s["min"], s["max"])
                self.assertGreater(s["step"], 0)

    def test_valid_change_applies_to_a_copy_not_the_original(self):
        new = settings.apply_changes(self.cfg, {"pointer.gain_x": 100})
        self.assertEqual(new["pointer"]["gain_x"], 100)
        self.assertNotEqual(self.cfg["pointer"]["gain_x"], 100)

    def test_rejects_unknown_keys_out_of_range_and_wrong_types(self):
        bad = [{"pointer.nope": 1}, {"pointer.gain_x": 9999}, {"pointer.gain_x": -5}, {"pointer.gain_x": "fast"},
               {"pointer.gain_x": True}, {"pointer.invert_x": 1}, {"scroll.toggle_gesture": "wink"},
               {"pointer.gain_x": float("nan")}, {"pointer.gain_x": float("inf")}, {"camera.width": 100},
               {"hotkeys.pause": "x"}, {}, None, "string"]
        for changes in bad:
            with self.assertRaises(ValueError, msg=str(changes)):
                settings.apply_changes(self.cfg, changes)

    def test_all_or_nothing(self):
        with self.assertRaises(ValueError):
            settings.apply_changes(self.cfg, {"pointer.gain_x": 100, "pointer.gain_y": 99999})

    def test_int_settings_are_rounded(self):
        new = settings.apply_changes(self.cfg, {"click.hold_ms": 301.7})
        self.assertEqual(new["click"]["hold_ms"], 302)
        self.assertIsInstance(new["click"]["hold_ms"], int)

    def test_save_is_atomic_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            settings.save_config(self.cfg, p)
            self.assertEqual(json.load(open(p)), self.cfg)
            self.assertEqual(os.listdir(d), ["config.json"])      # no temp files left behind

    def test_ui_schema_carries_values(self):
        by = {s["key"]: s for s in settings.schema_for_ui(self.cfg)}
        self.assertEqual(by["click.threshold"]["value"], self.cfg["click"]["threshold"])

    def test_dashboard_cannot_touch_hotkeys_camera_or_voice(self):
        keys = {s["key"] for s in settings.SCHEMA}
        self.assertFalse(any(k.startswith(("hotkeys", "camera", "voice", "ui")) for k in keys))


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)

    def test_reload_changes_behaviour_immediately(self):
        new = settings.apply_changes(self.cfg, {"click.hold_ms": 800})
        self.c.reload(new)
        for i in range(15):                                   # 0.5 s of mouth open: too short for 800 ms
            self.c.update(face(jawOpen=0.9), i / FPS)
        self.assertEqual(self.fake.clicks, 0)

    def test_reload_keeps_what_was_learned_and_current_mode(self):
        for i in range(50):
            self.c.update(face(browInnerUp=0.2), i / FPS)
        self.assertTrue(self.c.neutral.ready)
        self.c.mode = "SCROLL"
        self.c.reload(settings.apply_changes(self.cfg, {"pointer.gain_x": 120}))
        self.assertTrue(self.c.neutral.ready)
        self.assertEqual(self.c.mode, "SCROLL")
        self.assertEqual(self.c.pointer.gain_x, 120)

    def test_reload_keeps_eye_jump_calibration(self):
        c = Controller(self.cfg, self.fake, calibrated_mapper())
        c.reload(settings.apply_changes(self.cfg, {"gaze.dwell_ms": 900}))
        self.assertIsNotNone(c.gaze)
        self.assertAlmostEqual(c.gaze_jump.dwell, 0.9)

    def test_reload_can_switch_toggle_gesture(self):
        self.c.reload(settings.apply_changes(self.cfg, {"scroll.toggle_gesture": "pucker"}))
        self.assertEqual(self.c.toggle_shape, "mouthPucker")

    def test_gaze_enabled_setting_applies(self):
        c = Controller(self.cfg, self.fake, calibrated_mapper())
        c.reload(settings.apply_changes(self.cfg, {"gaze.enabled": False}))
        self.assertFalse(c.gaze_enabled)

    def test_reload_never_moves_the_mouse(self):
        self.c.reload(settings.apply_changes(self.cfg, {"pointer.gain_x": 120}))
        self.assertEqual(self.fake.moves, [])


class ActivityLogTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)
        self.t = 0.0

    def step(self, n, **blend):
        for _ in range(n):
            self.c.update(face(**blend), self.t)
            self.t += 1 / FPS

    def kinds(self):
        return [e["kind"] for e in self.c.log]

    def texts(self):
        return " | ".join(e["text"] for e in self.c.log)

    def test_neutral_learned_is_logged_once(self):
        self.step(120)
        self.assertEqual(self.kinds().count("neutral"), 1)

    def test_click_is_logged(self):
        self.step(50)
        self.step(15, jawOpen=0.9)
        self.assertIn("click", self.kinds())

    def test_mode_switch_and_pause_are_logged(self):
        self.step(50)
        self.step(18, browInnerUp=0.9)
        self.step(15)
        self.assertIn("SCROLL", self.texts())
        self.fake.paused = True
        self.step(3)
        self.fake.paused = False
        self.step(3)
        self.assertIn("Paused", self.texts())
        self.assertIn("Resumed", self.texts())

    def test_log_is_bounded_and_entries_have_time_kind_text(self):
        for _ in range(200):
            self.c._log("x", "spam")
        self.assertLessEqual(len(self.c.log), 60)
        self.assertEqual(set(self.c.log[-1]), {"t", "kind", "text"})


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions(1512, 982)

    def run_frames(self, c, n=60, **blend):
        for i in range(n):
            c.update(face(**blend), i / FPS)

    def test_snapshot_is_json_serializable_with_and_without_gaze_and_face(self):
        c = Controller(self.cfg, self.fake)
        json.dumps(c.snapshot())                                  # before any frame
        self.run_frames(c)
        json.dumps(c.snapshot())
        c2 = Controller(self.cfg, FakeActions(1000, 800), calibrated_mapper())
        self.run_frames(c2)
        json.dumps(c2.snapshot())

    def test_snapshot_fields(self):
        c = Controller(self.cfg, self.fake)
        self.run_frames(c, 60, jawOpen=0.2)
        s = c.snapshot()
        self.assertEqual(s["mode"], "POINT")
        self.assertTrue(s["face"])
        self.assertEqual(s["screen"], [1512, 982])
        self.assertEqual(s["click"]["value"], 0.2)
        self.assertTrue(s["neutral"]["ready"])
        self.assertFalse(s["gaze"]["calibrated"])
        self.assertIn("NOT CALIBRATED", s["gaze"]["status"])
        self.assertLessEqual(len(s["trace"]), 90)

    def test_snapshot_shows_paused_and_no_face(self):
        self.fake.paused = True
        c = Controller(self.cfg, self.fake)
        c.update(None, 0.0)
        s = c.snapshot()
        self.assertEqual(s["mode"], "PAUSED")
        self.assertTrue(s["paused"])
        self.assertFalse(s["face"])

    def test_snapshot_gaze_quality(self):
        c = Controller(self.cfg, FakeActions(1000, 800), calibrated_mapper())
        g = c.snapshot()["gaze"]
        self.assertTrue(g["calibrated"])
        self.assertIn("model", g)
        self.assertGreaterEqual(g["error_pct"], 0)


if __name__ == "__main__":
    unittest.main()
