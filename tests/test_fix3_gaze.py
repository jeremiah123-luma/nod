"""Fix 3: eye jump works with noisy real-world gaze, and tells you why it did or didn't fire."""
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nod import hotkeys
from nod.config import load_config
from nod.controller import Controller
from nod.face import FaceData
from nod.gaze import GazeJump, GazeMapper, calibration_targets, fit_best
from nod.gaze_calibration import signal_tag, verdict
from nod.hud import draw_hud
from tests.fakes import FakeActions, face
from tests.test_m4 import SCREEN, calibrated_mapper, eyes_for

FPS = 30


class JumpWithNoisyGazeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()["gaze"]
        self.jump = GazeJump(self.cfg, SCREEN)

    def look(self, center, sigma, seconds=1.0, cursor=(500, 400), speed=0.0, seed=1, t0=0.0, outliers=0):
        r = random.Random(seed)
        out = []
        for i in range(int(seconds * FPS)):
            g = (center[0] + r.gauss(0, sigma), center[1] + r.gauss(0, sigma))
            if outliers and i % outliers == 0:
                g = (g[0] + 400, g[1] - 300)           # a wild frame
            res = self.jump.update(g, cursor, speed, t0 + i / FPS)
            if res:
                out.append(res)
        return out

    def test_realistically_noisy_but_steady_look_jumps(self):
        out = self.look((850, 650), sigma=60)
        self.assertEqual(len(out), 1)
        self.assertLess(math.dist(out[0], (850, 650)), 60)

    def test_one_wild_frame_in_ten_does_not_ruin_it(self):
        """The old rule restarted its timer on every outlier, so it never completed."""
        self.assertEqual(len(self.look((850, 650), sigma=30, outliers=10)), 1)

    def test_truly_unsteady_gaze_does_not_jump(self):
        out = []
        for i in range(60):
            g = (600, 450) if i % 2 else (950, 750)           # flicking between two far-apart spots
            res = self.jump.update(g, (50, 50), 0.0, i / FPS)
            if res:
                out.append(res)
        self.assertEqual(out, [])
        self.assertIn("unsteady", self.jump.reason)

    def test_reasons_are_reported(self):
        j = self.jump
        j.update(None, (500, 400), 0, 0.0)
        self.assertIn("waiting", j.reason)
        j.update((520, 410), (500, 400), 0, 0.1)
        self.assertIn("near cursor", j.reason)
        j.update((900, 700), (500, 400), 400.0, 0.2)
        self.assertIn("head moving", j.reason)
        j.update((900, 700), (500, 400), 0.0, 0.3)
        self.assertIn("holding", j.reason)
        for i in range(40):
            j.update((900, 700), (500, 400), 0.0, 0.4 + i / FPS)
        self.assertIn("cooldown", j.reason)          # it jumped, now cooling down

    def test_jumped_reason_on_the_jump_frame(self):
        j, seen = self.jump, []
        for i in range(40):
            if j.update((900, 700), (500, 400), 0.0, i / FPS):
                seen.append(j.reason)
        self.assertEqual(seen, ["JUMPED"])

    def test_progress_grows_while_holding(self):
        for i in range(8):
            self.jump.update((900, 700), (500, 400), 0.0, i / FPS)
        self.assertGreater(self.jump.progress, 0.3)


class ControllerRecheckTests(unittest.TestCase):
    """Re-check after fix 1: the 'head must be still' test passes with real head noise."""

    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions(*SCREEN)
        self.c = Controller(self.cfg, self.fake, calibrated_mapper())
        self.rng = random.Random(7)
        self.t = 0.0

    def step(self, x, y, frames, head_noise=True, gaze_noise=0.004):
        for _ in range(frames):
            f = eyes_for(x, y, noise=gaze_noise, rng=self.rng)
            f.head_x = self.rng.gauss(0, 0.14) if head_noise else 0.0     # measured real noise levels
            f.head_y = self.rng.gauss(0, 0.25) if head_noise else 0.0
            self.c.update(f, self.t)
            self.t += 1 / FPS

    def test_head_counts_as_still_with_real_noise(self):
        self.step(500, 400, 90)
        self.assertLess(self.c.pointer.speed_px, self.cfg["gaze"]["max_head_speed_px_s"])

    def test_noisy_head_and_noisy_eyes_still_jump(self):
        self.step(500, 400, 45)
        self.step(900, 700, 60)
        self.assertLess(math.hypot(self.c.cursor[0] - 900, self.c.cursor[1] - 700), 200)
        self.assertEqual(self.fake.clicks, 0)

    def test_hud_says_why(self):
        self.step(500, 400, 30)
        self.assertIn("near cursor", self.c.jump_status())
        self.step(900, 700, 8)
        self.assertIn("holding", self.c.jump_status())

    def test_status_messages_for_every_non_firing_case(self):
        c = Controller(self.cfg, self.fake)
        self.assertIn("NOT CALIBRATED", c.jump_status())
        self.assertIn("ctrl+alt+k", c.jump_status())
        self.c.toggle_gaze()
        self.assertIn("OFF", self.c.jump_status())
        self.c.toggle_gaze()
        self.c.mode = "SCROLL"
        self.assertIn("SCROLL", self.c.jump_status())
        self.c.mode = "POINT"
        self.fake.paused = True
        self.assertEqual(self.c.jump_status(), "paused")

    def test_blink_is_reported(self):
        f = eyes_for(900, 700)
        f.blend = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}
        self.c.update(f, 0.0)
        self.assertIn("blink", self.c.jump_status())

    def test_hud_map_has_gaze_dot_cursor_and_flash_after_jump(self):
        self.step(500, 400, 20)
        gm = self.c.hud_info()["gaze_map"]
        self.assertIsNotNone(gm["gaze"])
        self.assertEqual(gm["screen"], SCREEN)
        flashed = False
        for _ in range(60):                                    # the flash is brief (0.8 s) - look for it as it happens
            self.step(900, 700, 1)
            flashed = flashed or self.c.hud_info()["gaze_map"]["flash"] is not None
        self.assertTrue(flashed)
        big = [m for a, m in zip([(0, 0)] + self.fake.moves, self.fake.moves) if math.dist(a, m) > 200]
        self.assertEqual(len(big), 1)                           # exactly one big jump

    def test_hud_map_absent_without_calibration(self):
        self.assertIsNone(Controller(self.cfg, self.fake).hud_info()["gaze_map"])

    def test_hud_draws_the_map(self):
        self.step(900, 700, 60)
        frame = np.zeros((480, 640, 3), np.uint8)
        draw_hud(frame, self.c.hud_info(), 30.0, True)
        self.assertGreater(frame[300:480, 400:640].sum(), 0)      # something drawn in the map corner

    def test_iris_numbers_are_on_the_hud(self):
        self.step(900, 700, 5)
        self.assertTrue(any(l.startswith("iris h") for l in self.c.hud_info()["lines"]))


class ModelSelectionTests(unittest.TestCase):
    def dots(self, head_matters, n=18, seed=3, noise=0.005):
        rng = random.Random(seed)
        targets = calibration_targets(*SCREEN) * (n // 9)
        out = []
        for x, y in targets:
            hx, hy = rng.uniform(-6, 6), rng.uniform(-6, 6)         # head wanders while you look at the dots
            gh = (x / SCREEN[0] - 0.5) * 0.30 + rng.gauss(0, noise)
            gv = (y / SCREEN[1] - 0.5) * 0.20 + rng.gauss(0, noise)
            if head_matters:                                          # eyes compensate when the head turns
                gh -= hx * 0.01
                gv -= hy * 0.01
            out.append((SimpleNamespace(gaze_h=gh, gaze_v=gv, head_x=hx, head_y=hy), (x, y)))
        return out

    def test_picks_eyes_plus_head_when_head_matters(self):
        m, report = fit_best(self.dots(True), SCREEN, 0.1)
        self.assertEqual(m.label, "eyes+head")
        self.assertEqual(len(report), 3)

    def test_prefers_simpler_model_when_head_is_just_noise(self):
        for seed in (3, 4, 5, 6, 7, 8):
            m, _ = fit_best(self.dots(False, seed=seed), SCREEN, 0.1)
            self.assertEqual(m.label, "eyes only", seed)

    def test_picks_eyes_plus_head_across_seeds(self):
        for seed in (3, 4, 5, 6):
            m, _ = fit_best(self.dots(True, seed=seed), SCREEN, 0.1)
            self.assertEqual(m.label, "eyes+head", seed)

    def test_model_setting_forces_a_choice(self):
        m, report = fit_best(self.dots(False), SCREEN, 0.1, model="eyes+head")
        self.assertEqual(m.label, "eyes+head")
        self.assertEqual(len(report), 1)

    def test_too_few_dots_gives_none(self):
        m, report = fit_best(self.dots(False)[:2], SCREEN, 0.1)
        self.assertIsNone(m)

    def test_eighteen_dots_report_a_sane_error(self):
        m, _ = fit_best(self.dots(False, noise=0.01), SCREEN, 0.1)
        self.assertLess(m.error_px, 150)


class VersioningAndVerdictTests(unittest.TestCase):
    def test_old_calibration_is_rejected_new_one_accepted(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            m = calibrated_mapper()
            m.tag = ""                                   # made before the head-signal change
            m.save(p)
            self.assertIsNone(GazeMapper.load(p, SCREEN, signal_tag(cfg)))
            m.tag = signal_tag(cfg)
            m.save(p)
            self.assertIsNotNone(GazeMapper.load(p, SCREEN, signal_tag(cfg)))

    def test_verdict_levels(self):
        diag = math.hypot(*SCREEN)
        self.assertTrue(verdict(0.05 * diag, SCREEN)[1].startswith("GOOD"))
        self.assertTrue(verdict(0.10 * diag, SCREEN)[1].startswith("OK"))
        self.assertTrue(verdict(0.17 * diag, SCREEN)[1].startswith("POOR"))     # what your first calibration was

    def test_calibrate_has_a_global_hotkey(self):
        self.assertIn("<ctrl>+<alt>+k", hotkeys.build_bindings(load_config()))

    def test_calibration_uses_two_passes_so_it_has_enough_dots(self):
        g = load_config()["gaze"]
        self.assertGreaterEqual(g["passes"] * 9, 18)


if __name__ == "__main__":
    unittest.main()
