import math
import random
import tempfile
import unittest
from pathlib import Path

from nod.config import load_config
from nod.controller import Controller
from nod.face import FaceData
from nod.gaze import (GazeJump, GazeMapper, GazeTracker, calibration_targets,
                      feature_vector, is_blinking)
from nod.gaze_calibration import average_features
from tests.fakes import FakeActions, face

SCREEN = (1000, 800)


def eyes_for(x, y, head_x=0.0, head_y=0.0, noise=0.0, rng=None):
    """Fake the eye numbers a person would produce when looking at pixel (x, y).

    Made-up but plausible rule: iris offset grows linearly across the screen.
    """
    r = rng or random
    h = (x / SCREEN[0] - 0.5) * 0.30 + r.gauss(0, noise)
    v = (y / SCREEN[1] - 0.5) * 0.20 + r.gauss(0, noise)
    return FaceData(head_x=head_x, head_y=head_y, blend={}, gaze_h=h, gaze_v=v)


def calibrated_mapper(noise=0.0, use_head=True, quadratic=False, seed=1):
    rng = random.Random(seed)
    targets = calibration_targets(*SCREEN)
    m = GazeMapper(SCREEN, use_head, quadratic, ridge=load_config()["gaze"]["ridge"])
    feats = [feature_vector(eyes_for(x, y, rng.uniform(-3, 3), rng.uniform(-3, 3), noise, rng), use_head, quadratic)
             for x, y in targets]
    m.fit(feats, targets)
    return m


class MapperTests(unittest.TestCase):
    def test_grid_has_nine_points_inside_screen_center_first(self):
        t = calibration_targets(*SCREEN)
        self.assertEqual(len(t), 9)
        self.assertEqual(t[0], (500, 400))
        self.assertTrue(all(0 < x < SCREEN[0] and 0 < y < SCREEN[1] for x, y in t))

    def test_recovers_screen_position_from_clean_eyes(self):
        m = calibrated_mapper()
        for x, y in [(100, 100), (900, 700), (500, 400), (200, 650)]:
            px, py = m.predict(eyes_for(x, y))
            self.assertLess(math.hypot(px - x, py - y), 40)

    def test_noisy_calibration_still_reasonable_and_reports_error(self):
        m = calibrated_mapper(noise=0.01)
        self.assertGreater(m.error_px, 0)
        self.assertLess(m.error_px, 250)
        px, py = m.predict(eyes_for(800, 200))
        self.assertLess(math.hypot(px - 800, py - 200), 250)

    def test_works_without_head_features_and_with_quadratic(self):
        for use_head, quad in [(False, False), (True, True)]:
            m = calibrated_mapper(use_head=use_head, quadratic=quad)
            px, py = m.predict(eyes_for(700, 300))
            self.assertLess(math.hypot(px - 700, py - 300), 60, (use_head, quad))

    def test_predictions_are_clamped_to_screen(self):
        m = calibrated_mapper()
        far = FaceData(0, 0, {}, gaze_h=5.0, gaze_v=-5.0)
        x, y = m.predict(far)
        self.assertTrue(0 <= x <= SCREEN[0] - 1 and 0 <= y <= SCREEN[1] - 1)

    def test_too_few_points_is_a_clear_error(self):
        m = GazeMapper(SCREEN)
        with self.assertRaises(ValueError):
            m.fit([feature_vector(eyes_for(1, 1), True, False)] * 3, [(1, 1)] * 3)

    def test_save_and_load_roundtrip(self):
        m = calibrated_mapper()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            m.save(p)
            m2 = GazeMapper.load(p, SCREEN)
            self.assertTrue(m2.ready)
            f = eyes_for(300, 500)
            self.assertEqual(m.predict(f), m2.predict(f))

    def test_load_rejects_missing_corrupt_or_other_screen(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            self.assertIsNone(GazeMapper.load(p, SCREEN))
            p.write_text("not json {{{")
            self.assertIsNone(GazeMapper.load(p, SCREEN))
            calibrated_mapper().save(p)
            self.assertIsNone(GazeMapper.load(p, (1920, 1080)))

    def test_median_averaging_ignores_outlier_frame(self):
        import numpy as np
        samples = [np.array([0.1, 0.1])] * 9 + [np.array([5.0, 5.0])]
        self.assertAlmostEqual(average_features(samples)[0], 0.1)

    def test_blink_detection(self):
        self.assertTrue(is_blinking(face(eyeBlinkLeft=0.9)))
        self.assertFalse(is_blinking(face(eyeBlinkLeft=0.1, eyeBlinkRight=0.2)))


class JumpTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()["gaze"]
        self.jump = GazeJump(self.cfg, SCREEN)

    def look(self, gaze, cursor=(500, 400), head_speed=0.0, seconds=1.0, t0=0.0, fps=30):
        out = []
        for i in range(int(seconds * fps)):
            r = self.jump.update(gaze, cursor, head_speed, t0 + i / fps)
            if r:
                out.append(r)
        return out

    def test_steady_far_look_jumps_once(self):
        self.assertEqual(len(self.look((900, 700), seconds=2.0)), 1)   # cooldown stops repeats

    def test_look_near_cursor_never_jumps(self):
        self.assertEqual(self.look((520, 410), seconds=3.0), [])

    def test_quick_glance_does_not_jump(self):
        self.assertEqual(self.look((900, 700), seconds=0.25), [])

    def test_wandering_gaze_does_not_jump(self):
        out = []
        for i in range(90):
            g = (200 + 300 * (i % 2), 700)    # flicking between two far spots
            r = self.jump.update(g, (500, 100), 0.0, i / 30)
            if r:
                out.append(r)
        self.assertEqual(out, [])

    def test_head_moving_blocks_jump(self):
        self.assertEqual(self.look((900, 700), head_speed=500.0, seconds=2.0), [])

    def test_lost_gaze_resets(self):
        self.look((900, 700), seconds=0.3)
        self.jump.update(None, (500, 400), 0, 0.31)
        self.assertEqual(self.look((900, 700), seconds=0.3, t0=0.4), [])

    def test_cooldown_between_jumps(self):
        first = self.look((900, 700), seconds=1.0)
        self.assertEqual(len(first), 1)
        again = self.look((100, 100), seconds=0.6, t0=1.0)   # within the 1.2s cooldown
        self.assertEqual(again, [])

    def test_jump_target_is_where_you_looked(self):
        (x, y), = self.look((850, 650), seconds=1.0)
        self.assertLess(math.hypot(x - 850, y - 650), 5)


class ControllerGazeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions(*SCREEN)
        self.mapper = calibrated_mapper()
        self.c = Controller(self.cfg, self.fake, self.mapper)
        self.t = 0.0

    def look_at(self, x, y, frames, head_x=0.0, head_y=0.0, **blend):
        f = eyes_for(x, y, head_x, head_y)
        f.blend = blend
        for _ in range(frames):
            self.c.update(f, self.t)
            self.t += 1 / 30

    def test_no_calibration_means_eye_jump_off(self):
        c = Controller(self.cfg, self.fake)
        self.assertFalse(c.gaze_enabled)
        self.assertIsNone(c.toggle_gaze())
        self.assertIn("not calibrated", " ".join(c.hud_info()["lines"]))

    def test_looking_at_far_corner_jumps_cursor_there(self):
        self.look_at(900, 700, 45)
        self.assertLess(math.hypot(self.c.cursor[0] - 900, self.c.cursor[1] - 700), 120)
        self.assertEqual(self.fake.clicks, 0)   # jump never clicks

    def test_head_can_fine_aim_after_jump(self):
        self.look_at(900, 700, 45)
        jumped = list(self.c.cursor)
        for i in range(30):                       # now turn head left; eyes stay on the target
            self.c.update(eyes_for(900, 700, head_x=-i * 0.3), self.t)
            self.t += 1 / 30
        self.assertLess(self.c.cursor[0], jumped[0] - 20)   # head moved it; gaze didn't yank it back

    def test_eyes_alone_do_not_move_cursor_when_close(self):
        self.look_at(500, 400, 90)
        self.assertEqual(self.fake.moves, [])

    def test_moving_head_blocks_jump(self):
        for i in range(60):
            self.c.update(eyes_for(900, 700, head_x=i * 2.0), self.t)   # fast head sweep
            self.t += 1 / 30
        far = math.hypot(self.c.cursor[0] - 900, self.c.cursor[1] - 700)
        self.assertGreater(far, 50)   # it was steered by the head, not teleported by the eyes

    def test_blinks_are_ignored_and_dont_break_the_look(self):
        for i in range(45):
            blink = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9} if i in (10, 11, 12, 13) else {}
            f = eyes_for(900, 700)
            f.blend = blink
            if blink:
                f.gaze_h, f.gaze_v = -9, 9      # garbage iris numbers during the blink
            self.c.update(f, self.t); self.t += 1 / 30
        self.assertLess(math.hypot(self.c.cursor[0] - 900, self.c.cursor[1] - 700), 120)

    def test_paused_and_scroll_mode_never_jump(self):
        self.fake.paused = True
        self.look_at(900, 700, 60)
        self.assertEqual(self.fake.moves, [])
        self.fake.paused = False
        self.c.mode = "SCROLL"
        self.look_at(900, 700, 60)
        self.assertEqual(self.fake.moves, [])

    def test_toggle_off_stops_jumping(self):
        self.assertFalse(self.c.toggle_gaze())
        self.look_at(900, 700, 60)
        self.assertEqual(self.fake.moves, [])
        self.assertTrue(self.c.toggle_gaze())

    def test_recalibrating_swaps_mapper(self):
        self.c.set_gaze_mapper(None)
        self.assertFalse(self.c.gaze_enabled)
        self.c.set_gaze_mapper(calibrated_mapper())
        self.assertTrue(self.c.gaze_enabled)

    def test_hud_shows_eye_jump_progress(self):
        self.look_at(900, 700, 8)
        bars = dict((b[0], b[1]) for b in self.c.hud_info()["bars"])
        self.assertGreater(bars["eye jump"], 0)


if __name__ == "__main__":
    unittest.main()
