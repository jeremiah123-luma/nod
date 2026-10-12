"""Fix 2: tilting the head up must NOT toggle scroll mode; the toggle is relative to YOUR relaxed face."""
import unittest

from nod.config import load_config
from nod.controller import Controller
from tests.fakes import FakeActions, face


class Base(unittest.TestCase):
    def make(self, **scroll_overrides):
        self.cfg = load_config()
        self.cfg["scroll"].update(scroll_overrides)
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)
        self.t = 0.0

    def setUp(self):
        self.make()

    def step(self, frames, head_x=0.0, head_y=0.0, **blend):
        for _ in range(frames):
            self.c.update(face(head_x, head_y, **blend), self.t)
            self.t += 1 / 30

    def relax(self, head_y=0.0, **blend):
        """Startup / recenter: hold a relaxed face for the neutral capture."""
        self.c.neutral.restart()
        self.step(50, 0.0, head_y, **blend)
        self.assertTrue(self.c.neutral.ready)

    def raise_gesture(self, shape="browInnerUp", value=0.9, head_y=0.0, seconds=0.6):
        self.step(int(seconds * 30), 0.0, head_y, **{shape: value})
        self.step(15, 0.0, head_y)


class TiltUpBugTests(Base):
    def test_tilting_head_up_with_lifted_brows_does_not_toggle(self):
        """The real bug: looking up lifts your brows too."""
        self.relax()
        for _ in range(3):                           # tilt up repeatedly, brows riding along
            self.step(60, head_y=-15.0, browInnerUp=0.6)
            self.step(30, head_y=0.0)
        self.assertEqual(self.c.mode, "POINT")

    def test_blocked_reason_is_shown_on_the_hud(self):
        self.relax()
        self.step(60, head_y=-15.0, browInnerUp=0.6)
        self.assertIn("pitch", self.c.toggle_note)
        self.assertTrue(any("scroll toggle: blocked" in line for line in self.c.hud_info()["lines"]))

    def test_coming_back_level_with_brows_still_up_needs_a_calm_moment_first(self):
        self.relax()
        self.step(60, head_y=-15.0, browInnerUp=0.6)
        self.step(6, head_y=0.0, browInnerUp=0.6)    # back level, brows still lifted for a moment...
        self.step(20, head_y=0.0)                    # ...then they relax
        self.assertEqual(self.c.mode, "POINT")

    def test_deliberate_raise_with_level_still_head_still_works(self):
        self.relax()
        self.raise_gesture()
        self.assertEqual(self.c.mode, "SCROLL")

    def test_fast_head_movement_blocks_toggle(self):
        self.relax()
        for i in range(60):                          # sweeping head left/right with brows up
            self.c.update(face(head_x=(i % 10) * 6.0, head_y=0.0, browInnerUp=0.9), self.t)
            self.t += 1 / 30
        self.assertEqual(self.c.mode, "POINT")
        self.assertIn("moving", self.c.toggle_note)

    def test_holding_brows_up_while_head_slowly_drifts_out_of_level_stops(self):
        self.relax()
        self.step(5, 0.0, 0.0, browInnerUp=0.9)      # brows start up...
        self.step(80, 0.0, -20.0, browInnerUp=0.9)   # ...and the head tips back
        self.assertEqual(self.c.mode, "POINT")


class RelativeToNeutralTests(Base):
    def test_person_whose_brows_rest_high_does_not_trigger_constantly(self):
        """A fixed 0.5 threshold would fire right away for this person."""
        self.relax(browInnerUp=0.55)
        self.step(120, browInnerUp=0.55)
        self.assertEqual(self.c.mode, "POINT")
        self.assertAlmostEqual(self.c.neutral.shape["browInnerUp"], 0.55, places=2)

    def test_but_a_real_raise_from_that_baseline_does_trigger(self):
        self.relax(browInnerUp=0.55)
        self.raise_gesture(value=0.95)
        self.assertEqual(self.c.mode, "SCROLL")

    def test_small_rise_below_delta_does_not_trigger(self):
        self.relax(browInnerUp=0.10)
        self.step(120, browInnerUp=0.30)             # +0.20 < +0.30 needed
        self.assertEqual(self.c.mode, "POINT")

    def test_hud_shows_relaxed_value_and_rise(self):
        self.relax(browInnerUp=0.2)
        self.step(10, browInnerUp=0.5)
        line = next(l for l in self.c.hud_info()["lines"] if "relaxed" in l)
        self.assertIn("0.20", line)
        self.assertIn("+0.30", line)


class CalibrationTests(Base):
    def test_toggle_disabled_until_neutral_is_captured(self):
        self.step(20, browInnerUp=0.9)               # 0.67 s: capture still running
        self.assertEqual(self.c.mode, "POINT")
        self.assertIn("calibrating", self.c.toggle_note)

    def test_recenter_recaptures_neutral(self):
        self.relax()
        self.c.recenter()
        self.assertFalse(self.c.neutral.ready)
        self.step(50)
        self.assertTrue(self.c.neutral.ready)

    def test_neutral_pitch_follows_the_recalibrated_posture(self):
        self.relax(head_y=40.0)                      # you slouch: nose sits lower in the frame
        self.step(60, head_y=40.0)
        self.raise_gesture(head_y=40.0)
        self.assertEqual(self.c.mode, "SCROLL")

    def test_losing_the_face_mid_capture_restarts_it(self):
        self.step(20)
        self.c.update(None, self.t)
        self.t += 1 / 30
        self.assertEqual(self.c.neutral.progress, 0.0)

    def test_capture_runs_even_while_paused(self):
        self.fake.paused = True
        self.step(50)
        self.assertTrue(self.c.neutral.ready)
        self.assertEqual(self.fake.moves, [])


class GestureSettingTests(Base):
    def test_default_is_brows_and_both_options_exist(self):
        cfg = load_config()["scroll"]
        self.assertEqual(cfg["toggle_gesture"], "brows")
        self.assertEqual(cfg["gestures"]["pucker"]["blendshape"], "mouthPucker")

    def test_pucker_option_toggles_on_pucker_and_ignores_brows(self):
        self.make(toggle_gesture="pucker")
        self.relax()
        self.raise_gesture("browInnerUp", 0.95)
        self.assertEqual(self.c.mode, "POINT")
        self.raise_gesture("mouthPucker", 0.9)
        self.assertEqual(self.c.mode, "SCROLL")

    def test_pucker_is_also_gated_by_head_pose(self):
        self.make(toggle_gesture="pucker")
        self.relax()
        self.step(80, head_y=-15.0, mouthPucker=0.9)
        self.assertEqual(self.c.mode, "POINT")

    def test_unknown_gesture_name_is_a_clear_error(self):
        with self.assertRaises(ValueError) as e:
            self.make(toggle_gesture="wink")
        self.assertIn("brows", str(e.exception))


class LeavingScrollModeTests(Base):
    def test_exit_needs_level_head_but_scrolling_still_works_meanwhile(self):
        self.relax()
        self.raise_gesture()
        self.assertEqual(self.c.mode, "SCROLL")
        self.step(30)                                # settle
        self.step(60, head_y=12.0, browInnerUp=0.9)  # tilted to scroll, brows up: must NOT exit
        self.assertEqual(self.c.mode, "SCROLL")
        self.assertLess(sum(self.fake.scrolls), 0)   # and it scrolled down
        self.step(60, head_y=0.0)                    # level again
        self.raise_gesture()
        self.assertEqual(self.c.mode, "POINT")


if __name__ == "__main__":
    unittest.main()
