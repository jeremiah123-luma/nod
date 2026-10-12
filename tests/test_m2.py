import unittest

from nod.config import load_config
from nod.controller import Controller
from nod.gestures import DwellClick, HoldTrigger
from tests.fakes import FakeActions, face


def run(trigger, values, fps=30, t0=0.0):
    """Feed values at `fps`; return list of times when it fired."""
    return [round(t0 + i / fps, 3) for i, v in enumerate(values) if trigger.update(v, t0 + i / fps)]


class HoldTriggerTests(unittest.TestCase):
    def make(self):
        return HoldTrigger(threshold=0.5, release_threshold=0.25, hold_ms=250, cooldown_ms=600)

    def test_fires_once_after_hold_time(self):
        fires = run(self.make(), [0.9] * 30)
        self.assertEqual(len(fires), 1)
        self.assertGreaterEqual(fires[0], 0.25)

    def test_quick_blip_does_not_fire(self):
        self.assertEqual(run(self.make(), [0.9] * 4 + [0.0] * 30), [])

    def test_holding_open_does_not_repeat_even_after_cooldown(self):
        self.assertEqual(len(run(self.make(), [0.9] * 150)), 1)  # 5 seconds of yawning

    def test_ten_deliberate_clicks_in_a_row(self):
        t = self.make()
        fires, now = 0, 0.0
        for _ in range(10):
            for _ in range(12):   # open ~0.4s
                fires += t.update(0.9, now); now += 1 / 30
            for _ in range(24):   # close ~0.8s
                fires += t.update(0.0, now); now += 1 / 30
        self.assertEqual(fires, 10)

    def test_cooldown_blocks_fast_second_click(self):
        t = self.make()
        fires, now = 0, 0.0
        for _ in range(3):        # open 0.4s, close 0.1s, repeat: faster than 600ms cooldown
            for _ in range(12):
                fires += t.update(0.9, now); now += 1 / 30
            for _ in range(3):
                fires += t.update(0.0, now); now += 1 / 30
        self.assertLess(fires, 3)

    def test_flicker_between_thresholds_keeps_progress(self):
        t = self.make()
        values = [0.9, 0.9, 0.4, 0.9, 0.4, 0.9, 0.9, 0.9, 0.9, 0.9]  # dips to 0.4 (> release 0.25)
        self.assertEqual(len(run(t, values)), 1)

    def test_progress_for_hud(self):
        t = self.make()
        t.update(0.9, 0.0)
        t.update(0.9, 0.125)
        self.assertAlmostEqual(t.progress, 0.5, places=2)
        t.update(0.0, 0.2)
        self.assertEqual(t.progress, 0.0)

    def test_talking_pattern_rarely_fires(self):
        # Talking: jaw flaps 0.1-0.6 fast (~4Hz). Peaks are shorter than the hold time.
        import math
        values = [0.35 + 0.3 * math.sin(2 * math.pi * 4 * i / 30) for i in range(300)]
        self.assertEqual(run(self.make(), values), [])


class DwellTests(unittest.TestCase):
    def test_clicks_after_holding_still(self):
        d = DwellClick(dwell_ms=600, radius_px=12)
        fired = [d.update((100, 100), i / 30) for i in range(40)]
        self.assertEqual(sum(fired), 1)

    def test_moving_resets_and_no_repeat_while_resting(self):
        d = DwellClick(dwell_ms=600, radius_px=12)
        fired = 0
        for i in range(120):
            fired += d.update((100, 100), i / 30)
        self.assertEqual(fired, 1)
        for i in range(120, 160):
            fired += d.update((300, 300), i / 30)
        self.assertEqual(fired, 2)

    def test_jitter_inside_radius_still_counts_as_still(self):
        d = DwellClick(dwell_ms=600, radius_px=12)
        fired = sum(d.update((100 + (i % 3), 100 - (i % 2)), i / 30) for i in range(40))
        self.assertEqual(fired, 1)


class ControllerClickTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)

    def test_mouth_open_clicks_once(self):
        for i in range(60):
            self.c.update(face(jawOpen=0.9), i / 30)
        self.assertEqual(self.fake.clicks, 1)

    def test_opening_mouth_does_not_move_cursor(self):
        for i in range(30):
            self.c.update(face(jawOpen=0.9), i / 30)
        self.assertEqual(self.fake.moves, [])

    def test_paused_never_clicks(self):
        self.fake.paused = True
        for i in range(60):
            self.c.update(face(jawOpen=0.9), i / 30)
        self.assertEqual(self.fake.clicks, 0)

    def test_losing_face_cancels_a_click_in_progress(self):
        for i in range(5):
            self.c.update(face(jawOpen=0.9), i / 30)
        self.c.update(None, 5 / 30)
        for i in range(6, 12):
            self.c.update(face(jawOpen=0.0), i / 30)
        self.assertEqual(self.fake.clicks, 0)

    def test_dwell_toggle_and_click(self):
        self.assertFalse(self.c.dwell_enabled)
        for i in range(60):
            self.c.update(face(), i / 30)
        self.assertEqual(self.fake.clicks, 0)      # dwell is off by default
        self.assertTrue(self.c.toggle_dwell())
        for i in range(60, 160):
            self.c.update(face(), i / 30)
        self.assertEqual(self.fake.clicks, 1)

    def test_hud_shows_click_progress(self):
        for i in range(4):
            self.c.update(face(jawOpen=0.9), i / 30)
        bars = dict((b[0], b[1]) for b in self.c.hud_info()["bars"])
        self.assertGreater(bars["click"], 0)


if __name__ == "__main__":
    unittest.main()
