import unittest

from nod.config import load_config
from nod.controller import Controller
from tests.fakes import FakeActions, face


class ScrollModeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)
        self.t = 0.0

    def step(self, frames, head_x=0.0, head_y=0.0, **blend):
        for _ in range(frames):
            self.c.update(face(head_x, head_y, **blend), self.t)
            self.t += 1 / 30

    def brow_raise(self, head_x=0.0, head_y=0.0):
        self.step(15, head_x, head_y, browInnerUp=0.9)   # 0.5s held
        self.step(15, head_x, head_y, browInnerUp=0.0)

    def enter_scroll(self, head_y=0.0):
        self.step(5, head_y=head_y)
        self.brow_raise(head_y=head_y)
        self.assertEqual(self.c.mode, "SCROLL")
        self.step(20, head_y=head_y)     # let it settle

    def test_eyebrow_raise_toggles_mode_both_ways(self):
        self.brow_raise()
        self.assertEqual(self.c.mode, "SCROLL")
        self.step(30)                    # wait out the cooldown
        self.brow_raise()
        self.assertEqual(self.c.mode, "POINT")

    def test_quick_eyebrow_twitch_does_not_toggle(self):
        self.step(3, browInnerUp=0.9)
        self.step(10)
        self.assertEqual(self.c.mode, "POINT")

    def test_holding_brows_up_toggles_only_once(self):
        self.step(120, browInnerUp=0.9)
        self.assertEqual(self.c.mode, "SCROLL")

    def test_head_down_scrolls_down_and_up_scrolls_up(self):
        self.enter_scroll()
        self.step(60, head_y=12.0)
        down = sum(self.fake.scrolls)
        self.assertLess(down, 0)         # negative = down
        self.fake.scrolls.clear()
        self.step(60, head_y=-12.0)
        self.assertGreater(sum(self.fake.scrolls), 0)

    def test_deadzone_no_scroll_when_level(self):
        self.enter_scroll()
        self.step(90, head_y=2.0)        # inside the 4-unit deadzone
        self.assertEqual(self.fake.scrolls, [])

    def test_further_tilt_scrolls_faster(self):
        self.enter_scroll()
        self.step(60, head_y=8.0)
        slow = abs(sum(self.fake.scrolls))
        self.fake.scrolls.clear()
        self.step(60, head_y=16.0)
        self.assertGreater(abs(sum(self.fake.scrolls)), slow)

    def test_speed_is_capped(self):
        self.enter_scroll()
        self.step(30, head_y=500.0)
        self.assertLessEqual(abs(sum(self.fake.scrolls)), 31)

    def test_cursor_frozen_and_clicks_off_in_scroll_mode(self):
        self.enter_scroll()
        moves_before = len(self.fake.moves)
        self.step(40, head_x=15.0, head_y=0.0, jawOpen=0.9)
        self.assertEqual(len(self.fake.moves), moves_before)
        self.assertEqual(self.fake.clicks, 0)

    def test_neutral_is_where_head_was_on_entry_not_zero(self):
        self.enter_scroll(head_y=60.0)   # nose sits ~60 below eyes normally
        self.step(60, head_y=60.0)
        self.assertEqual(self.fake.scrolls, [])

    def test_no_cursor_jump_after_returning_to_point_mode(self):
        self.step(20)
        self.enter_scroll()
        self.step(30, head_x=25.0, head_y=0.0)   # head turned a lot while scrolling
        self.step(30, head_x=25.0)
        self.brow_raise(head_x=25.0)
        self.assertEqual(self.c.mode, "POINT")
        before = list(self.c.cursor)
        self.step(3, head_x=25.0)
        self.assertAlmostEqual(self.c.cursor[0], before[0], delta=1.0)

    def test_paused_never_scrolls(self):
        self.enter_scroll()
        self.fake.paused = True
        self.step(60, head_y=20.0)
        self.assertEqual(self.fake.scrolls, [])

    def test_invert_flips_direction(self):
        self.cfg["scroll"]["invert"] = True
        self.c = Controller(self.cfg, self.fake)
        self.enter_scroll()
        self.step(60, head_y=12.0)
        self.assertGreater(sum(self.fake.scrolls), 0)

    def test_hud_shows_scroll_mode(self):
        self.enter_scroll()
        self.assertEqual(self.c.hud_info()["mode"], "SCROLL")


if __name__ == "__main__":
    unittest.main()
