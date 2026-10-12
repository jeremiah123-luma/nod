import unittest

from nod import actions
from nod.config import load_config
from nod.controller import Controller
from nod.face import analyze_landmarks, compute_head_signals
from nod.filters import OneEuroFilter
from nod.pointer import HeadPointer
from tests.fakes import FakeActions, face, make_landmarks


class FilterTests(unittest.TestCase):
    def test_first_value_passes_through(self):
        self.assertEqual(OneEuroFilter()(5.0, 0.0), 5.0)

    def test_smooths_jitter_when_still(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        outs = [f(10 + (0.5 if i % 2 else -0.5), i / 30) for i in range(60)]
        self.assertLess(max(outs[30:]) - min(outs[30:]), 0.3)  # raw jitter is 1.0 wide

    def test_follows_fast_motion_with_beta(self):
        slow = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        fast = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        for i in range(30):
            a, b = slow(i * 5.0, i / 30), fast(i * 5.0, i / 30)
        self.assertGreater(b, a)  # beta reduces lag on a fast ramp


class HeadSignalTests(unittest.TestCase):
    def test_turning_right_increases_head_x(self):
        base = compute_head_signals(make_landmarks())
        right = compute_head_signals(make_landmarks(yaw=0.15))
        left = compute_head_signals(make_landmarks(yaw=-0.15))
        self.assertGreater(right[0], base[0])
        self.assertLess(left[0], base[0])

    def test_tilting_down_increases_head_y(self):
        base = compute_head_signals(make_landmarks())
        down = compute_head_signals(make_landmarks(pitch=0.15))
        self.assertGreater(down[1], base[1])

    def test_opening_mouth_does_not_move_pointer_signals(self):
        self.assertEqual(compute_head_signals(make_landmarks()), compute_head_signals(make_landmarks(jaw=1.0)))

    def test_head_roll_does_not_change_signals_much(self):
        a = compute_head_signals(make_landmarks())
        b = compute_head_signals(make_landmarks(roll_deg=15))
        self.assertAlmostEqual(a[0], b[0], delta=1.0)
        self.assertAlmostEqual(a[1], b[1], delta=1.0)

    def test_distance_from_camera_does_not_matter(self):
        # Same pose, but the face is twice as small in the image.
        near = make_landmarks(yaw=0.1)
        far = make_landmarks(yaw=0.1)
        for p in far:
            p.x, p.y = 0.5 + (p.x - 0.5) / 2, 0.5 + (p.y - 0.5) / 2
        self.assertAlmostEqual(compute_head_signals(near)[0], compute_head_signals(far)[0], places=6)

    def test_analyze_landmarks_without_iris_points(self):
        data = analyze_landmarks(make_landmarks()[:400], {"jawOpen": 0.2})
        self.assertEqual(data.blend["jawOpen"], 0.2)


class PointerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg["pointer"].update(acceleration=0.0, deadzone_px=0.0)

    def run_ramp(self, pointer, start, step, frames=30):
        total = [0.0, 0.0]
        for i in range(frames):
            dx, dy, _ = pointer.update(start + i * step, 0.0, i / 30)
            total[0] += dx
            total[1] += dy
        return total

    def test_first_frame_does_not_move(self):
        self.assertEqual(HeadPointer(self.cfg).update(50.0, 50.0, 0.0), (0.0, 0.0, 0.0))

    def test_turning_head_moves_cursor_right(self):
        self.assertGreater(self.run_ramp(HeadPointer(self.cfg), 0, 0.3)[0], 50)

    def test_turning_left_moves_cursor_left(self):
        self.assertLess(self.run_ramp(HeadPointer(self.cfg), 0, -0.3)[0], -50)

    def test_invert_x_flips_direction(self):
        self.cfg["pointer"]["invert_x"] = True
        self.assertLess(self.run_ramp(HeadPointer(self.cfg), 0, 0.3)[0], -50)

    def test_holding_still_does_not_drift(self):
        p = HeadPointer(self.cfg)
        total = self.run_ramp(p, 10.0, 0.0, frames=60)
        self.assertLess(abs(total[0]) + abs(total[1]), 1.0)

    def test_deadzone_ignores_tiny_wiggle_but_keeps_slow_drift(self):
        self.cfg["pointer"]["deadzone_px"] = 2.0
        p = HeadPointer(self.cfg)
        total = self.run_ramp(p, 0, 0.02, frames=120)  # very slow, steady turn
        self.assertGreater(total[0], 10)

    def test_reset_prevents_jump(self):
        p = HeadPointer(self.cfg)
        p.update(0.0, 0.0, 0.0)
        p.reset()
        self.assertEqual(p.update(40.0, 40.0, 1.0), (0.0, 0.0, 0.0))


class ControllerPointerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fake = FakeActions()
        self.c = Controller(self.cfg, self.fake)

    def feed(self, head_x, frames=40, t0=0.0):
        for i in range(frames):
            self.c.update(face(head_x=head_x * i / frames), t0 + i / 30)

    def test_cursor_moves_with_head(self):
        self.feed(20.0)
        self.assertGreater(self.c.cursor[0], 600)

    def test_cursor_never_leaves_screen_or_hits_failsafe_corner(self):
        self.feed(-500.0)
        self.assertGreaterEqual(self.c.cursor[0], 1)
        self.assertGreaterEqual(min(m[0] for m in self.fake.moves), 1)
        self.feed(500.0, t0=10.0)
        self.assertLessEqual(self.c.cursor[0], self.fake.w - 2)

    def test_paused_means_no_mouse_actions(self):
        self.fake.paused = True
        self.feed(20.0)
        self.c.recenter()
        self.assertEqual(self.fake.moves, [])
        self.assertEqual(self.c.display_mode, "PAUSED")

    def test_resume_starts_from_real_mouse_position_without_jump(self):
        self.fake.paused = True
        self.feed(10.0)
        self.fake.pos = (123, 456)
        self.fake.paused = False
        self.c.update(face(head_x=10.0), 5.0)
        self.assertEqual(self.c.cursor, [123, 456])

    def test_no_face_does_nothing_and_no_jump_on_return(self):
        self.c.update(face(head_x=0.0), 0.0)
        self.c.update(None, 0.1)
        self.c.update(face(head_x=30.0), 0.2)
        self.assertEqual(self.fake.moves, [])

    def test_recenter(self):
        self.feed(20.0)
        self.c.recenter()
        self.assertEqual(self.fake.moves[-1], (500, 400))


class RealActionsPauseTests(unittest.TestCase):
    def test_paused_blocks_everything_without_touching_pyautogui(self):
        def boom():
            raise AssertionError("pyautogui must not be used while paused")
        old = actions._pg
        actions._pg = boom
        actions.set_paused(True)
        try:
            self.assertFalse(actions.move_to(5, 5))
            self.assertFalse(actions.click())
            self.assertFalse(actions.scroll(3))
        finally:
            actions._pg = old
            actions.set_paused(False)

    def test_toggle_pause(self):
        actions.set_paused(False)
        self.assertTrue(actions.toggle_pause())
        self.assertFalse(actions.toggle_pause())

    def test_dry_run_never_touches_pyautogui(self):
        old = actions._pg
        actions._pg = lambda: (_ for _ in ()).throw(AssertionError("no pyautogui in dry run"))
        actions.set_dry_run(True)
        try:
            self.assertTrue(actions.move_to(1, 1))
            self.assertTrue(actions.click())
        finally:
            actions._pg = old
            actions.set_dry_run(False)


if __name__ == "__main__":
    unittest.main()
