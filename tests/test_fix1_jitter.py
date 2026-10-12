"""Fix 1: the pointer must not jitter, and must not lag."""
import math
import random
import unittest
from types import SimpleNamespace

import numpy as np

from nod.config import load_config
from nod.face import FaceTracker, matrix_head_signals
from nod.filters import OneEuroFilter
from nod.pointer import HeadPointer
from tests.fakes import FakeActions, face, make_landmarks
from nod.controller import Controller

FPS = 30
# Measured on a real webcam at 30 fps with the matrix signal (std dev, in head units):
NOISE_X, NOISE_Y = 0.14, 0.25


def rot_matrix(yaw=0.0, pitch=0.0, scale=1.0):
    """4x4 like MediaPipe's: camera space has x right, y UP, face looks along +z."""
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    m = np.eye(4)
    m[:3, :3] = scale * (ry @ rx)
    return m


def drive(pointer, fn, seconds, k=1.0, seed=1):
    """Feed head_x/head_y = fn(t) + measured noise. Returns the cursor path in px."""
    r = random.Random(seed)
    x = y = 0.0
    path = []
    for i in range(int(seconds * FPS)):
        bx, by = fn(i / FPS)
        dx, dy, _ = pointer.update(bx + r.gauss(0, NOISE_X * k), by + r.gauss(0, NOISE_Y * k), i / FPS)
        x += dx
        y += dy
        path.append((x, y))
    return path


class MatrixSignalTests(unittest.TestCase):
    def test_turning_right_is_positive_x_and_left_negative(self):
        self.assertGreater(matrix_head_signals(rot_matrix(yaw=0.3))[0], 5)
        self.assertLess(matrix_head_signals(rot_matrix(yaw=-0.3))[0], -5)

    def test_tilting_down_is_positive_y_and_up_negative(self):
        self.assertGreater(matrix_head_signals(rot_matrix(pitch=0.3))[1], 5)
        self.assertLess(matrix_head_signals(rot_matrix(pitch=-0.3))[1], -5)

    def test_straight_ahead_is_zero(self):
        x, y = matrix_head_signals(rot_matrix())
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 0, places=6)

    def test_matrix_scale_and_translation_do_not_matter(self):
        a = matrix_head_signals(rot_matrix(yaw=0.2, pitch=0.1))
        m = rot_matrix(yaw=0.2, pitch=0.1, scale=7.5)
        m[:3, 3] = [10, -20, -50]
        b = matrix_head_signals(m)
        self.assertAlmostEqual(a[0], b[0], places=6)
        self.assertAlmostEqual(a[1], b[1], places=6)

    def test_broken_matrix_returns_none(self):
        self.assertIsNone(matrix_head_signals(np.zeros((4, 4))))

    def _tracker(self, source):
        t = FaceTracker.__new__(FaceTracker)    # skip loading the model file
        t.source, t.units_per_sin = source, 45.0
        return t

    def test_tracker_uses_matrix_by_default_and_landmarks_when_asked(self):
        result = SimpleNamespace(face_landmarks=[make_landmarks(yaw=0.0)], face_blendshapes=[],
                                 facial_transformation_matrixes=[rot_matrix(yaw=0.3)])
        self.assertGreater(self._tracker("matrix").analyze(result, 1.0).head_x, 5)
        self.assertAlmostEqual(self._tracker("landmarks").analyze(result, 1.0).head_x, 0, places=3)

    def test_tracker_falls_back_to_landmarks_without_matrix(self):
        result = SimpleNamespace(face_landmarks=[make_landmarks(yaw=0.15)], face_blendshapes=[],
                                 facial_transformation_matrixes=[])
        self.assertGreater(self._tracker("matrix").analyze(result, 1.0).head_x, 5)

    def test_averaged_nose_points_give_same_answer_as_single_point_on_clean_data(self):
        from nod.face import compute_head_signals
        self.assertGreater(compute_head_signals(make_landmarks(yaw=0.2))[0], 10)


class FilterTests(unittest.TestCase):
    def test_speed_floor_keeps_heavy_smoothing_on_noise(self):
        r = random.Random(5)
        noise = [r.gauss(0, 0.25) for _ in range(300)]
        open_filter = OneEuroFilter(0.3, 0.3, 0.8, speed_floor=0.0)
        floored = OneEuroFilter(0.3, 0.3, 0.8, speed_floor=3.0)
        a = [open_filter(v, i / FPS) for i, v in enumerate(noise)]
        b = [floored(v, i / FPS) for i, v in enumerate(noise)]
        self.assertLess(np.std(b[60:]), np.std(a[60:]))


class JitterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_camera_asks_for_30_fps(self):
        self.assertEqual(self.cfg["camera"]["fps"], 30)

    def test_still_head_with_real_noise_barely_moves_the_cursor(self):
        path = drive(HeadPointer(self.cfg), lambda t: (0, 0), 10)
        travel = sum(math.dist(path[i], path[i - 1]) for i in range(1, len(path)))
        drift = max(math.hypot(*p) for p in path)
        self.assertLess(drift, 40, f"cursor wandered {drift:.0f}px while the head was still")
        self.assertLess(travel, 300, f"cursor traveled {travel:.0f}px while the head was still")

    def test_still_head_is_locked(self):
        p = HeadPointer(self.cfg)
        drive(p, lambda t: (0, 0), 5)
        self.assertEqual(p.state, "LOCKED")

    def test_tiny_wobble_inside_deadzone_moves_nothing(self):
        p = HeadPointer(self.cfg)
        p.update(0.0, 0.0, 0.0)
        for i in range(1, 60):   # +-0.05 units * 70 = +-3.5 px, well inside the deadzone
            self.assertEqual(p.update(0.05 * (-1) ** i, 0.0, i / FPS)[:2], (0.0, 0.0))

    def test_deliberate_move_releases_lock_then_relocks_when_it_stops(self):
        p = HeadPointer(self.cfg)
        drive(p, lambda t: (0, 0), 2)
        self.assertEqual(p.state, "LOCKED")
        r = random.Random(3)
        states = []
        for i in range(60, 160):
            t = i / FPS
            hx = 0 if t < 2.2 else min(6, 6 * (t - 2.2) / 0.4)
            p.update(hx + r.gauss(0, NOISE_X), r.gauss(0, NOISE_Y), t)
            states.append(p.state)
        self.assertIn("MOVING", states)
        self.assertEqual(states[-1], "LOCKED")   # relocked after stopping

    def test_fast_move_follows_with_about_one_frame_of_lag(self):
        p = HeadPointer(self.cfg)
        path = drive(p, lambda t: (0 if t < 1 else min(12, 12 * (t - 1) / 0.4), 0), 3)
        final = path[-1][0]
        self.assertGreater(final, 12 * 70 * 0.9)           # it actually gets there
        t50 = next(i / FPS for i, q in enumerate(path) if q[0] >= 0.5 * final and i / FPS >= 1)
        self.assertLess((t50 - 1.2) * 1000, 120)            # within ~3 frames of the ideal halfway moment

    def test_slow_move_does_not_stall(self):
        path = drive(HeadPointer(self.cfg), lambda t: (t, 0), 3)   # 1 unit/s = 70 px/s
        self.assertGreater(path[-1][0], 0.7 * 3 * 70)

    def test_hold_on_a_target_after_moving(self):
        """The M1 'done when': after moving, holding still keeps the cursor put."""
        p = HeadPointer(self.cfg)
        path = drive(p, lambda t: (0 if t < 1 else min(5, 5 * (t - 1) / 0.5), 0), 8)
        settled = path[int(2.5 * FPS):]
        spread = max(q[0] for q in settled) - min(q[0] for q in settled)
        self.assertLess(spread, 25)

    def test_hud_gets_raw_and_smoothed_numbers(self):
        c = Controller(self.cfg, FakeActions())
        for i in range(10):
            c.update(face(head_x=i * 0.1), i / FPS)
        info = c.hud_info()
        self.assertTrue(any("RAW" in line for line in info["lines"]))
        self.assertTrue(any("SMOOTH" in line for line in info["lines"]))
        self.assertGreater(len(info["trace"]), 5)

    def test_hud_numbers_update_even_while_paused_but_mouse_never_moves(self):
        fake = FakeActions()
        fake.paused = True
        c = Controller(self.cfg, fake)
        for i in range(30):
            c.update(face(head_x=i * 0.5), i / FPS)
        self.assertGreater(len(c.trace), 10)
        self.assertEqual(fake.moves, [])


if __name__ == "__main__":
    unittest.main()
