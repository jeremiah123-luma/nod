import unittest

import numpy as np

from nod.hud import draw_hud, draw_mesh
from tests.fakes import make_landmarks


class HudTests(unittest.TestCase):
    def test_draws_without_crashing(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        info = {"mode": "POINT", "lines": ["head x +1.0 y +2.0"], "bars": [("click", 0.5, (0, 255, 0))]}
        draw_hud(frame, info, 30.0, True)
        self.assertGreater(frame.sum(), 0)

    def test_every_mode_has_a_color(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        for mode in ("POINT", "SCROLL", "PAUSED", "SOMETHING_ELSE"):
            draw_hud(frame, {"mode": mode, "lines": [], "bars": []}, 20.0, False)

    def test_mesh_draws_with_fake_landmarks(self):
        draw_mesh(np.zeros((480, 640, 3), np.uint8), make_landmarks())


if __name__ == "__main__":
    unittest.main()
