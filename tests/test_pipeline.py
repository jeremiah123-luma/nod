"""End to end with fake landmarks: landmarks -> FaceData -> Controller -> (fake) mouse."""
import unittest

from nod.config import load_config
from nod.controller import Controller
from nod.face import analyze_landmarks
from nod.gaze_calibration import _draw
from tests.fakes import FakeActions, make_landmarks, set_iris


class PipelineTests(unittest.TestCase):
    def test_turning_head_right_moves_cursor_right(self):
        fake = FakeActions()
        c = Controller(load_config(), fake)
        for i in range(60):
            lm = make_landmarks(yaw=i * 0.003)            # slowly turn right
            c.update(analyze_landmarks(lm, {}, aspect=4 / 3), i / 30)
        self.assertGreater(c.cursor[0], 600)

    def test_jaw_blendshape_clicks_through_the_whole_chain(self):
        fake = FakeActions()
        c = Controller(load_config(), fake)
        for i in range(30):
            c.update(analyze_landmarks(make_landmarks(jaw=1.0), {"jawOpen": 0.9}, 4 / 3), i / 30)
        self.assertEqual(fake.clicks, 1)
        self.assertEqual(fake.moves, [])                  # mouth open didn't move the cursor

    def test_iris_features_have_the_right_sign(self):
        right = analyze_landmarks(set_iris(make_landmarks(), h=0.5), {})
        left = analyze_landmarks(set_iris(make_landmarks(), h=-0.5), {})
        down = analyze_landmarks(set_iris(make_landmarks(), v=0.5), {})
        self.assertGreater(right.gaze_h, 0.2)
        self.assertLess(left.gaze_h, -0.2)
        self.assertGreater(down.gaze_v, 0.2)

    def test_calibration_screen_draws(self):
        img = _draw((800, 600), (400, 300), 0.5, True, "hello")
        self.assertEqual(img.shape, (600, 800, 3))


if __name__ == "__main__":
    unittest.main()
