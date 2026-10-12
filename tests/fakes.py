"""Fake stand-ins so tests never need a camera, a screen, or a real mouse."""
import math
from types import SimpleNamespace

from nod.face import NOSE_POINTS, FaceData


def make_landmarks(yaw=0.0, pitch=0.0, roll_deg=0.0, jaw=0.0):
    """478 fake landmarks for a face looking at the camera, then nudged.

    yaw/pitch shift the nose tip relative to the eyes (in eye-distance fractions),
    roll_deg rotates the whole face in the image, jaw moves the chin (must NOT change head signals).
    """
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    eye_half = 0.1
    lm[33] = SimpleNamespace(x=0.5 - eye_half, y=0.4, z=0.0)
    lm[263] = SimpleNamespace(x=0.5 + eye_half, y=0.4, z=0.0)
    lm[133] = SimpleNamespace(x=lm[33].x, y=0.4, z=0.0)     # inner corners: same spot, so eye centers = outer corners
    lm[362] = SimpleNamespace(x=lm[263].x, y=0.4, z=0.0)
    eye_dist = 2 * eye_half
    nose = SimpleNamespace(x=0.5 + yaw * eye_dist, y=0.4 + (0.6 + pitch) * eye_dist, z=0.0)
    for i in NOSE_POINTS:                                    # every nose point sits at the nose position
        lm[i] = SimpleNamespace(x=nose.x, y=nose.y, z=0.0)
    lm[152] = SimpleNamespace(x=0.5, y=0.75 + jaw * 0.1, z=0.0)  # chin
    if roll_deg:
        r = math.radians(roll_deg)
        for i in (33, 133, 362, 263, 152, *NOSE_POINTS):
            p = lm[i]
            dx, dy = p.x - 0.5, p.y - 0.5
            lm[i] = SimpleNamespace(x=0.5 + dx * math.cos(r) - dy * math.sin(r),
                                    y=0.5 + dx * math.sin(r) + dy * math.cos(r), z=0.0)
    return lm


def set_iris(lm, h=0.0, v=0.0):
    """Place both irises inside the eyes. h,v are fractions of eye width."""
    for iris, c1, c2, top, bottom in [(468, 33, 133, 159, 145), (473, 362, 263, 386, 374)]:
        cx = 0.5 + (-0.1 if c1 == 33 else 0.1)
        lm[c1] = SimpleNamespace(x=cx - 0.03, y=0.4, z=0)
        lm[c2] = SimpleNamespace(x=cx + 0.03, y=0.4, z=0)
        lm[top] = SimpleNamespace(x=cx, y=0.4 - 0.01, z=0)
        lm[bottom] = SimpleNamespace(x=cx, y=0.4 + 0.01, z=0)
        lm[iris] = SimpleNamespace(x=cx + h * 0.06, y=0.4 + v * 0.06, z=0)
    return lm


def face(head_x=0.0, head_y=0.0, **blend):
    return FaceData(head_x=head_x, head_y=head_y, blend=blend)


class FakeActions:
    """Records what the controller asked for; moves nothing."""

    def __init__(self, w=1000, h=800):
        self.w, self.h = w, h
        self.paused = False
        self.moves, self.clicks, self.scrolls, self.typed = [], 0, [], []
        self.pos = (w // 2, h // 2)

    def screen_size(self):
        return (self.w, self.h)

    def position(self):
        return self.pos

    def is_paused(self):
        return self.paused

    def move_to(self, x, y):
        if self.paused:
            return False
        self.moves.append((x, y))
        self.pos = (x, y)
        return True

    def click(self):
        if not self.paused:
            self.clicks += 1

    def scroll(self, n):
        if not self.paused and n:
            self.scrolls.append(n)

    def set_paused(self, value):
        self.paused = value

    def type_text(self, text):
        if not self.paused and text:
            self.typed.append(text)
