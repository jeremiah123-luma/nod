"""Head movement -> cursor movement (relative, like a trackpad)."""
import math

from nod.filters import OneEuroFilter


class HeadPointer:
    def __init__(self, cfg):
        p = cfg["pointer"]
        self.gain_x, self.gain_y = p["gain_x"], p["gain_y"]
        self.sign_x = -1 if p["invert_x"] else 1
        self.sign_y = -1 if p["invert_y"] else 1
        self.acceleration = p["acceleration"]
        self.deadzone = p["deadzone_px"]
        self._fx = OneEuroFilter(p["filter_min_cutoff"], p["filter_beta"], p["filter_d_cutoff"])
        self._fy = OneEuroFilter(p["filter_min_cutoff"], p["filter_beta"], p["filter_d_cutoff"])
        self.reset()

    def reset(self):
        """Forget the last head position (use after losing the face, pausing, recentering)."""
        self._fx.reset()
        self._fy.reset()
        self._prev = None
        self._prev_t = None
        self.filtered = None

    def update(self, head_x, head_y, now):
        """Returns (dx_px, dy_px, speed_px_per_s): how far to nudge the cursor this frame."""
        fx, fy = self._fx(head_x, now), self._fy(head_y, now)
        self.filtered = (fx, fy)   # smoothed head position; scroll mode reads this
        if self._prev is None:
            self._prev, self._prev_t = (fx, fy), now
            return 0.0, 0.0, 0.0

        dt = max(now - self._prev_t, 1e-3)
        dx = (fx - self._prev[0]) * self.gain_x * self.sign_x
        dy = (fy - self._prev[1]) * self.gain_y * self.sign_y
        dist = math.hypot(dx, dy)

        # Tiny wiggles are ignored. We DON'T move _prev in that case, so a slow,
        # steady movement still adds up until it passes the deadzone.
        if dist < self.deadzone:
            return 0.0, 0.0, 0.0

        speed = dist / dt
        boost = 1.0 + self.acceleration * speed / 1000.0  # faster head = a bit more reach
        self._prev, self._prev_t = (fx, fy), now
        return dx * boost, dy * boost, speed
