"""Head movement -> cursor movement (relative, like a trackpad).

Two layers fight jitter:
1. A One Euro filter: heavy smoothing while the head is still, light smoothing when it moves fast.
2. A lock-and-release deadzone. While LOCKED the cursor does not move at all; the head has to
   travel `deadzone_px` (measured in cursor pixels) before it releases and starts following.
   It locks again once the head has stayed inside a small circle (`still_radius_px`) for
   `still_ms`. Jitter stays inside that circle; a deliberate move leaves it immediately.
"""
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
        self.still_radius = p["still_radius_px"]
        self.still_time = p["still_ms"] / 1000.0
        self._fx = OneEuroFilter(p["filter_min_cutoff"], p["filter_beta"], p["filter_d_cutoff"], p["filter_speed_floor"])
        self._fy = OneEuroFilter(p["filter_min_cutoff"], p["filter_beta"], p["filter_d_cutoff"], p["filter_speed_floor"])
        self.reset()

    def reset(self):
        """Forget the last head position (use after losing the face, pausing, recentering)."""
        self._fx.reset()
        self._fy.reset()
        self._pos = None          # smoothed head position, converted to cursor-pixel units
        self._pos_t = None
        self._anchor = None       # where the head was when we locked
        self._win_center = None   # center of the "has the head stayed put?" circle
        self._win_start = 0.0
        self.state = "LOCKED"     # LOCKED = cursor frozen, MOVING = cursor follows the head
        self.raw = None           # latest raw head numbers (for the HUD)
        self.filtered = None      # latest smoothed head numbers (for the HUD and scroll mode)
        self.speed_px = 0.0       # smoothed head speed in cursor-px/s; "is the head still?" reads this

    def update(self, head_x, head_y, now):
        """Returns (dx_px, dy_px, speed_px_per_s): how far to nudge the cursor this frame."""
        self.raw = (head_x, head_y)
        fx, fy = self._fx(head_x, now), self._fy(head_y, now)
        self.filtered = (fx, fy)
        pos = (fx * self.gain_x * self.sign_x, fy * self.gain_y * self.sign_y)

        if self._pos is None:
            self._pos, self._pos_t, self._anchor = pos, now, pos
            return 0.0, 0.0, 0.0

        dt = max(now - self._pos_t, 1e-3)
        step = (pos[0] - self._pos[0], pos[1] - self._pos[1])
        instant = math.hypot(*step) / dt
        self.speed_px = 0.7 * self.speed_px + 0.3 * instant   # a little smoothing so one frame can't fool it
        self._pos, self._pos_t = pos, now

        if self.state == "LOCKED":
            if math.dist(pos, self._anchor) > self.deadzone:
                self.state = "MOVING"                            # released: follow from here on
                self._win_center, self._win_start = pos, now
            return 0.0, 0.0, self.speed_px

        # MOVING: follow the head. Lock again once it has stayed inside a small circle for a moment.
        if math.dist(pos, self._win_center) > self.still_radius:
            self._win_center, self._win_start = pos, now
        elif now - self._win_start >= self.still_time:
            self.state, self._anchor = "LOCKED", pos
            return 0.0, 0.0, self.speed_px

        boost = 1.0 + self.acceleration * instant / 1000.0   # faster head = a bit more reach
        return step[0] * boost, step[1] * boost, self.speed_px
