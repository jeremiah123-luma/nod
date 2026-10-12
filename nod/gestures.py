"""Gesture detectors. They only decide WHEN something happened; the controller decides what to do.

All times are in seconds (`now`) and all settings come from config.json.
"""
import math
import statistics


class HoldTrigger:
    """Fires once when a value stays above `threshold` for `hold_ms`.

    - Hysteresis: after firing, the value must drop below `release_threshold` before it can
      fire again, so holding your mouth open (or yawning) gives ONE click, not a stream.
    - Cooldown: after firing, nothing can start for `cooldown_ms`.
    - Between release_threshold and threshold the hold is kept, so a flicker in the
      blendshape number doesn't reset your progress.
    """

    def __init__(self, threshold, release_threshold, hold_ms, cooldown_ms):
        self.threshold = threshold
        self.release = release_threshold
        self.hold = hold_ms / 1000.0
        self.cooldown = cooldown_ms / 1000.0
        self.reset()

    def reset(self):
        self._hold_start = None
        self._armed = True
        self._last_fire = -1e9
        self.progress = 0.0   # 0..1, for the HUD bar

    def update(self, value, now):
        """Returns True on the single frame the gesture fires."""
        if value < self.release:
            self._armed = True
            self._hold_start = None
        can_start = self._armed and (now - self._last_fire) >= self.cooldown
        if value >= self.threshold and can_start and self._hold_start is None:
            self._hold_start = now
        if self._hold_start is None or not can_start:
            self.progress = 0.0
            return False
        self.progress = min(1.0, (now - self._hold_start) / self.hold)
        if self.progress >= 1.0:
            self._armed = False
            self._hold_start = None
            self._last_fire = now
            self.progress = 0.0
            return True
        return False


class DwellClick:
    """Optional: hold the cursor still for `dwell_ms` and it clicks.

    After a dwell click you must move out of `radius_px` before another can happen,
    otherwise resting the cursor would click over and over.
    """

    def __init__(self, dwell_ms, radius_px):
        self.dwell = dwell_ms / 1000.0
        self.radius = radius_px
        self.reset()

    def reset(self):
        self._anchor = None
        self._start = 0.0
        self._done = False
        self.progress = 0.0

    def update(self, pos, now):
        if self._anchor is None or math.dist(pos, self._anchor) > self.radius:
            self._anchor, self._start, self._done = tuple(pos), now, False
            self.progress = 0.0
            return False
        if self._done:
            self.progress = 0.0
            return False
        self.progress = min(1.0, (now - self._start) / self.dwell)
        if self.progress >= 1.0:
            self._done = True
            self.progress = 0.0
            return True
        return False


class Neutral:
    """Your relaxed face and head pose: the reference the toggle gesture is measured against.

    Collects `duration_s` of frames, then keeps the MEDIAN of each number (so a blink or a
    twitch during that moment doesn't spoil it). Restart it (recenter) whenever you change posture.
    """

    def __init__(self, duration_s, shapes):
        self.duration = duration_s
        self.shapes = list(shapes)
        self.restart()

    def restart(self):
        self._start = None
        self._progress = 0.0
        self._head_y = []
        self._shape_vals = {s: [] for s in self.shapes}
        self.ready = False
        self.head_y = 0.0
        self.shape = {s: 0.0 for s in self.shapes}   # relaxed value of each blendshape

    @property
    def progress(self):
        return 1.0 if self.ready else self._progress

    def update(self, face, now):
        if self.ready:
            return
        if self._start is None:
            self._start = now
        self._head_y.append(face.head_y)
        for s in self.shapes:
            self._shape_vals[s].append(face.blend.get(s, 0.0))
        self._progress = min(1.0, (now - self._start) / self.duration)
        if self._progress >= 1.0:
            self.head_y = statistics.median(self._head_y)
            self.shape = {s: statistics.median(v) for s, v in self._shape_vals.items()}
            self.ready = True
