"""Eye jump: where on the screen am I looking, and when should the cursor jump there?

The idea (also used by AACTools/MacGaze, MIT; our code is written from scratch):
  1. Calibrate: look at known dots on screen while we record eye numbers.
  2. Fit a simple mapping from eye numbers -> screen position.
  3. Later, if you look at a spot far from the cursor and keep looking, jump the cursor there.
     Webcam gaze is only accurate to roughly "which part of the screen", so the head pointer
     does the fine aim afterwards.
"""
import json
import math
from pathlib import Path

import numpy as np

from nod.filters import OneEuroFilter


def is_blinking(face, threshold=0.5):
    """During a blink the iris numbers are garbage, so we skip those frames."""
    b = face.blend
    return max(b.get("eyeBlinkLeft", 0.0), b.get("eyeBlinkRight", 0.0)) > threshold


def feature_vector(face, use_head, quadratic):
    f = [face.gaze_h, face.gaze_v]
    if use_head:
        f += [face.head_x / 100.0, face.head_y / 100.0]  # same units as gaze (fractions)
    if quadratic:
        f += [face.gaze_h ** 2, face.gaze_v ** 2, face.gaze_h * face.gaze_v]
    return np.array(f, dtype=float)


def calibration_targets(screen_w, screen_h, margin=0.08):
    """3x3 grid of dots, as (x, y) pixels. Order: center first, then around the edge."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    order = [(1, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    return [(xs[i] * screen_w, ys[j] * screen_h) for i, j in order]


class GazeMapper:
    """Ridge regression (a gentle least-squares line fit) from eye features to screen pixels."""

    def __init__(self, screen_size, use_head=True, quadratic=False, ridge=1.0):
        self.screen = tuple(screen_size)
        self.use_head, self.quadratic, self.ridge = use_head, quadratic, ridge
        self.mean = self.std = self.weights = None
        self.error_px = None        # leave-one-out error: honest estimate of accuracy
        self.n_samples = 0

    @property
    def ready(self):
        return self.weights is not None

    # ---- training ----
    def _solve(self, X, Y):
        mean = X.mean(axis=0)
        std = np.maximum(X.std(axis=0), 1e-3)   # floor so a feature that never varied can't explode
        Xs = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
        penalty = self.ridge * np.eye(Xs.shape[1])
        penalty[-1, -1] = 0.0                   # don't shrink the bias term
        W = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ Y)
        return mean, std, W

    def fit(self, features, targets):
        """features: list of feature vectors; targets: list of (x, y) pixels. Returns LOO error in px."""
        X, Y = np.array(features, dtype=float), np.array(targets, dtype=float)
        if len(X) < X.shape[1] + 2:
            raise ValueError(f"Need at least {X.shape[1] + 2} calibration points, got {len(X)}")
        self.mean, self.std, self.weights = self._solve(X, Y)
        self.n_samples = len(X)
        errs = []
        for i in range(len(X)):                  # leave-one-out: predict each dot from the others
            keep = np.arange(len(X)) != i
            m, s, W = self._solve(X[keep], Y[keep])
            pred = np.append((X[i] - m) / s, 1.0) @ W
            errs.append(np.linalg.norm(pred - Y[i]))
        self.error_px = float(np.sqrt(np.mean(np.square(errs))))
        return self.error_px

    # ---- using it ----
    def predict(self, face):
        x = feature_vector(face, self.use_head, self.quadratic)
        out = np.append((x - self.mean) / self.std, 1.0) @ self.weights
        return (float(min(max(out[0], 0), self.screen[0] - 1)),
                float(min(max(out[1], 0), self.screen[1] - 1)))

    # ---- saving ----
    def save(self, path):
        Path(path).write_text(json.dumps({
            "screen": list(self.screen), "use_head": self.use_head, "quadratic": self.quadratic,
            "ridge": self.ridge, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "weights": self.weights.tolist(), "error_px": self.error_px, "n_samples": self.n_samples,
        }))

    @classmethod
    def load(cls, path, screen_size=None):
        """Returns a ready mapper, or None if there's no usable calibration file."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            m = cls(d["screen"], d["use_head"], d["quadratic"], d["ridge"])
            m.mean, m.std = np.array(d["mean"]), np.array(d["std"])
            m.weights, m.error_px, m.n_samples = np.array(d["weights"]), d["error_px"], d["n_samples"]
        except (ValueError, KeyError, TypeError):
            return None
        if screen_size is not None and tuple(screen_size) != m.screen:
            return None  # calibrated for a different screen size: not valid any more
        return m


class GazeTracker:
    """Mapper + smoothing: turns each frame's face into a steady 'looking at' point."""

    def __init__(self, mapper, gcfg):
        self.mapper = mapper
        self._fx = OneEuroFilter(gcfg["smooth_min_cutoff"], gcfg["smooth_beta"], 1.0)
        self._fy = OneEuroFilter(gcfg["smooth_min_cutoff"], gcfg["smooth_beta"], 1.0)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def update(self, face, now):
        x, y = self.mapper.predict(face)
        return self._fx(x, now), self._fy(y, now)


class GazeJump:
    """Decides when a look is 'deliberate' enough to move the cursor.

    All of these must be true, for `dwell_ms`:
      - the cursor is far from where you're looking (more than min_jump_fraction of the screen diagonal)
      - your gaze is steady (stays within a small radius)
      - your head is nearly still (so we never fight you while you're fine-aiming)
    After a jump there's a cooldown.
    """

    def __init__(self, gcfg, screen_size):
        diag = math.hypot(*screen_size)
        self.min_dist = gcfg["min_jump_fraction"] * diag
        self.stable_radius = gcfg["stable_radius_fraction"] * diag
        self.dwell = gcfg["dwell_ms"] / 1000.0
        self.max_head_speed = gcfg["max_head_speed_px_s"]
        self.cooldown = gcfg["cooldown_ms"] / 1000.0
        self.reset()
        self._last_jump = -1e9

    def reset(self):
        self._anchor = None
        self._start = 0.0
        self.progress = 0.0

    def update(self, gaze, cursor, head_speed, now):
        """Returns the (x, y) to jump to, or None."""
        if (gaze is None or head_speed > self.max_head_speed
                or now - self._last_jump < self.cooldown
                or math.dist(gaze, cursor) < self.min_dist):
            self.reset()
            return None
        if self._anchor is None or math.dist(gaze, self._anchor) > self.stable_radius:
            self._anchor, self._start = gaze, now       # gaze wandered: start counting again
            self.progress = 0.0
            return None
        self.progress = min(1.0, (now - self._start) / self.dwell)
        if self.progress >= 1.0:
            target = gaze
            self._last_jump = now
            self.reset()
            return target
        return None
