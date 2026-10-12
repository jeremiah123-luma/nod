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
import statistics
from collections import deque
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

    def __init__(self, screen_size, use_head=True, quadratic=False, ridge=1.0, tag=""):
        self.screen = tuple(screen_size)
        self.use_head, self.quadratic, self.ridge = use_head, quadratic, ridge
        self.tag = tag    # which version of the head signal this was calibrated with
        self.mean = self.std = self.weights = None
        self.error_px = None        # leave-one-out error: honest estimate of accuracy
        self.n_samples = 0

    @property
    def ready(self):
        return self.weights is not None

    @property
    def label(self):
        return ("eyes+head" if self.use_head else "eyes only") + (" curved" if self.quadratic else "")

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
        # Leave-one-POSITION-out: hold back every sample of one dot, fit on the rest, predict the held-out
        # dot. (Calibration looks at each dot twice; leaving out just one sample would leave its twin in
        # the training data and make flexible models look better than they are.)
        keys = [tuple(np.round(t, 1)) for t in Y]
        errs = []
        for key in set(keys):
            held = np.array([k == key for k in keys])
            if (~held).sum() < X.shape[1] + 2:
                continue
            m, s, W = self._solve(X[~held], Y[~held])
            for xi, yi in zip(X[held], Y[held]):
                errs.append(np.linalg.norm(np.append((xi - m) / s, 1.0) @ W - yi))
        if not errs:   # too few samples to hold anything out: fall back to the (optimistic) fit error
            errs = [np.linalg.norm(np.append((xi - self.mean) / self.std, 1.0) @ self.weights - yi) for xi, yi in zip(X, Y)]
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
            "tag": self.tag,
        }))

    @classmethod
    def load(cls, path, screen_size=None, tag=None):
        """Returns a ready mapper, or None if there's no usable calibration file."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            m = cls(d["screen"], d["use_head"], d["quadratic"], d["ridge"], d.get("tag", ""))
            m.mean, m.std = np.array(d["mean"]), np.array(d["std"])
            m.weights, m.error_px, m.n_samples = np.array(d["weights"]), d["error_px"], d["n_samples"]
        except (ValueError, KeyError, TypeError):
            return None
        if screen_size is not None and tuple(screen_size) != m.screen:
            return None  # calibrated for a different screen size: not valid any more
        if tag is not None and m.tag != tag:
            return None  # calibrated with an older head signal: the numbers no longer mean the same thing
        return m


CANDIDATES = {"eyes": (False, False), "eyes+head": (True, False), "curved": (False, True)}


def fit_best(dots, screen, ridge, model="auto", tag=""):
    """Fit the candidate models and keep the one that predicts held-out dots best.

    dots: list of (sample, (x, y)) where sample has gaze_h, gaze_v, head_x, head_y.
    Why: with only a few calibration points, a fancier model can fit them perfectly and still be
    worse on new looks. The leave-one-out error is the honest comparison.
    Returns (mapper, report) where report = [(label, error_px)] for every model tried.
    """
    names = list(CANDIDATES) if model == "auto" else [model]
    best, report = None, []
    for name in names:
        use_head, quad = CANDIDATES[name]
        m = GazeMapper(screen, use_head, quad, ridge, tag)
        try:
            err = m.fit([feature_vector(s, use_head, quad) for s, _ in dots], [t for _, t in dots])
        except ValueError:
            continue
        report.append((m.label, err))
        # Candidates run simplest-first. A fancier model must beat the best so far by 10%+, otherwise
        # its "win" is probably luck with so few calibration points.
        if best is None or err < best.error_px * 0.90:
            best = m
    return best, report


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
    """Decides when a look is 'deliberate' enough to move the cursor, and says WHY it didn't.

    A jump happens when, over the last `dwell_ms`:
      - your head is nearly still (so we never fight you while you're fine-aiming)
      - the cursor is far from where you're looking (more than min_jump_fraction of the screen diagonal)
      - your gaze was steady: at least `stable_fraction` of the samples sit within a small radius of
        their median (a median, so one noisy frame can't spoil it)
    The cursor then jumps to that median point. After a jump there's a cooldown.
    `reason` is a short human-readable status for the HUD.
    """

    def __init__(self, gcfg, screen_size):
        diag = math.hypot(*screen_size)
        self.min_dist = gcfg["min_jump_fraction"] * diag
        self.stable_radius = gcfg["stable_radius_fraction"] * diag
        self.stable_fraction = gcfg["stable_fraction"]
        self.dwell = gcfg["dwell_ms"] / 1000.0
        self.max_head_speed = gcfg["max_head_speed_px_s"]
        self.cooldown = gcfg["cooldown_ms"] / 1000.0
        self._samples = deque()
        self._last_jump = -1e9
        self.reset()

    def reset(self):
        self._samples.clear()
        self.progress = 0.0
        self.reason = "waiting for gaze"

    def update(self, gaze, cursor, head_speed, now):
        """Returns the (x, y) to jump to, or None."""
        if gaze is None:
            self.reset()
            return None
        remaining = self.cooldown - (now - self._last_jump)
        if remaining > 0:
            self._samples.clear()
            self.progress = 0.0
            self.reason = f"cooldown {remaining:.1f}s"
            return None
        if head_speed > self.max_head_speed:
            self._samples.clear()
            self.progress = 0.0
            self.reason = f"head moving ({head_speed:.0f} px/s, max {self.max_head_speed:.0f})"
            return None
        d = math.dist(gaze, cursor)
        if d < self.min_dist:
            self._samples.clear()
            self.progress = 0.0
            self.reason = f"looking near cursor ({d:.0f}px, need {self.min_dist:.0f}px+)"
            return None

        self._samples.append((now, gaze))
        while self._samples and now - self._samples[0][0] > self.dwell:
            self._samples.popleft()
        span = now - self._samples[0][0]
        self.progress = min(1.0, span / self.dwell)
        if span < self.dwell * 0.85:
            self.reason = f"far look - holding {self.progress:.0%}"
            return None

        pts = [p for _, p in self._samples]
        med = (statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts))
        close = sum(math.dist(p, med) <= self.stable_radius for p in pts) / len(pts)
        if close < self.stable_fraction:
            self.reason = f"gaze unsteady ({close:.0%} within {self.stable_radius:.0f}px, need {self.stable_fraction:.0%})"
            return None
        self._last_jump = now
        self._samples.clear()
        self.progress = 0.0
        self.reason = "JUMPED"
        return med
