"""One Euro filter: steady when you hold still, fast to respond when you move.

Idea from the paper "1 Euro Filter" (Casiez et al., 2012), written from scratch here.
- min_cutoff: lower = smoother when still (but laggier).
- beta: higher = less lag during fast movement.
- speed_floor (our addition): speeds below this are treated as noise, not movement. Without it,
  the noise in the speed estimate itself opens the filter up while you're holding still.
"""
import math


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, speed_floor=0.0):
        self.min_cutoff = min_cutoff
        self.speed_floor = speed_floor
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = 0.0   # last filtered value
        self._dx = 0.0  # last filtered speed

    def __call__(self, x, t):
        """x = new raw value, t = time in seconds. Returns the filtered value."""
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        raw_speed = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * raw_speed + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * max(0.0, abs(self._dx) - self.speed_floor)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x
