"""The brain. Each frame it takes the face numbers and decides what the mouse should do.

It talks to the mouse ONLY through the `actions` object it is given, so tests can hand it
a fake one and check what it would have done without touching the real mouse.
"""
import math

from nod.gestures import DwellClick, HoldTrigger
from nod.pointer import HeadPointer


class Controller:
    def __init__(self, cfg, actions):
        self.cfg = cfg
        self.actions = actions
        self.pointer = HeadPointer(cfg)
        self.screen_w, self.screen_h = actions.screen_size()
        self.margin = cfg["pointer"]["edge_margin_px"]
        self.cursor = [self.screen_w / 2, self.screen_h / 2]
        k = cfg["click"]
        self.click_trigger = HoldTrigger(k["threshold"], k["release_threshold"], k["hold_ms"], k["cooldown_ms"])
        self.click_shape = k["blendshape"]
        d = cfg["dwell"]
        self.dwell = DwellClick(d["dwell_ms"], d["radius_px"])
        self.dwell_enabled = d["enabled"]
        sc = cfg["scroll"]
        self.scroll_cfg = sc
        self.mode_trigger = HoldTrigger(sc["threshold"], sc["release_threshold"], sc["hold_ms"], sc["cooldown_ms"])
        self.mode = "POINT"
        self._scroll_neutral = None
        self._scroll_entered = 0.0
        self._scroll_acc = 0.0
        self._scroll_last_t = None
        self.scroll_rate = 0.0
        self._was_paused = False
        self.debug = {}

    # ---- helpers ----
    def _clamp(self):
        # Stay 1px inside the screen edge. The top-left corner pixel is pyautogui's
        # emergency-stop corner, and we don't want our own pointer to trip it.
        m = self.margin
        self.cursor[0] = min(max(self.cursor[0], m), self.screen_w - 1 - m)
        self.cursor[1] = min(max(self.cursor[1], m), self.screen_h - 1 - m)

    def _move(self):
        self._clamp()
        self.actions.move_to(*self.cursor)

    def recenter(self):
        """Put the cursor in the middle of the screen and forget the old head position."""
        self.cursor = [self.screen_w / 2, self.screen_h / 2]
        self.pointer.reset()
        self._move()

    def toggle_dwell(self):
        self.dwell_enabled = not self.dwell_enabled
        self.dwell.reset()
        return self.dwell_enabled

    def _reset_gestures(self):
        self.click_trigger.reset()
        self.dwell.reset()
        self.mode_trigger.reset()

    def _set_mode(self, mode, now):
        """Switch POINT <-> SCROLL. Resets the pointer so there's no jump when coming back."""
        self.mode = mode
        self.pointer.reset()
        self.click_trigger.reset()
        self.dwell.reset()
        self._scroll_neutral = None
        self._scroll_acc = 0.0
        self._scroll_last_t = None
        self._scroll_entered = now
        self.scroll_rate = 0.0

    def _scroll(self, head_y, now):
        """Head pitch (relative to where it was when you entered scroll mode) -> scroll speed."""
        sc = self.scroll_cfg
        dt = 0.0 if self._scroll_last_t is None else now - self._scroll_last_t
        self._scroll_last_t = now
        settling = now - self._scroll_entered < sc["settle_ms"] / 1000.0
        if self._scroll_neutral is None or settling:
            # Raising eyebrows can nudge your head; wait a moment, then take THIS as "level".
            self._scroll_neutral = head_y
            self.scroll_rate = 0.0
            return
        offset = head_y - self._scroll_neutral
        beyond = abs(offset) - sc["deadzone"]
        if beyond <= 0:
            self.scroll_rate = 0.0
            self._scroll_acc = 0.0
            return
        rate = min(beyond * sc["gain"], sc["max_clicks_per_s"])
        direction = 1 if offset > 0 else -1          # head down = scroll down
        if sc["invert"]:
            direction = -direction
        self.scroll_rate = rate * direction
        self._scroll_acc += self.scroll_rate * dt
        whole = math.trunc(self._scroll_acc)
        if whole:
            self._scroll_acc -= whole
            self.actions.scroll(-whole)              # pyautogui: negative = down

    @property
    def display_mode(self):
        return "PAUSED" if self.actions.is_paused() else self.mode

    # ---- the main step, called once per camera frame ----
    def update(self, face, now):
        if self.actions.is_paused():
            self._was_paused = True
            return
        if self._was_paused:
            # We were paused: the user may have moved the real mouse. Start from where it is now.
            self._was_paused = False
            self.cursor = list(self.actions.position())
            self.pointer.reset()
            self._reset_gestures()

        if face is None:
            self.pointer.reset()  # avoid a jump when the face comes back
            self._reset_gestures()
            self._scroll_neutral = None
            self._scroll_last_t = None
            self.debug = {}
            return

        dx, dy, speed = self.pointer.update(face.head_x, face.head_y, now)
        jaw = face.blend.get(self.click_shape, 0.0)
        brow = face.blend.get(self.scroll_cfg["toggle_blendshape"], 0.0)
        self.debug = {"head_x": face.head_x, "head_y": face.head_y, "jaw": jaw, "brow": brow}

        # Eyebrow raise (held briefly) flips between POINT and SCROLL.
        if self.mode_trigger.update(brow, now):
            self._set_mode("SCROLL" if self.mode == "POINT" else "POINT", now)
            return

        if self.mode == "SCROLL":
            self._scroll(self.pointer.filtered[1], now)   # cursor stays frozen, clicks are off
            return

        if dx or dy:
            self.cursor[0] += dx
            self.cursor[1] += dy
            self._move()

        if self.click_trigger.update(jaw, now):
            self.actions.click()
        if self.dwell_enabled and self.dwell.update(self.cursor, now):
            self.actions.click()

    # ---- what the HUD should show ----
    def hud_info(self):
        lines = [f"head x {self.debug.get('head_x', 0):+6.1f}   y {self.debug.get('head_y', 0):+6.1f}",
                 f"{self.click_shape} {self.debug.get('jaw', 0):.2f}  (click at {self.click_trigger.threshold})",
                 f"{self.scroll_cfg['toggle_blendshape']} {self.debug.get('brow', 0):.2f}  (toggle at {self.mode_trigger.threshold})"]
        if self.mode == "SCROLL":
            lines.append(f"scroll {self.scroll_rate:+5.1f} lines/s  (head down = down)")
        bars = [("click", self.click_trigger.progress, (80, 200, 80)),
                ("scroll toggle", self.mode_trigger.progress, (255, 160, 60))]
        if self.dwell_enabled:
            bars.append(("dwell", self.dwell.progress, (200, 200, 60)))
        return {"mode": self.display_mode, "lines": lines, "bars": bars}
