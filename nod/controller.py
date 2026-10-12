"""The brain. Each frame it takes the face numbers and decides what the mouse should do.

It talks to the mouse ONLY through the `actions` object it is given, so tests can hand it
a fake one and check what it would have done without touching the real mouse.
"""
from nod.pointer import HeadPointer


class Controller:
    def __init__(self, cfg, actions):
        self.cfg = cfg
        self.actions = actions
        self.pointer = HeadPointer(cfg)
        self.screen_w, self.screen_h = actions.screen_size()
        self.margin = cfg["pointer"]["edge_margin_px"]
        self.cursor = [self.screen_w / 2, self.screen_h / 2]
        self.mode = "POINT"
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

        if face is None:
            self.pointer.reset()  # avoid a jump when the face comes back
            self.debug = {}
            return

        dx, dy, speed = self.pointer.update(face.head_x, face.head_y, now)
        self.debug = {"head_x": face.head_x, "head_y": face.head_y}
        if dx or dy:
            self.cursor[0] += dx
            self.cursor[1] += dy
            self._move()

    # ---- what the HUD should show ----
    def hud_info(self):
        lines = [f"head x {self.debug.get('head_x', 0):+6.1f}   y {self.debug.get('head_y', 0):+6.1f}"]
        return {"mode": self.display_mode, "lines": lines, "bars": []}
