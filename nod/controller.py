"""The brain. Each frame it takes the face numbers and decides what the mouse should do.

It talks to the mouse ONLY through the `actions` object it is given, so tests can hand it
a fake one and check what it would have done without touching the real mouse.
"""
import math
from collections import deque

from nod.gaze import GazeJump, GazeTracker, is_blinking
from nod.gestures import DwellClick, HoldTrigger, Neutral
from nod.pointer import HeadPointer


class Controller:
    def __init__(self, cfg, actions, gaze_mapper=None):
        self.actions = actions
        self.screen_w, self.screen_h = actions.screen_size()
        self.cursor = [self.screen_w / 2, self.screen_h / 2]
        self.neutral = None
        self.gaze = None
        self.gaze_enabled = False
        self.gaze_point = None
        self.gaze_trail = deque(maxlen=25)      # recent gaze points, drawn on the HUD minimap
        self._jump_note = "waiting for gaze"
        self._jump_flash = None                 # (target, until_time) briefly highlights a jump
        self._gate_ok_since = None
        self.toggle_note = "calibrating neutral..."     # shown on the HUD: why the toggle can/can't fire
        self.mode = "POINT"
        self._scroll_neutral = None
        self._scroll_entered = 0.0
        self._scroll_acc = 0.0
        self._scroll_last_t = None
        self.scroll_rate = 0.0
        self._was_paused = False
        self._last_paused = None
        self._neutral_logged = False
        self._last_now = 0.0
        self.debug = {}
        self.trace = deque(maxlen=120)   # recent (raw_x, smooth_x, raw_y, smooth_y), drawn on the HUD
        self.log = deque(maxlen=60)      # activity feed shown on the dashboard
        self._build(cfg)
        self.set_gaze_mapper(gaze_mapper)

    def _build(self, cfg):
        """(Re)creates everything that comes from config.json. Used at startup and by reload()."""
        self.cfg = cfg
        self.pointer = HeadPointer(cfg)
        self.margin = cfg["pointer"]["edge_margin_px"]
        k = cfg["click"]
        self.click_trigger = HoldTrigger(k["threshold"], k["release_threshold"], k["hold_ms"], k["cooldown_ms"])
        self.click_shape = k["blendshape"]
        d = cfg["dwell"]
        self.dwell = DwellClick(d["dwell_ms"], d["radius_px"])
        self.dwell_enabled = d["enabled"]
        sc = cfg["scroll"]
        self.scroll_cfg = sc
        gestures = sc["gestures"]
        if sc["toggle_gesture"] not in gestures:
            raise ValueError(f'scroll.toggle_gesture must be one of {sorted(gestures)}, got "{sc["toggle_gesture"]}"')
        tg = gestures[sc["toggle_gesture"]]
        self.toggle_shape = tg["blendshape"]
        # The toggle fires on a RISE above your relaxed value ("delta"), not on a fixed number.
        self.mode_trigger = HoldTrigger(tg["delta"], tg["release_delta"], tg["hold_ms"], sc["cooldown_ms"])
        shapes = [g["blendshape"] for g in gestures.values()]
        if self.neutral is None or self.neutral.shapes != shapes:
            self.neutral = Neutral(sc["neutral_ms"] / 1000.0, shapes)     # keep what we learned otherwise
        self.gaze_cfg = cfg["gaze"]
        self.gaze_jump = GazeJump(self.gaze_cfg, (self.screen_w, self.screen_h))
        if self.gaze is not None:
            self.gaze = GazeTracker(self.gaze.mapper, self.gaze_cfg)
            self.gaze_enabled = self.gaze_cfg["enabled"]

    def reload(self, cfg):
        """Apply changed settings while running (the dashboard sliders use this)."""
        was_mode = self.mode
        self._build(cfg)
        self.mode = was_mode
        self._log("settings", "Settings updated")

    def _log(self, kind, text):
        import time as _time
        self.log.append({"t": _time.time(), "kind": kind, "text": text})

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
        self.neutral.restart()      # also re-learn your relaxed face + level head pitch
        self._neutral_logged = False
        self._gate_ok_since = None
        self._move()
        self._log("recenter", "Recentered - hold a relaxed face for a moment")

    def set_gaze_mapper(self, mapper):
        """Give the controller a (re)calibrated mapper, or None to turn eye jump off."""
        self.gaze_jump = GazeJump(self.gaze_cfg, (self.screen_w, self.screen_h))
        if mapper is None or not mapper.ready:
            self.gaze, self.gaze_enabled = None, False
        else:
            self.gaze = GazeTracker(mapper, self.gaze_cfg)
            self.gaze_enabled = self.gaze_cfg["enabled"]
        self.gaze_point = None
        self.gaze_trail.clear()
        if self.gaze is not None:
            self._log("gaze", f"Eye jump ready ({mapper.label}, ~{mapper.error_px:.0f}px error)")

    def toggle_gaze(self):
        """Returns True/False for on/off, or None if there's no calibration yet."""
        if self.gaze is None:
            return None
        self.gaze_enabled = not self.gaze_enabled
        self.gaze_jump.reset()
        self.gaze.reset()
        self._log("gaze", "Eye jump ON" if self.gaze_enabled else "Eye jump OFF")
        return self.gaze_enabled

    def _toggle_value(self, face, now):
        """How far past 'relaxed' the toggle gesture is, or 0 if the head isn't calm and level.

        This is what stops tilting your head up (which also lifts your brows) from toggling scroll.
        """
        sc = self.scroll_cfg
        fy = self.pointer.filtered[1]
        if not self.neutral.ready:
            self._gate_ok_since = None
            self.toggle_note = f"calibrating neutral {self.neutral.progress:.0%} (relax your face, look at the screen)"
            return 0.0
        ref_y = self._scroll_neutral if (self.mode == "SCROLL" and self._scroll_neutral is not None) else self.neutral.head_y
        off = abs(fy - ref_y)
        if off > sc["gate_pitch_window"]:
            self._gate_ok_since = None
            self.toggle_note = f"blocked: head pitch {off:.1f} from level (max {sc['gate_pitch_window']:.0f})"
            return 0.0
        if self.pointer.speed_px > sc["gate_max_head_speed_px_s"]:
            self._gate_ok_since = None
            self.toggle_note = f"blocked: head moving ({self.pointer.speed_px:.0f}px/s)"
            return 0.0
        if self._gate_ok_since is None:
            self._gate_ok_since = now
        if now - self._gate_ok_since < sc["gate_settle_ms"] / 1000.0:
            self.toggle_note = "steadying..."
            return 0.0
        self.toggle_note = "ready"
        return face.blend.get(self.toggle_shape, 0.0) - self.neutral.shape[self.toggle_shape]

    def toggle_dwell(self):
        self.dwell_enabled = not self.dwell_enabled
        self.dwell.reset()
        self._log("click", "Dwell click ON" if self.dwell_enabled else "Dwell click OFF")
        return self.dwell_enabled

    def _reset_gestures(self):
        self.click_trigger.reset()
        self.dwell.reset()
        self.mode_trigger.reset()
        self.gaze_jump.reset()
        if self.gaze:
            self.gaze.reset()

    def _set_mode(self, mode, now):
        """Switch POINT <-> SCROLL. Resets the pointer so there's no jump when coming back."""
        self.mode = mode
        self._log("mode", "SCROLL mode on - tilt your head to scroll" if mode == "SCROLL" else "Back to POINT mode")
        self.pointer.reset()
        self.click_trigger.reset()
        self.dwell.reset()
        self.gaze_jump.reset()
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
        self._last_now = now
        paused = self.actions.is_paused()
        if paused != self._last_paused:
            if self._last_paused is not None:
                self._log("pause", "Paused - mouse control off" if paused else "Resumed - you're in control")
            self._last_paused = paused
        if face is None:
            self.pointer.reset()  # avoid a jump when the face comes back
            self._reset_gestures()
            self._scroll_neutral = None
            self._scroll_last_t = None
            self._gate_ok_since = None
            if not self.neutral.ready:
                self.neutral.restart()    # lost the face mid-capture: start the capture over
            self.debug = {}
            self._was_paused = self._was_paused or paused
            return

        # The pointer filter always runs, even while paused, so the HUD can show raw vs smoothed.
        # It only MOVES the cursor when we're not paused (below).
        dx, dy, speed = self.pointer.update(face.head_x, face.head_y, now)
        self.trace.append((*self.pointer.raw[:1], self.pointer.filtered[0], self.pointer.raw[1], self.pointer.filtered[1]))
        jaw = face.blend.get(self.click_shape, 0.0)
        brow = face.blend.get(self.toggle_shape, 0.0)
        self.debug = {"head_x": face.head_x, "head_y": face.head_y, "jaw": jaw, "brow": brow, "speed": speed,
                      "gaze_h": face.gaze_h, "gaze_v": face.gaze_v}
        self.neutral.update(face, now)    # learns your relaxed face; runs even while paused
        if self.neutral.ready and not self._neutral_logged:
            self._neutral_logged = True
            self._log("neutral", "Learned your relaxed face and level head")

        if paused:
            self._was_paused = True
            return
        if self._was_paused:
            # We were paused: the user may have moved the real mouse. Start from where it is now.
            self._was_paused = False
            self.cursor = list(self.actions.position())
            self.pointer.reset()
            self._reset_gestures()
            return

        # Eyebrow raise (held briefly) flips between POINT and SCROLL.
        toggle_delta = self._toggle_value(face, now)
        self.debug["toggle_delta"] = toggle_delta
        if self.mode_trigger.update(toggle_delta, now):
            self._gate_ok_since = None
            self._set_mode("SCROLL" if self.mode == "POINT" else "POINT", now)
            return

        if self.mode == "SCROLL":
            self._scroll(self.pointer.filtered[1], now)   # cursor stays frozen, clicks are off
            return

        if dx or dy:
            self.cursor[0] += dx
            self.cursor[1] += dy
            self._move()

        self._eye_jump(face, speed, now)

        if self.click_trigger.update(jaw, now):
            self.actions.click()
            self._log("click", "Click (mouth)")
        if self.dwell_enabled and self.dwell.update(self.cursor, now):
            self.actions.click()
            self._log("click", "Click (dwell)")

    def _eye_jump(self, face, head_speed, now):
        """Big jump with the eyes; the head pointer does the fine aim afterwards."""
        if not (self.gaze_enabled and self.gaze):
            return
        if is_blinking(face, self.gaze_cfg["blink_threshold"]):
            self._jump_note = "blink (ignored)"
            return   # ignore blink frames entirely (don't reset the look timer either)
        self.gaze_point = self.gaze.update(face, now)
        self.gaze_trail.append(self.gaze_point)
        target = self.gaze_jump.update(self.gaze_point, self.cursor, head_speed, now)
        self._jump_note = self.gaze_jump.reason
        if target:
            self.cursor = list(target)
            self._move()
            self.pointer.reset()   # head deltas restart from here; no leftover motion
            self._jump_flash = (target, now + 0.8)
            self._jump_note = f"JUMPED to ({target[0]:.0f}, {target[1]:.0f})"
            self._log("jump", f"Eye jump to ({target[0]:.0f}, {target[1]:.0f})")

    def jump_status(self):
        """One line saying why an eye jump did or didn't fire right now."""
        if self.gaze is None:
            return "NOT CALIBRATED - press ctrl+alt+k"
        if not self.gaze_enabled:
            return "OFF (ctrl+alt+g turns it on)"
        if self.actions.is_paused():
            return "paused"
        if self.mode == "SCROLL":
            return "idle while in SCROLL mode"
        return self._jump_note

    # ---- what the HUD should show ----
    def hud_info(self):
        raw, sm = self.pointer.raw or (0, 0), self.pointer.filtered or (0, 0)
        lines = [f"head RAW    x {raw[0]:+7.2f}  y {raw[1]:+7.2f}",
                 f"head SMOOTH x {sm[0]:+7.2f}  y {sm[1]:+7.2f}   pointer {self.pointer.state}  {self.pointer.speed_px:4.0f}px/s",
                 f"{self.click_shape} {self.debug.get('jaw', 0):.2f}  (click at {self.click_trigger.threshold})",
                 f"{self.toggle_shape} {self.debug.get('brow', 0):.2f}  relaxed {self.neutral.shape.get(self.toggle_shape, 0):.2f}"
                 f"  rise {self.debug.get('toggle_delta', 0):+.2f} (need +{self.mode_trigger.threshold:.2f})",
                 f"scroll toggle: {self.toggle_note}"]
        if self.mode == "SCROLL":
            lines.append(f"scroll {self.scroll_rate:+5.1f} lines/s  (head down = down)")
        bars = [("click", self.click_trigger.progress, (80, 200, 80)),
                ("scroll toggle", self.mode_trigger.progress, (255, 160, 60))]
        lines.append(f"eye jump: {self.jump_status()}")
        if self.gaze is not None:
            gp = self.gaze_point
            lines.append(f"iris h {self.debug.get('gaze_h', 0):+.2f} v {self.debug.get('gaze_v', 0):+.2f}   gaze "
                         + (f"({gp[0]:.0f}, {gp[1]:.0f})" if gp else "-")
                         + f"   cal {self.gaze.mapper.label} ~{self.gaze.mapper.error_px:.0f}px")
            if self.gaze_enabled:
                bars.append(("eye jump", self.gaze_jump.progress, (255, 100, 200)))
        flash = self._jump_flash[0] if self._jump_flash and self._jump_flash[1] > self._last_now else None
        gaze_map = None
        if self.gaze is not None:
            gaze_map = {"screen": (self.screen_w, self.screen_h), "gaze": self.gaze_point if self.gaze_enabled else None,
                        "trail": list(self.gaze_trail), "cursor": tuple(self.cursor),
                        "radius": self.gaze_jump.min_dist, "enabled": self.gaze_enabled, "flash": flash}
        return {"mode": self.display_mode, "lines": lines, "bars": bars, "trace": list(self.trace), "gaze_map": gaze_map}

    # ---- what the dashboard should show (plain numbers/strings only, safe to send as JSON) ----
    def snapshot(self):
        r2 = lambda v: round(float(v), 2)
        raw, sm = self.pointer.raw or (0.0, 0.0), self.pointer.filtered or (0.0, 0.0)
        jaw = self.debug.get("jaw", 0.0)
        brow = self.debug.get("brow", 0.0)
        relaxed = self.neutral.shape.get(self.toggle_shape, 0.0)
        gaze = {"calibrated": self.gaze is not None, "enabled": self.gaze_enabled, "status": self.jump_status(),
                "progress": r2(self.gaze_jump.progress), "radius": round(self.gaze_jump.min_dist)}
        if self.gaze is not None:
            m = self.gaze.mapper
            diag = math.hypot(self.screen_w, self.screen_h)
            gaze.update(model=m.label, error_px=round(m.error_px), error_pct=round(100 * m.error_px / diag, 1),
                        point=[round(v) for v in self.gaze_point] if (self.gaze_point and self.gaze_enabled) else None,
                        trail=[[round(p[0]), round(p[1])] for p in self.gaze_trail],
                        flash=[round(v) for v in self._jump_flash[0]]
                        if self._jump_flash and self._jump_flash[1] > self._last_now else None)
        return {
            "mode": self.display_mode,
            "paused": self.actions.is_paused(),
            "face": bool(self.debug),
            "screen": [self.screen_w, self.screen_h],
            "cursor": [round(self.cursor[0]), round(self.cursor[1])],
            "pointer": {"state": self.pointer.state, "speed": round(self.pointer.speed_px),
                        "raw": [r2(raw[0]), r2(raw[1])], "smooth": [r2(sm[0]), r2(sm[1])]},
            "trace": [[r2(v) for v in t] for t in list(self.trace)[-90:]],
            "iris": [r2(self.debug.get("gaze_h", 0.0)), r2(self.debug.get("gaze_v", 0.0))],
            "click": {"shape": self.click_shape, "value": r2(jaw), "threshold": self.click_trigger.threshold,
                      "progress": r2(self.click_trigger.progress)},
            "dwell": {"enabled": self.dwell_enabled, "progress": r2(self.dwell.progress)},
            "toggle": {"gesture": self.scroll_cfg["toggle_gesture"], "shape": self.toggle_shape, "value": r2(brow),
                       "relaxed": r2(relaxed), "rise": r2(self.debug.get("toggle_delta", 0.0)),
                       "need": self.mode_trigger.threshold, "note": self.toggle_note,
                       "progress": r2(self.mode_trigger.progress)},
            "neutral": {"ready": self.neutral.ready, "progress": r2(self.neutral.progress)},
            "scroll": {"active": self.mode == "SCROLL", "rate": r2(self.scroll_rate)},
            "gaze": gaze,
            "log": list(self.log)[-30:],
        }
