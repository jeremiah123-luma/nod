"""The look-at-the-dots calibration screen. Run it from Nod with the 'k' key, or on its own:

    .venv/bin/python -m nod.gaze_calibration

Sit the way you normally sit, then look at each dot until it moves. Press ESC to cancel.
It never touches the mouse. Nothing is saved except a few numbers in gaze_calibration.json.
"""
import time
from types import SimpleNamespace

import cv2
import numpy as np

from nod import actions
from nod.config import CONFIG_PATH, load_config
from nod.gaze import calibration_targets, fit_best, is_blinking

CAL_PATH = CONFIG_PATH.parent / "gaze_calibration.json"


def signal_tag(cfg):
    """Names the head-signal version. A calibration made with a different one is thrown away."""
    return f"{cfg['pointer']['source']}-v2"


def verdict(error_px, screen):
    """Plain-English quality of a calibration."""
    pct = 100 * error_px / (screen[0] ** 2 + screen[1] ** 2) ** 0.5
    if pct <= 8:
        return pct, "GOOD: eye jump should work well."
    if pct <= 13:
        return pct, "OK: eye jump will work for far jumps. Retry for better (sit centered, steady light)."
    return pct, "POOR: jumps will often miss. Retry: sit centered, face the light, keep your head natural."
WINDOW = "Nod eye calibration (ESC cancels)"

BLACK, TEXT, MUTED, ORANGE, GREEN, RED = (0, 0, 0), (224, 224, 224), (134, 134, 134), (0, 74, 245), (132, 220, 61), (58, 69, 255)
INTRO_SECONDS, OUTRO_SECONDS = 2.5, 3.0


def average_features(samples):
    """Median is robust to the odd bad frame (a half-blink, a glance away)."""
    return np.median(np.array(samples), axis=0)


def _put(img, text, x, y, scale, color, thick=1, center=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    if center:
        x -= cv2.getTextSize(text, font, scale, thick)[0][0] // 2
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def _wordmark(img, k):
    _put(img, "NO", 28 * k, 48 * k, 1.0 * k, TEXT, 2)
    _put(img, "D", 28 * k + 42 * k, 48 * k, 1.0 * k, ORANGE, 2)
    _put(img, "eye calibration", 28 * k + 70 * k, 46 * k, 0.55 * k, MUTED)


def _draw(screen_size, target, phase_progress, collecting, message, step=None, total=None, face_ok=True):
    """One frame of the dot screen. Black canvas, a ring that fills (orange) while we record,
    a step counter, and a warning if we can't see your face."""
    w, h = screen_size
    k = w / 1512.0                                   # scale text for any screen size
    img = np.zeros((h, w, 3), np.uint8)
    _wordmark(img, k)
    x, y = int(target[0]), int(target[1])
    ring = int(40 * k) + 6
    cv2.circle(img, (x, y), ring, (60, 60, 60), 2, cv2.LINE_AA)
    if collecting and phase_progress > 0:                      # progress arc, clockwise from the top
        cv2.ellipse(img, (x, y), (ring, ring), -90, 0, 360 * min(1.0, phase_progress), ORANGE, 4, cv2.LINE_AA)
    cv2.circle(img, (x, y), int(13 * k) + 2, ORANGE if collecting else TEXT, -1, cv2.LINE_AA)
    _put(img, message, w / 2, 56 * k, 0.8 * k, TEXT, 1, center=True)
    if not face_ok:
        _put(img, "I can't see your face. Look at the screen, with light on your face.", w / 2, 92 * k, 0.65 * k, RED, 1, center=True)
    if step and total:
        cv2.rectangle(img, (0, h - 6), (int(w * (step - 1 + min(1.0, phase_progress)) / total), h), ORANGE, -1)
        _put(img, f"{step} / {total}", w / 2, h - 34 * k, 0.7 * k, MUTED, 1, center=True)
    _put(img, "ESC to cancel", w - 150 * k, h - 34 * k, 0.5 * k, MUTED)
    return img


def _card(screen_size, title, lines, accent=ORANGE, title_color=TEXT):
    """A message screen (intro / result)."""
    w, h = screen_size
    k = w / 1512.0
    img = np.zeros((h, w, 3), np.uint8)
    _wordmark(img, k)
    top = h / 2 - (len(lines) * 24 + 40) * k
    cv2.rectangle(img, (int(w / 2 - 40 * k), int(top - 70 * k)), (int(w / 2 + 40 * k), int(top - 64 * k)), accent, -1)
    _put(img, title, w / 2, top, 1.7 * k, title_color, 2, center=True)
    for i, line in enumerate(lines):
        _put(img, line, w / 2, top + (60 + 44 * i) * k, 0.8 * k, MUTED if i else TEXT, 1, center=True)
    return img


def _show(window, img, seconds):
    """Show a card for a few seconds. Returns False if ESC was pressed."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cv2.imshow(window, img)
        if cv2.waitKey(30) & 0xFF == 27:
            return False
    return True


def run_calibration(camera, tracker, aspect, cfg, screen_size=None, on_tick=None):
    """Shows the dots, records your eyes, fits the mapping. Returns a GazeMapper, or None if cancelled/failed.

    on_tick(step, total, text) is called every frame so the caller can show progress elsewhere.
    """
    g = cfg["gaze"]
    screen = tuple(screen_size or actions.screen_size())
    base = calibration_targets(*screen, margin=g["grid_margin"])
    plan = []
    for p in range(g["passes"]):
        plan += (base if p % 2 == 0 else base[::-1])   # second pass runs backwards: no pattern to anticipate
    settle, collect = g["settle_ms"] / 1000.0, g["collect_ms"] / 1000.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    dots, skipped, mapper = [], 0, None
    try:
        intro = _card(screen, "Eye calibration", [
            "Sit the way you normally sit.",
            "Look at each dot until it moves.",
            "Keep your head natural. Blinking is fine.",
            f"{len(plan)} dots, about {int(len(plan) * (settle + collect))} seconds."])
        if not _show(WINDOW, intro, INTRO_SECONDS):
            return None
        last_face = time.monotonic()
        for n, target in enumerate(plan, 1):
            start, frames = time.monotonic(), []
            while True:
                t = time.monotonic() - start
                if t > settle + collect:
                    break
                frame = camera.read()
                if frame is not None:
                    face = tracker.analyze(tracker.process(frame), aspect)
                    if face is not None:
                        last_face = time.monotonic()
                        if t > settle and not is_blinking(face, g["blink_threshold"]):
                            frames.append((face.gaze_h, face.gaze_v, face.head_x, face.head_y))
                collecting = t > settle
                progress = (t - settle) / collect if collecting else 0.0
                if on_tick:
                    on_tick(n, len(plan), f"Look at the dot ({n}/{len(plan)})")
                img = _draw(screen, target, max(0.0, progress), collecting, "Look at the dot", n, len(plan),
                            face_ok=time.monotonic() - last_face < 1.0)
                cv2.imshow(WINDOW, img)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    return None
            if len(frames) >= 8:                   # need a handful of good frames per dot
                gh, gv, hx, hy = np.median(np.array(frames), axis=0)   # median ignores the odd bad frame
                dots.append((SimpleNamespace(gaze_h=gh, gaze_v=gv, head_x=hx, head_y=hy), target))
            else:
                skipped += 1

        mapper, report = fit_best(dots, screen, g["ridge"], g["model"], signal_tag(cfg))
        if mapper is None:
            print(f"Calibration failed: only {len(dots)} good dots (skipped {skipped}). Face the camera and retry.")
            _show(WINDOW, _card(screen, "Couldn't calibrate", ["I couldn't see your face for enough of the dots.",
                                                               "Face the camera, with light on your face, and try again."],
                                RED, RED), OUTRO_SECONDS)
            return None
        pct, text = verdict(mapper.error_px, screen)
        print(f"Calibration: {len(dots)} dots used, {skipped} skipped.")
        for label, err in report:
            print(f"  model {label:<16} typical error {err:5.0f} px" + ("   <- chosen" if label == mapper.label else ""))
        print(f"  Result: ~{mapper.error_px:.0f} px = {pct:.0f}% of the screen diagonal. {text}")
        mapper.save(CAL_PATH)
        word = text.split(":")[0]
        color = GREEN if word == "GOOD" else ORANGE if word == "OK" else RED
        advice = text.split(": ", 1)[1]
        _show(WINDOW, _card(screen, word.capitalize(), [f"About {pct:.0f}% error. {advice[0].upper() + advice[1:]}",
                                                        "Eye jump is on. Look far away and hold steady."], color, color), OUTRO_SECONDS)
        return mapper
    finally:
        cv2.destroyWindow(WINDOW)


if __name__ == "__main__":
    from nod.camera import Camera
    from nod.face import FaceTracker

    cfg = load_config()
    cam = Camera(cfg["camera"]["index"], cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"])
    trk = FaceTracker(cfg)
    try:
        run_calibration(cam, trk, cfg["camera"]["width"] / cfg["camera"]["height"], cfg)
    finally:
        trk.close()
        cam.release()
