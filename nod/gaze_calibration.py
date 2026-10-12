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
WINDOW = "Nod gaze calibration (ESC cancels)"


def average_features(samples):
    """Median is robust to the odd bad frame (a half-blink, a glance away)."""
    return np.median(np.array(samples), axis=0)


def _draw(screen_size, target, phase_progress, collecting, message):
    w, h = screen_size
    img = np.zeros((h, w, 3), np.uint8) + 20
    x, y = int(target[0]), int(target[1])
    radius = int(28 - 18 * phase_progress)   # dot shrinks while you look at it
    cv2.circle(img, (x, y), radius + 8, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius, (60, 220, 90) if collecting else (60, 160, 255), -1, cv2.LINE_AA)
    cv2.putText(img, message, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return img


def run_calibration(camera, tracker, aspect, cfg, screen_size=None):
    """Shows the dots, records your eyes, fits the mapping. Returns a GazeMapper, or None if cancelled/failed."""
    g = cfg["gaze"]
    screen = tuple(screen_size or actions.screen_size())
    base = calibration_targets(*screen, margin=g["grid_margin"])
    plan = []
    for p in range(g["passes"]):
        plan += (base if p % 2 == 0 else base[::-1])   # second pass runs backwards: no pattern to anticipate
    settle, collect = g["settle_ms"] / 1000.0, g["collect_ms"] / 1000.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    dots, skipped = [], 0
    try:
        for n, target in enumerate(plan, 1):
            start, frames = time.monotonic(), []
            while True:
                t = time.monotonic() - start
                if t > settle + collect:
                    break
                frame = camera.read()
                if frame is not None:
                    face = tracker.analyze(tracker.process(frame), aspect)
                    if face is not None and t > settle and not is_blinking(face, g["blink_threshold"]):
                        frames.append((face.gaze_h, face.gaze_v, face.head_x, face.head_y))
                collecting = t > settle
                progress = (t - settle) / collect if collecting else 0.0
                msg = f"Look at the dot ({n}/{len(plan)}). Keep your head natural. ESC cancels."
                cv2.imshow(WINDOW, _draw(screen, target, max(0.0, progress), collecting, msg))
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    return None
            if len(frames) >= 8:                   # need a handful of good frames per dot
                gh, gv, hx, hy = np.median(np.array(frames), axis=0)   # median ignores the odd bad frame
                dots.append((SimpleNamespace(gaze_h=gh, gaze_v=gv, head_x=hx, head_y=hy), target))
            else:
                skipped += 1
    finally:
        cv2.destroyWindow(WINDOW)

    mapper, report = fit_best(dots, screen, g["ridge"], g["model"], signal_tag(cfg))
    if mapper is None:
        print(f"Calibration failed: only {len(dots)} good dots (skipped {skipped}). Face the camera and retry.")
        return None
    pct, text = verdict(mapper.error_px, screen)
    print(f"Calibration: {len(dots)} dots used, {skipped} skipped.")
    for label, err in report:
        print(f"  model {label:<16} typical error {err:5.0f} px" + ("   <- chosen" if label == mapper.label else ""))
    print(f"  Result: ~{mapper.error_px:.0f} px = {pct:.0f}% of the screen diagonal. {text}")
    mapper.save(CAL_PATH)
    return mapper


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
