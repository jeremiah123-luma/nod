"""The look-at-the-dots calibration screen. Run it from Nod with the 'k' key, or on its own:

    .venv/bin/python -m nod.gaze_calibration

Sit the way you normally sit, then look at each dot until it moves. Press ESC to cancel.
It never touches the mouse. Nothing is saved except a few numbers in gaze_calibration.json.
"""
import time

import cv2
import numpy as np

from nod import actions
from nod.config import CONFIG_PATH, load_config
from nod.gaze import GazeMapper, calibration_targets, feature_vector, is_blinking

CAL_PATH = CONFIG_PATH.parent / "gaze_calibration.json"
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
    """Shows the dots, records your eyes, fits the mapping. Returns a GazeMapper, or None if cancelled."""
    g = cfg["gaze"]
    screen = tuple(screen_size or actions.screen_size())
    mapper = GazeMapper(screen, g["use_head_features"], g["quadratic"], g["ridge"])
    targets = calibration_targets(*screen, margin=g["grid_margin"])
    settle, collect = g["settle_ms"] / 1000.0, g["collect_ms"] / 1000.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    features, used_targets = [], []
    try:
        for n, target in enumerate(targets, 1):
            start, samples = time.monotonic(), []
            while True:
                t = time.monotonic() - start
                if t > settle + collect:
                    break
                frame = camera.read()
                if frame is not None:
                    face = tracker.analyze(tracker.process(frame), aspect)
                    if face is not None and t > settle and not is_blinking(face):
                        samples.append(feature_vector(face, g["use_head_features"], g["quadratic"]))
                collecting = t > settle
                progress = (t - settle) / collect if collecting else 0.0
                msg = f"Look at the dot ({n}/{len(targets)}). Keep your head natural."
                cv2.imshow(WINDOW, _draw(screen, target, max(0.0, progress), collecting, msg))
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    return None
            if len(samples) >= 5:                  # need a handful of good frames per dot
                features.append(average_features(samples))
                used_targets.append(target)
            else:
                print(f"  (dot {n}: no face / too many blinks, skipped)")
    finally:
        cv2.destroyWindow(WINDOW)

    try:
        error = mapper.fit(features, used_targets)
    except ValueError as e:
        print(f"Calibration failed: {e}")
        return None
    diag = (screen[0] ** 2 + screen[1] ** 2) ** 0.5
    print(f"Calibration done. Typical error ~{error:.0f} px ({100 * error / diag:.0f}% of the screen diagonal).")
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
