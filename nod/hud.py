"""The preview window: face mesh + mode banner + progress bars + raw numbers for tuning."""
import cv2
from mediapipe.tasks.python import vision

CONNECTIONS = vision.FaceLandmarksConnections

MODE_COLORS = {            # BGR
    "POINT": (80, 200, 80),
    "SCROLL": (255, 160, 60),
    "PAUSED": (60, 60, 230),
}


def draw_mesh(frame, landmarks):
    """Landmarks are 0..1 fractions, so scale by frame size."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for c in CONNECTIONS.FACE_LANDMARKS_TESSELATION:
        cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1, cv2.LINE_AA)
    for c in CONNECTIONS.FACE_LANDMARKS_CONTOURS:
        cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 160), 1, cv2.LINE_AA)


def _text(frame, s, x, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(frame, info, fps, face_found):
    """info = controller.hud_info(): {"mode": str, "lines": [str], "bars": [(label, 0..1, BGR)]}"""
    h, w = frame.shape[:2]
    mode = info["mode"]
    color = MODE_COLORS.get(mode, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.circle(frame, (20, 17), 9, color, -1)
    _text(frame, mode, 38, 24, color, 0.7)
    _text(frame, f"FPS {fps:.0f}" + ("" if face_found else "  |  no face"), w - 190, 24)

    y = h - 12 - 20 * (len(info["lines"]) + len(info["bars"]))
    for line in info["lines"]:                       # raw numbers, for tuning
        _text(frame, line, 10, y)
        y += 20
    for label, value, bar_color in info["bars"]:     # progress bars (click, dwell, jump...)
        _text(frame, label, 10, y)
        x0, bw = 130, 160
        cv2.rectangle(frame, (x0, y - 12), (x0 + bw, y + 2), (70, 70, 70), -1)
        cv2.rectangle(frame, (x0, y - 12), (x0 + int(bw * max(0.0, min(1.0, value))), y + 2), bar_color, -1)
        y += 20
