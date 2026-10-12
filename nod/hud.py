"""The preview window: face mesh + mode banner + progress bars + raw numbers for tuning."""
import cv2
import numpy as np
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


def _plot(frame, label, raw, smooth, x0, y0, w=150, h=44):
    """Raw (grey) vs smoothed (green) over the last ~4 seconds, auto-scaled around the smoothed value."""
    if len(smooth) < 2:
        return
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    center = sum(smooth) / len(smooth)
    span = max(2.0, max(abs(v - center) for v in raw + smooth))   # at least +-2 units tall
    def pt(i, v):
        return (x0 + int(i * (w - 1) / (len(smooth) - 1)), y0 + h // 2 - int((v - center) / span * (h / 2 - 2)))
    cv2.polylines(frame, [np.array([pt(i, v) for i, v in enumerate(raw)], np.int32)], False, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.polylines(frame, [np.array([pt(i, v) for i, v in enumerate(smooth)], np.int32)], False, (0, 230, 120), 2, cv2.LINE_AA)
    _text(frame, f"{label} (+-{span:.1f})", x0 + 3, y0 + 12, (255, 255, 255), 0.4)


def draw_hud(frame, info, fps, face_found):
    """info = controller.hud_info(): {"mode": str, "lines": [str], "bars": [(label, 0..1, BGR)]}"""
    h, w = frame.shape[:2]
    mode = info["mode"]
    color = MODE_COLORS.get(mode, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.circle(frame, (20, 17), 9, color, -1)
    _text(frame, mode, 38, 24, color, 0.7)
    _text(frame, f"FPS {fps:.0f}" + ("" if face_found else "  |  no face"), w - 190, 24)

    trace = info.get("trace") or []
    if len(trace) > 2:                                # grey = raw, green = smoothed
        _plot(frame, "head x", [t[0] for t in trace], [t[1] for t in trace], w - 160, 42)
        _plot(frame, "head y", [t[2] for t in trace], [t[3] for t in trace], w - 160, 92)

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
