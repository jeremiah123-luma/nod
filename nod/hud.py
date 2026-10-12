"""The preview window: face mesh + mode banner + progress bars + raw numbers for tuning.

Styled to match the browser Command Center: black panels, off-white text, one orange accent.
(OpenCV colours are BGR, so orange #F54A00 is written (0, 74, 245).)
"""
import cv2
import numpy as np
from mediapipe.tasks.python import vision

CONNECTIONS = vision.FaceLandmarksConnections

BLACK = (0, 0, 0)
TEXT = (224, 224, 224)       # #E0E0E0
MUTED = (134, 134, 134)      # #868686
DIM = (70, 70, 70)
ORANGE = (0, 74, 245)        # #F54A00 accent
GREEN = (132, 220, 61)       # #3DDC84
RED = (58, 69, 255)          # #FF453A

MODE_COLORS = {"POINT": GREEN, "SCROLL": ORANGE, "PAUSED": RED}
MODE_HINT = {"POINT": "head moves the cursor, open mouth clicks",
             "SCROLL": "tilt head to scroll",
             "PAUSED": "mouse control is off"}


def draw_mesh(frame, landmarks):
    """Landmarks are 0..1 fractions, so scale by frame size. Faint mesh, brighter contours."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for c in CONNECTIONS.FACE_LANDMARKS_TESSELATION:
        cv2.line(frame, pts[c.start], pts[c.end], (48, 48, 48), 1, cv2.LINE_AA)
    for c in CONNECTIONS.FACE_LANDMARKS_CONTOURS:
        cv2.line(frame, pts[c.start], pts[c.end], (190, 190, 190), 1, cv2.LINE_AA)
    for c in CONNECTIONS.FACE_LANDMARKS_LEFT_IRIS + CONNECTIONS.FACE_LANDMARKS_RIGHT_IRIS:
        cv2.line(frame, pts[c.start], pts[c.end], ORANGE, 1, cv2.LINE_AA)


def _text(frame, s, x, y, color=TEXT, scale=0.5, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX, outline=True):
    """outline=True adds a dark halo so text reads on top of video. Skip it inside black panels:
    OpenCV spaces thick text wider than thin text, so the halo would drift away from the letters."""
    if outline:
        cv2.putText(frame, s, (x, y), font, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _panel(frame, x0, y0, x1, y1, alpha=0.72):
    """A translucent black rectangle: keeps text readable without hiding the video."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (40, 40, 40), 1)


def _plot(frame, label, raw, smooth, x0, y0, w=150, h=44):
    """Raw (grey) vs smoothed (orange) over the last ~4 seconds, auto-scaled around the smoothed value."""
    if len(smooth) < 2:
        return
    _panel(frame, x0, y0, x0 + w, y0 + h, 0.8)
    center = sum(smooth) / len(smooth)
    span = max(2.0, max(abs(v - center) for v in raw + smooth))   # at least +-2 units tall

    def pt(i, v):
        return (x0 + int(i * (w - 1) / (len(smooth) - 1)), y0 + h // 2 - int((v - center) / span * (h / 2 - 3)))

    cv2.polylines(frame, [np.array([pt(i, v) for i, v in enumerate(raw)], np.int32)], False, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.polylines(frame, [np.array([pt(i, v) for i, v in enumerate(smooth)], np.int32)], False, ORANGE, 1, cv2.LINE_AA)
    _text(frame, f"{label}  +-{span:.1f}", x0 + 4, y0 + 12, MUTED, 0.38, outline=False)


def draw_gaze_map(frame, gm):
    """A tiny map of the SCREEN: white dot = where you're looking, orange ring = cursor,
    dashed circle = the 'near cursor' zone (look outside it, hold steady, and the cursor jumps)."""
    fh, fw = frame.shape[:2]
    sw, sh = gm["screen"]
    mw = 200
    mh = max(40, int(mw * sh / sw))
    x0, y0 = fw - mw - 10, fh - mh - 12
    k = mw / sw
    _panel(frame, x0, y0, x0 + mw, y0 + mh, 0.8)
    for i in (1, 2):
        cv2.line(frame, (x0 + mw * i // 3, y0), (x0 + mw * i // 3, y0 + mh), (40, 40, 40), 1)
        cv2.line(frame, (x0, y0 + mh * i // 3), (x0 + mw, y0 + mh * i // 3), (40, 40, 40), 1)

    def at(p):
        return (x0 + int(max(0, min(sw, p[0])) * k), y0 + int(max(0, min(sh, p[1])) * k))

    cx, cy = at(gm["cursor"])
    cv2.circle(frame, (cx, cy), max(2, int(gm["radius"] * k)), (90, 90, 90), 1, cv2.LINE_AA)
    for p in gm["trail"][:-1]:
        cv2.circle(frame, at(p), 2, (120, 120, 120), -1)
    if gm["gaze"] is not None:
        cv2.circle(frame, at(gm["gaze"]), 6, TEXT, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, ORANGE, 2, cv2.LINE_AA)
    if gm.get("flash"):
        cv2.circle(frame, at(gm["flash"]), 12, ORANGE, 2, cv2.LINE_AA)
    _text(frame, "screen map: gaze white, cursor orange" + ("" if gm["enabled"] else "  (OFF)"), x0 - 24, y0 - 7, MUTED, 0.36, outline=False)


def draw_hud(frame, info, fps, face_found):
    """info = controller.hud_info(): {"mode": str, "lines": [str], "bars": [(label, 0..1, BGR)]}"""
    h, w = frame.shape[:2]
    mode = info["mode"]
    color = MODE_COLORS.get(mode, MUTED)

    # ---- top banner: wordmark, mode, fps ----
    cv2.rectangle(frame, (0, 0), (w, 38), BLACK, -1)
    cv2.line(frame, (0, 38), (w, 38), (40, 40, 40), 1)
    cv2.putText(frame, "NO", (12, 27), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "D", (12 + 33, 27), cv2.FONT_HERSHEY_DUPLEX, 0.8, ORANGE, 2, cv2.LINE_AA)
    cv2.circle(frame, (92, 19), 6, color, -1, cv2.LINE_AA)
    cv2.putText(frame, mode, (106, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    _text(frame, MODE_HINT.get(mode, ""), 106 + 14 * len(mode) + 14, 26, MUTED, 0.42)
    status = f"{fps:.0f} fps" + ("" if face_found else "   NO FACE")
    _text(frame, status, w - 12 - 9 * len(status), 26, TEXT if face_found else RED, 0.5)

    # ---- raw vs smoothed graphs ----
    trace = info.get("trace") or []
    if len(trace) > 2:
        _plot(frame, "head x", [t[0] for t in trace], [t[1] for t in trace], w - 160, 46)
        _plot(frame, "head y", [t[2] for t in trace], [t[3] for t in trace], w - 160, 96)

    if info.get("gaze_map"):
        draw_gaze_map(frame, info["gaze_map"])

    # ---- numbers + progress bars, on a panel so they're easy to read ----
    n_rows = len(info["lines"]) + len(info["bars"])
    row = 19
    top = h - 12 - row * n_rows - 8
    text_w = max((cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0] for s in info["lines"]), default=0)
    width = min(w - 12, max(300, text_w + 24))          # measured, so long lines never spill out of the panel
    _panel(frame, 6, top, 6 + width, h - 6)
    y = top + 20
    for line in info["lines"]:
        _text(frame, line, 14, y, TEXT, 0.43, outline=False)
        y += row
    for label, value, _bar_color in info["bars"]:
        _text(frame, label, 14, y, MUTED, 0.43, outline=False)
        x0, bw = 130, 150
        cv2.rectangle(frame, (x0, y - 11), (x0 + bw, y + 1), (50, 50, 50), -1)
        filled = int(bw * max(0.0, min(1.0, value)))
        if filled > 0:
            cv2.rectangle(frame, (x0, y - 11), (x0 + filled, y + 1), ORANGE, -1)
        y += row
