"""Turns face landmarks into a tiny list of points the dashboard can draw as a wireframe.

Only POINT COORDINATES leave this file, never camera pixels. ~130 points, 0-1000 integers.
"""
from mediapipe.tasks.python import vision

C = vision.FaceLandmarksConnections
GROUP_SOURCES = {
    "oval": C.FACE_LANDMARKS_FACE_OVAL,
    "brows": C.FACE_LANDMARKS_LEFT_EYEBROW + C.FACE_LANDMARKS_RIGHT_EYEBROW,
    "eyes": C.FACE_LANDMARKS_LEFT_EYE + C.FACE_LANDMARKS_RIGHT_EYE,
    "lips": C.FACE_LANDMARKS_LIPS,
    "nose": C.FACE_LANDMARKS_NOSE,
    "irises": C.FACE_LANDMARKS_LEFT_IRIS + C.FACE_LANDMARKS_RIGHT_IRIS,
}

# Landmark numbers we send, in a fixed order; groups refer to positions in that list.
POINT_IDS = sorted({p for conns in GROUP_SOURCES.values() for c in conns for p in (c.start, c.end)})
_POS = {lm: i for i, lm in enumerate(POINT_IDS)}
GROUPS = {name: [[_POS[c.start], _POS[c.end]] for c in conns] for name, conns in GROUP_SOURCES.items()}


def compact(landmarks):
    """[x0, y0, x1, y1, ...] in 0..1000, for POINT_IDS only. Landmarks are 0..1 fractions of the frame."""
    out = []
    for lm in POINT_IDS:
        p = landmarks[lm]
        out.append(round(p.x * 1000))
        out.append(round(p.y * 1000))
    return out


def meta():
    return {"count": len(POINT_IDS), "groups": GROUPS}
