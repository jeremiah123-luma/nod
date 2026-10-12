"""MediaPipe Face Landmarker setup. Turns a camera frame into simple face numbers."""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Landmark numbers in MediaPipe's 478-point face mesh.
NOSE_TIP = 1
# Several points down the nose and on both nostril wings. Averaging them cancels some of the
# per-point wobble that a single nose-tip landmark has.
NOSE_POINTS = [1, 2, 4, 5, 6, 195, 197, 98, 327, 129, 358]
LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS = (33, 133), (362, 263)
# Per eye: (iris center, corner, corner, upper lid, lower lid)
EYES = [(468, 33, 133, 159, 145), (473, 362, 263, 386, 374)]


@dataclass
class FaceData:
    """Everything the rest of the app needs to know about one frame."""
    head_x: float                 # grows as the head turns toward image-right (mirrored view)
    head_y: float                 # grows as the head tilts down
    blend: dict = field(default_factory=dict)   # e.g. {"jawOpen": 0.7, "browInnerUp": 0.1}
    gaze_h: float = 0.0           # iris offset in the eye, + = looking image-right
    gaze_v: float = 0.0           # + = looking down
    landmarks: list = None        # kept only for drawing the preview


def compute_head_signals(lm, aspect=1.0):
    """Head turn/tilt from landmarks, in 'eye-distance percent' units (the fallback signal).

    We measure where the average of several nose points sits relative to the midpoint between
    the eyes, along the eye line (head_x) and across it (head_y). That's:
    - independent of how far you sit from the camera (divided by eye distance)
    - independent of head roll (we measure along the eye line, not the screen axes)
    - independent of mouth opening (we don't use the jaw at all)
    `aspect` = frame width / height, because landmark x and y are in different units.
    """
    def center(idx):
        return (sum(lm[i].x for i in idx) / len(idx) * aspect, sum(lm[i].y for i in idx) / len(idx))

    a, b = center(LEFT_EYE_CORNERS), center(RIGHT_EYE_CORNERS)
    if a[0] > b[0]:
        a, b = b, a
    ex, ey = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(ex, ey)
    if dist < 1e-6:
        return 0.0, 0.0
    ux, uy = ex / dist, ey / dist          # unit vector along the eye line
    px, py = -uy, ux                        # perpendicular, pointing "down" the face
    nx, ny = center(NOSE_POINTS)
    ox, oy = nx - (a[0] + b[0]) / 2, ny - (a[1] + b[1]) / 2
    return 100.0 * (ox * ux + oy * uy) / dist, 100.0 * (ox * px + oy * py) / dist


def matrix_head_signals(matrix, units_per_sin=45.0):
    """Head turn/tilt from MediaPipe's face transformation matrix (much steadier than landmarks).

    The matrix is MediaPipe's best fit of a 3D face model to ALL the landmarks at once, so
    single-point wobble averages out. We ask: where does the face's "nose direction" point?
    - pointing toward image-right  -> head_x grows   (turning right)
    - pointing down                -> head_y grows   (tilting down)
    Signs were checked against the landmark signal on a real camera (correlation +0.98 / +0.79).
    """
    rot = np.asarray(matrix, dtype=float)[:3, :3]
    det = np.linalg.det(rot)
    if det <= 1e-9:
        return None
    rot = rot / np.cbrt(det)                       # strip any scale, keep the rotation
    forward = rot @ np.array([0.0, 0.0, 1.0])      # canonical face looks along +z
    return units_per_sin * forward[0], -units_per_sin * forward[1]   # matrix y points UP, screen y points down


def compute_gaze_features(lm, aspect=1.0):
    """Where the irises sit inside the eyes (averaged over both eyes).

    Returns (h, v): 0 = looking straight ahead, + = toward image-right / down.
    Divided by eye width so it doesn't depend on distance from the camera.
    """
    hs, vs = [], []
    for iris, c1, c2, top, bottom in EYES:
        width = abs(lm[c1].x - lm[c2].x) * aspect
        if width < 1e-6:
            continue
        cx = (lm[c1].x + lm[c2].x) / 2
        cy = (lm[top].y + lm[bottom].y) / 2
        hs.append((lm[iris].x - cx) * aspect / width)
        vs.append((lm[iris].y - cy) * aspect / width)
    if not hs:
        return 0.0, 0.0
    return sum(hs) / len(hs), sum(vs) / len(vs)


def analyze_landmarks(landmarks, blend, aspect=1.0):
    head_x, head_y = compute_head_signals(landmarks, aspect)
    gaze_h, gaze_v = compute_gaze_features(landmarks, aspect) if len(landmarks) > 477 else (0.0, 0.0)
    return FaceData(head_x, head_y, blend, gaze_h, gaze_v, landmarks)


class FaceTracker:
    def __init__(self, cfg=None):
        p = (cfg or {}).get("pointer", {})
        self.source = p.get("source", "matrix")          # "matrix" or "landmarks"
        self.units_per_sin = p.get("matrix_units_per_sin", 45.0)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. See README.")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            # VIDEO mode lets MediaPipe track the face between frames (faster, steadier).
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,                  # jawOpen, browInnerUp, ...
            output_facial_transformation_matrixes=True,    # head pose: the steadiest signal for the pointer
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.monotonic()

    def process(self, bgr_frame):
        """Returns the raw MediaPipe result. result.face_landmarks is empty if no face."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB, OpenCV gives BGR
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        return self.landmarker.detect_for_video(image, timestamp_ms)

    def analyze(self, result, aspect):
        """Raw result -> FaceData (or None if no face)."""
        if not result.face_landmarks:
            return None
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
        data = analyze_landmarks(result.face_landmarks[0], blend, aspect)
        if self.source == "matrix" and result.facial_transformation_matrixes:
            sig = matrix_head_signals(result.facial_transformation_matrixes[0], self.units_per_sin)
            if sig:
                data.head_x, data.head_y = sig
        return data

    def close(self):
        self.landmarker.close()
