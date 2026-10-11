"""MediaPipe Face Landmarker setup. Turns a camera frame into face data."""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class FaceTracker:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. See README.")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            # VIDEO mode lets MediaPipe track the face between frames (faster, steadier).
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,                  # jawOpen, browInnerUp, ... (used from M2)
            output_facial_transformation_matrixes=True,    # head pose (used from M1)
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.monotonic()

    def process(self, bgr_frame):
        """Returns the MediaPipe result. result.face_landmarks is empty if no face."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB, OpenCV gives BGR
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        return self.landmarker.detect_for_video(image, timestamp_ms)

    def close(self):
        self.landmarker.close()
