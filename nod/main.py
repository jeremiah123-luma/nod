"""M0: webcam preview with face landmarks and FPS. Press q to quit."""
import time

import cv2
from mediapipe.tasks.python import vision

from nod.camera import Camera
from nod.face import FaceTracker

WINDOW = "Nod (press q to quit)"
CONNECTIONS = vision.FaceLandmarksConnections


def draw_face(frame, landmarks):
    """Draw the face mesh. Landmarks are 0..1 fractions, so scale by frame size."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for c in CONNECTIONS.FACE_LANDMARKS_TESSELATION:
        cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1, cv2.LINE_AA)
    for group in (CONNECTIONS.FACE_LANDMARKS_CONTOURS,):
        for c in group:
            cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 160), 1, cv2.LINE_AA)


def main():
    camera = Camera()
    tracker = FaceTracker()
    fps, last = 0.0, time.monotonic()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            result = tracker.process(frame)
            if result.face_landmarks:
                draw_face(frame, result.face_landmarks[0])
                status = "face found"
            else:
                status = "no face"

            # Smoothed FPS so the number doesn't flicker.
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(frame, f"FPS {fps:.0f}  |  {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
