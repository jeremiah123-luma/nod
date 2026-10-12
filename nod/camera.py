"""Webcam capture. Frames stay in memory only; we never save or send them."""
import cv2


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index)
        # A smaller frame is faster to process, and landmarks don't need HD.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # macOS webcams quietly default to 15 fps. Asking explicitly gets 30, which halves the lag.
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the webcam. On macOS, check System Settings > "
                "Privacy & Security > Camera and allow your terminal app."
            )

    def read(self):
        """Returns one mirrored BGR frame, or None if the read failed."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        # Mirror it so moving your head right moves the image right, like a mirror.
        return cv2.flip(frame, 1)

    def release(self):
        self.cap.release()
