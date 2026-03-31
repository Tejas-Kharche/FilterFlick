"""
FilterFlick — Camera Manager
Handles webcam capture and dual-resolution frame delivery.
"""

import cv2
import numpy as np
from config import CAMERA_INDEX, RESOLUTION, PROCESSING_SCALE


class CameraManager:
    """Manages webcam lifecycle and produces full + half-scale frames."""

    def __init__(self, index: int = CAMERA_INDEX, resolution: tuple = RESOLUTION):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self._scale = PROCESSING_SCALE

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {index}. "
                "Check that a webcam is connected."
            )

        # Read actual resolution (camera may not support requested)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened at {actual_w}x{actual_h}")

    def read_frame(self) -> tuple:
        """
        Capture a single frame from the webcam.

        Returns
        -------
        (full_frame, processing_frame) : tuple[np.ndarray, np.ndarray]
            full_frame       — original resolution (for display & overlay)
            processing_frame — half-scale (for MediaPipe inference)
            Returns (None, None) if capture fails.
        """
        ret, full_frame = self._cap.read()
        if not ret or full_frame is None:
            return None, None

        # Flip horizontally for mirror effect (natural selfie view)
        full_frame = cv2.flip(full_frame, 1)

        # Create half-scale frame for faster MediaPipe processing
        h, w = full_frame.shape[:2]
        proc_w = int(w * self._scale)
        proc_h = int(h * self._scale)
        processing_frame = cv2.resize(
            full_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA
        )

        return full_frame, processing_frame

    def is_opened(self) -> bool:
        """Check if the camera is still open."""
        return self._cap.isOpened()

    def release(self):
        """Release camera resources."""
        if self._cap.isOpened():
            self._cap.release()
            print("[Camera] Released.")

    def __del__(self):
        self.release()
