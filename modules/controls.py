"""
FilterFlick — Keyboard Controls & Recording
Handles keyboard input, screenshot capture, and video recording.
"""

import os
import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from config import OUTPUT_DIR, RECORDING_FPS, RECORDING_CODEC, RESOLUTION


class Action(Enum):
    """Possible keyboard actions."""
    QUIT = auto()
    SCREENSHOT = auto()
    TOGGLE_RECORDING = auto()


# Key → Action mapping
KEY_MAP = {
    ord('q'): Action.QUIT,
    ord('Q'): Action.QUIT,
    ord('s'): Action.SCREENSHOT,
    ord('S'): Action.SCREENSHOT,
    ord('r'): Action.TOGGLE_RECORDING,
    ord('R'): Action.TOGGLE_RECORDING,
}


class ControlHandler:
    """Handles keyboard input, screenshots, and video recording."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self._output_dir = output_dir
        self._ensure_output_dir()

        # Recording state
        self._is_recording = False
        self._video_writer = None
        self._recording_path = None

    @property
    def is_recording(self) -> bool:
        """Whether video recording is currently active."""
        return self._is_recording

    def check_key(self) -> Optional[Action]:
        """
        Poll for a key press and return the corresponding action.

        Returns
        -------
        Optional[Action]
            The action to perform, or None if no relevant key was pressed.
        """
        key = cv2.waitKey(1) & 0xFF
        return KEY_MAP.get(key)

    def save_screenshot(self, frame: np.ndarray) -> str:
        """
        Save the current frame as a timestamped PNG screenshot.

        Returns
        -------
        str
            Path to the saved screenshot.
        """
        self._ensure_output_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join(self._output_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def toggle_recording(self, frame_shape: tuple = None):
        """
        Toggle video recording on/off.

        Parameters
        ----------
        frame_shape : tuple
            (height, width) of the frame — required when starting recording.
        """
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording(frame_shape)

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the active video recording."""
        if self._is_recording and self._video_writer is not None:
            # Ensure frame matches expected resolution
            h, w = frame.shape[:2]
            expected_w, expected_h = RESOLUTION
            if w != expected_w or h != expected_h:
                frame = cv2.resize(frame, (expected_w, expected_h))
            self._video_writer.write(frame)

    def release(self):
        """Release recording resources."""
        if self._is_recording:
            self._stop_recording()

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _start_recording(self, frame_shape: tuple = None):
        """Start video recording."""
        self._ensure_output_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_{timestamp}.avi"
        self._recording_path = os.path.join(self._output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        width, height = RESOLUTION

        self._video_writer = cv2.VideoWriter(
            self._recording_path, fourcc, RECORDING_FPS,
            (width, height),
        )

        if self._video_writer.isOpened():
            self._is_recording = True
            print(f"[Controls] Recording started: {self._recording_path}")
        else:
            print("[Controls] ERROR: Could not start recording.")
            self._video_writer = None

    def _stop_recording(self):
        """Stop video recording and release writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._is_recording = False
        print(f"[Controls] Recording saved: {self._recording_path}")

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self._output_dir, exist_ok=True)
