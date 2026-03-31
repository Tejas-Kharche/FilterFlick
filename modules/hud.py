"""
FilterFlick — On-Screen HUD Renderer
Draws FPS, active filter name, and last detected gesture onto the video frame.
"""

import cv2
import numpy as np

from config import (
    HUD_FONT_SCALE,
    HUD_THICKNESS,
    HUD_COLOR_FPS,
    HUD_COLOR_FILTER,
    HUD_COLOR_GESTURE,
)

# Gesture emoji map for display
GESTURE_EMOJI = {
    "peace":      "Peace",
    "thumbs_up":  "Thumbs Up",
    "fist":       "Fist",
    "open_palm":  "Open Palm",
    "one_finger": "One Finger",
    "None":       "---",
}


class HUD:
    """Renders real-time information overlay on the video frame."""

    def __init__(self):
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._scale = HUD_FONT_SCALE
        self._thickness = HUD_THICKNESS
        self._shadow_color = (0, 0, 0)
        self._shadow_thickness = HUD_THICKNESS + 2

    def draw(
        self,
        frame: np.ndarray,
        fps: float,
        active_filter: str,
        last_gesture: str,
        face_count: int = 0,
        is_recording: bool = False,
    ):
        """
        Draw the full HUD overlay onto the frame (modifies in-place).

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on.
        fps : float
            Current frames per second.
        active_filter : str
            Name of the currently active filter.
        last_gesture : str
            Name of the last detected gesture.
        face_count : int
            Number of faces currently detected.
        is_recording : bool
            Whether video recording is active.
        """
        h, w = frame.shape[:2]

        # ── Top-left: FPS counter ───────────────
        self._draw_text(
            frame, f"FPS: {fps:.0f}",
            (10, 30), HUD_COLOR_FPS,
        )

        # ── Top-right: Face count ───────────────
        face_text = f"Faces: {face_count}"
        face_size = cv2.getTextSize(face_text, self._font, self._scale, self._thickness)[0]
        self._draw_text(
            frame, face_text,
            (w - face_size[0] - 10, 30), (255, 200, 0),
        )

        # ── Recording indicator ──────────────────
        if is_recording:
            # Red dot + "REC" in top center
            rec_text = "REC"
            rec_size = cv2.getTextSize(rec_text, self._font, self._scale, self._thickness)[0]
            rec_x = (w - rec_size[0]) // 2
            # Pulsing red circle
            cv2.circle(frame, (rec_x - 15, 25), 8, (0, 0, 255), -1)
            self._draw_text(frame, rec_text, (rec_x, 30), (0, 0, 255))

        # ── Bottom-left: Active filter ──────────
        filter_display = active_filter.replace("_", " ").title()
        self._draw_text(
            frame, f"Filter: {filter_display}",
            (10, h - 15), HUD_COLOR_FILTER,
        )

        # ── Bottom-right: Last gesture ──────────
        gesture_display = GESTURE_EMOJI.get(last_gesture, last_gesture)
        gesture_text = f"Gesture: {gesture_display}"
        gesture_size = cv2.getTextSize(
            gesture_text, self._font, self._scale, self._thickness
        )[0]
        self._draw_text(
            frame, gesture_text,
            (w - gesture_size[0] - 10, h - 15), HUD_COLOR_GESTURE,
        )

        # ── Bottom-center: Controls hint ────────
        hint = "Q=Quit  S=Screenshot  R=Record  N=Next"
        hint_size = cv2.getTextSize(hint, self._font, 0.4, 1)[0]
        hint_x = (w - hint_size[0]) // 2
        self._draw_text(
            frame, hint,
            (hint_x, h - 40), (150, 150, 150),
            scale=0.4, thickness=1,
        )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: tuple,
        color: tuple,
        scale: float = None,
        thickness: int = None,
    ):
        """Draw text with a dark shadow for readability on any background."""
        s = scale or self._scale
        t = thickness or self._thickness

        # Shadow (offset by 1px)
        cv2.putText(
            frame, text, pos,
            self._font, s, self._shadow_color,
            t + 2, cv2.LINE_AA,
        )
        # Foreground
        cv2.putText(
            frame, text, pos,
            self._font, s, color,
            t, cv2.LINE_AA,
        )
