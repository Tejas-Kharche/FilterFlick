"""
FilterFlick — Hand Gesture Recognition
Uses MediaPipe Hands to classify hand poses into filter-switching gestures.
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    GESTURE_CONFIDENCE_THRESHOLD,
    GESTURE_COOLDOWN,
)


@dataclass
class GestureResult:
    """Result of a hand gesture classification."""
    name: str           # "peace", "thumbs_up", "fist", "open_palm", "one_finger"
    confidence: float   # 0.0 – 1.0
    action: str         # "next", "sunglasses", "none", "dog", "crown"


# ──────────────────────────────────────────────
# MediaPipe Hand Landmark Indices
# ──────────────────────────────────────────────
# Thumb
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2

# Index finger
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5

# Middle finger
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9

# Ring finger
RING_TIP = 16
RING_PIP = 14
RING_MCP = 13

# Pinky
PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17

# Wrist
WRIST = 0

# ──────────────────────────────────────────────
# Gesture → Action mapping
# ──────────────────────────────────────────────
GESTURE_MAP = {
    "peace":      "next",        # ✌️  → cycle to next filter
    "thumbs_up":  "sunglasses",  # 👍 → sunglasses
    "fist":       "none",        # ✊  → remove filter
    "open_palm":  "dog",         # 🖐️ → dog filter
    "one_finger": "crown",       # ☝️  → crown filter
    "rock":       "mask",        # 🤘 → mask filter
}


class GestureDetector:
    """Detects hand gestures using MediaPipe Hands and rule-based finger logic."""

    def __init__(self, max_hands: int = MAX_HANDS):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        self._last_gesture_time = 0.0

    def detect(self, frame: np.ndarray) -> Optional[GestureResult]:
        """
        Detect a hand gesture in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (processing-scale resolution).

        Returns
        -------
        Optional[GestureResult]
            Classified gesture with confidence, or None if no hand detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None

        best_result = None
        highest_confidence = 0.0

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness_info = results.multi_handedness[idx]
            handedness = handedness_info.classification[0].label  # "Left" or "Right"
            mp_confidence = handedness_info.classification[0].score

            # Classify gesture
            gesture_name, gesture_confidence = self._classify(
                hand_landmarks.landmark, handedness
            )

            if gesture_name is None:
                continue

            # Combine MediaPipe confidence with gesture clarity
            combined_confidence = mp_confidence * gesture_confidence

            if combined_confidence >= GESTURE_CONFIDENCE_THRESHOLD and combined_confidence > highest_confidence:
                highest_confidence = combined_confidence
                action = GESTURE_MAP.get(gesture_name, "none")
                best_result = GestureResult(
                    name=gesture_name,
                    confidence=combined_confidence,
                    action=action,
                )

        return best_result

    def check_cooldown(self) -> bool:
        """Check if enough time has passed since the last gesture trigger."""
        return (time.time() - self._last_gesture_time) >= GESTURE_COOLDOWN

    def reset_cooldown(self):
        """Reset the cooldown timer (call after a gesture is acted upon)."""
        self._last_gesture_time = time.time()

    def _classify(self, landmarks, handedness: str) -> tuple:
        """
        Classify the hand pose into a named gesture.

        Returns
        -------
        (gesture_name, confidence) or (None, 0.0)
        """
        # Determine finger extension states
        thumb_ext = self._is_thumb_extended(landmarks, handedness)
        index_ext = self._is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        middle_ext = self._is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_ext = self._is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP)
        pinky_ext = self._is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        fingers = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        extended_count = sum(fingers)

        # ── Pattern matching ────────────────────

        # ✊ Fist — no fingers extended
        if extended_count == 0:
            return ("fist", 0.95)

        # 👍 Thumbs Up — only thumb extended
        if thumb_ext and extended_count == 1:
            # Extra validation: thumb tip should be above wrist
            if landmarks[THUMB_TIP].y < landmarks[WRIST].y:
                return ("thumbs_up", 0.90)
            else:
                return ("thumbs_up", 0.70)

        # ☝️ One Finger — only index extended
        if index_ext and not thumb_ext and not middle_ext and not ring_ext and not pinky_ext:
            return ("one_finger", 0.90)

        # 🤘 Rock on — index and pinky extended
        if index_ext and pinky_ext and not thumb_ext and not middle_ext and not ring_ext:
            return ("rock", 0.90)

        # ✌️ Peace — index + middle only (no thumb, no ring, no pinky)
        if index_ext and middle_ext and not thumb_ext and not ring_ext and not pinky_ext:
            return ("peace", 0.90)

        # 🖐️ Open Palm — all 5 fingers extended
        if extended_count == 5:
            return ("open_palm", 0.90)

        # Near-matches with reduced confidence
        if extended_count == 4 and not thumb_ext:
            return ("open_palm", 0.65)

        if index_ext and middle_ext and extended_count == 2:
            return ("peace", 0.65)

        # No clear gesture
        return (None, 0.0)

    @staticmethod
    def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """
        Check if a non-thumb finger is extended.
        A finger is extended if its tip Y-value is above (less than) its MCP Y-value.
        Additional check: tip should be above PIP for stronger classification.
        """
        tip_y = landmarks[tip_idx].y
        mcp_y = landmarks[mcp_idx].y
        pip_y = landmarks[pip_idx].y

        # Primary check: tip above MCP
        # Secondary check: tip above PIP (stricter)
        return tip_y < mcp_y and tip_y < pip_y

    @staticmethod
    def _is_thumb_extended(landmarks, handedness: str) -> bool:
        """
        Check if the thumb is extended.
        For right hand (mirrored in selfie view = "Left" label):
            thumb extended if tip X > IP X
        For left hand (mirrored = "Right" label):
            thumb extended if tip X < IP X
        """
        tip_x = landmarks[THUMB_TIP].x
        ip_x = landmarks[THUMB_IP].x
        mcp_x = landmarks[THUMB_MCP].x

        # Note: MediaPipe reports handedness from the camera's perspective,
        # but we flip the frame, so labels are mirrored.
        # "Left" label in mirrored view = user's right hand
        if handedness == "Left":
            # User's right hand — thumb points right
            return tip_x > ip_x and abs(tip_x - ip_x) > abs(tip_x - mcp_x) * 0.3
        else:
            # User's left hand — thumb points left
            return tip_x < ip_x and abs(tip_x - ip_x) > abs(tip_x - mcp_x) * 0.3

    def release(self):
        """Release MediaPipe resources."""
        self._hands.close()
