"""
FilterFlick — Face Detector
Wraps MediaPipe Face Mesh to extract structured facial landmark data.
"""

import math
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_FACES,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    LM_RIGHT_EYE,
    LM_LEFT_EYE,
    LM_NOSE_TIP,
    LM_FOREHEAD,
    LM_CHIN,
)


@dataclass
class FaceData:
    """Structured facial landmark data for a single detected face."""

    left_eye_center: tuple = (0, 0)
    right_eye_center: tuple = (0, 0)
    nose_tip: tuple = (0, 0)
    forehead_center: tuple = (0, 0)
    chin_point: tuple = (0, 0)
    eye_distance: float = 0.0
    head_tilt_angle: float = 0.0
    bbox: tuple = (0, 0, 0, 0)  # (x, y, w, h)

    def scale(self, factor: float):
        """
        Scale all pixel coordinates by a factor.
        Used to map half-resolution coords → full-resolution coords.
        """
        self.left_eye_center = (
            int(self.left_eye_center[0] * factor),
            int(self.left_eye_center[1] * factor),
        )
        self.right_eye_center = (
            int(self.right_eye_center[0] * factor),
            int(self.right_eye_center[1] * factor),
        )
        self.nose_tip = (
            int(self.nose_tip[0] * factor),
            int(self.nose_tip[1] * factor),
        )
        self.forehead_center = (
            int(self.forehead_center[0] * factor),
            int(self.forehead_center[1] * factor),
        )
        self.chin_point = (
            int(self.chin_point[0] * factor),
            int(self.chin_point[1] * factor),
        )
        self.eye_distance *= factor
        bx, by, bw, bh = self.bbox
        self.bbox = (
            int(bx * factor),
            int(by * factor),
            int(bw * factor),
            int(bh * factor),
        )


class FaceDetector:
    """Detects faces using MediaPipe Face Mesh and extracts key landmarks."""

    def __init__(self, max_faces: int = MAX_FACES):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in a BGR frame and extract landmark data.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (processing-scale resolution).

        Returns
        -------
        list[FaceData]
            One FaceData per detected face (up to max_faces).
        """
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance boost
        results = self._face_mesh.process(rgb_frame)

        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_data = self._extract_landmarks(face_landmarks, w, h)
                faces.append(face_data)

        return faces

    def _extract_landmarks(self, face_landmarks, frame_w: int, frame_h: int) -> FaceData:
        """Convert normalized MediaPipe landmarks to pixel-space FaceData."""
        lm = face_landmarks.landmark

        # Key landmark pixel coordinates
        left_eye = self._to_pixel(lm[LM_LEFT_EYE], frame_w, frame_h)
        right_eye = self._to_pixel(lm[LM_RIGHT_EYE], frame_w, frame_h)
        nose_tip = self._to_pixel(lm[LM_NOSE_TIP], frame_w, frame_h)
        forehead = self._to_pixel(lm[LM_FOREHEAD], frame_w, frame_h)
        chin = self._to_pixel(lm[LM_CHIN], frame_w, frame_h)

        # Eye distance (Euclidean)
        eye_dist = math.hypot(
            left_eye[0] - right_eye[0],
            left_eye[1] - right_eye[1],
        )

        # Head tilt angle (degrees)
        tilt_angle = math.degrees(
            math.atan2(
                left_eye[1] - right_eye[1],
                left_eye[0] - right_eye[0],
            )
        )

        # Bounding box from all landmarks
        bbox = self._compute_bbox(lm, frame_w, frame_h)

        return FaceData(
            left_eye_center=left_eye,
            right_eye_center=right_eye,
            nose_tip=nose_tip,
            forehead_center=forehead,
            chin_point=chin,
            eye_distance=eye_dist,
            head_tilt_angle=tilt_angle,
            bbox=bbox,
        )

    def _compute_bbox(self, landmarks, frame_w: int, frame_h: int) -> tuple:
        """Compute axis-aligned bounding box from all 468 landmarks."""
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    @staticmethod
    def _to_pixel(landmark, frame_w: int, frame_h: int) -> tuple:
        """Convert a normalized MediaPipe landmark to integer pixel coords."""
        return (int(landmark.x * frame_w), int(landmark.y * frame_h))

    def release(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()
