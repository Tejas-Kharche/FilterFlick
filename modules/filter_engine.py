"""
FilterFlick — Filter Engine
Loads, transforms (resize/rotate), and overlays RGBA filter assets onto faces.
"""

import os
import logging

import cv2
import numpy as np

from config import (
    FILTER_ASSETS_DIR,
    FILTER_SCALES,
    FILTER_OFFSETS_Y,
)

logger = logging.getLogger(__name__)


class FilterEngine:
    """Handles filter loading, transformation, and alpha-blended overlay."""

    # Maps filter names to their PNG filenames
    _FILTER_FILES = {
        "sunglasses": ["sunglasses.png"],
        "dog":        ["dog_ears.png", "dog_nose.png", "dog_tongue.png"],
        "crown":      ["crown.png"],
        "mask":       ["mask.png"],
    }

    def __init__(self, assets_dir: str = FILTER_ASSETS_DIR):
        self._assets_dir = assets_dir
        self._cache = {}  # name -> np.ndarray (RGBA)
        self._load_all()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def apply(self, frame: np.ndarray, face_data, filter_name: str) -> np.ndarray:
        """
        Apply the named filter onto the frame using the given face data.

        Parameters
        ----------
        frame : np.ndarray
            BGR image at full resolution (will be modified in-place).
        face_data : FaceData
            Landmark data for one face (already scaled to full resolution).
        filter_name : str
            One of: "sunglasses", "dog", "crown", "mask", "none".

        Returns
        -------
        np.ndarray
            The frame with filter overlaid.
        """
        if filter_name == "none":
            return frame

        if filter_name == "dog":
            # Composite filter — 3 parts
            frame = self._apply_piece(frame, face_data, "dog_ears")
            frame = self._apply_piece(frame, face_data, "dog_nose")
            frame = self._apply_piece(frame, face_data, "dog_tongue")
        else:
            frame = self._apply_piece(frame, face_data, filter_name)

        return frame

    # ──────────────────────────────────────────
    # Private — single filter piece pipeline
    # ──────────────────────────────────────────

    def _apply_piece(self, frame: np.ndarray, face, piece_name: str) -> np.ndarray:
        """Apply a single filter piece (e.g. 'dog_ears') onto the frame."""
        img = self._cache.get(piece_name)
        if img is None:
            return frame  # Missing asset — skip silently

        eye_dist = face.eye_distance
        if eye_dist < 5:
            return frame  # Face too small or unreliable

        # 1. Compute target size
        target_w, target_h = self._compute_size(img, piece_name, face)
        if target_w < 2 or target_h < 2:
            return frame

        # 2. Resize
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 3. Rotate to match head tilt
        rotated = self._rotate(resized, face.head_tilt_angle)

        # 4. Compute anchor position
        anchor_x, anchor_y = self._compute_anchor(face, piece_name, rotated.shape)

        # 5. Overlay with alpha blending
        frame = self._overlay_rgba(frame, rotated, anchor_x, anchor_y)

        return frame

    def _compute_size(self, img: np.ndarray, piece_name: str, face) -> tuple:
        """Compute target (width, height) for a filter piece."""
        orig_h, orig_w = img.shape[:2]
        aspect = orig_h / orig_w if orig_w > 0 else 1.0

        scale_mult = FILTER_SCALES.get(piece_name, 2.0)

        if piece_name == "mask":
            # Mask scales relative to bbox width
            target_w = int(face.bbox[2] * scale_mult)
        else:
            target_w = int(face.eye_distance * scale_mult)

        target_h = int(target_w * aspect)
        return target_w, target_h

    def _compute_anchor(self, face, piece_name: str, overlay_shape: tuple) -> tuple:
        """
        Compute top-left (x, y) anchor point for placing the overlay.
        The overlay is centered horizontally on the anchor landmark.
        """
        oh, ow = overlay_shape[:2]

        if piece_name == "sunglasses":
            # Center between both eyes
            cx = (face.left_eye_center[0] + face.right_eye_center[0]) // 2
            cy = (face.left_eye_center[1] + face.right_eye_center[1]) // 2

        elif piece_name == "dog_ears":
            cx = face.forehead_center[0]
            cy = face.forehead_center[1]

        elif piece_name == "dog_nose":
            cx = face.nose_tip[0]
            cy = face.nose_tip[1]

        elif piece_name == "dog_tongue":
            cx = face.chin_point[0]
            cy = face.chin_point[1]

        elif piece_name == "crown":
            cx = face.forehead_center[0]
            cy = face.forehead_center[1]

        elif piece_name == "mask":
            # Midpoint between nose and chin
            cx = (face.nose_tip[0] + face.chin_point[0]) // 2
            cy = (face.nose_tip[1] + face.chin_point[1]) // 2

        else:
            cx = face.nose_tip[0]
            cy = face.nose_tip[1]

        # Apply vertical offset
        offset_mult = FILTER_OFFSETS_Y.get(piece_name, 0.0)
        cy += int(face.eye_distance * offset_mult)

        # Convert center → top-left corner
        x = cx - ow // 2
        y = cy - oh // 2

        return x, y

    # ──────────────────────────────────────────
    # Private — transform utilities
    # ──────────────────────────────────────────

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an RGBA image by `angle` degrees around its center.
        Uses affine transformation, preserves transparency.
        """
        if abs(angle) < 0.5:
            return img  # Skip near-zero rotations

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Compute new bounding box size
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        # Adjust translation to keep image centered
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return rotated

    @staticmethod
    def _overlay_rgba(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Alpha-blend an RGBA overlay onto a BGR background at position (x, y).
        Uses vectorized NumPy — no pixel loops.
        Clamps to frame boundaries to handle edge cases.
        """
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]

        # Compute the region of overlap (clamp to frame bounds)
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + ov_w, bg_w)
        y2 = min(y + ov_h, bg_h)

        if x1 >= x2 or y1 >= y2:
            return background  # Completely outside frame

        # Corresponding region in the overlay
        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)

        # Extract overlay ROI
        overlay_roi = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

        # Split alpha channel (normalized 0-1)
        alpha = overlay_roi[:, :, 3:4].astype(np.float32) / 255.0
        rgb = overlay_roi[:, :, :3].astype(np.float32)

        # Extract background ROI
        bg_roi = background[y1:y2, x1:x2].astype(np.float32)

        # Alpha blend: result = alpha * fg + (1 - alpha) * bg
        blended = alpha * rgb + (1.0 - alpha) * bg_roi

        background[y1:y2, x1:x2] = blended.astype(np.uint8)

        return background

    # ──────────────────────────────────────────
    # Private — asset loading
    # ──────────────────────────────────────────

    def _load_all(self):
        """Pre-load all filter PNG files into memory."""
        all_files = set()
        for file_list in self._FILTER_FILES.values():
            all_files.update(file_list)

        for filename in all_files:
            name_key = filename.replace(".png", "")
            path = os.path.join(self._assets_dir, filename)

            if not os.path.exists(path):
                logger.warning(f"Filter asset missing: {path} — will be skipped.")
                continue

            # Load with alpha channel (IMREAD_UNCHANGED)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Failed to load filter: {path}")
                continue

            # Ensure 4 channels (RGBA)
            if img.shape[2] == 3:
                # Add opaque alpha channel if missing
                alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)

            self._cache[name_key] = img
            print(f"[FilterEngine] Loaded: {name_key} ({img.shape[1]}x{img.shape[0]})")

        print(f"[FilterEngine] {len(self._cache)} filter assets loaded.")
