"""
FilterFlick — Main Application
Real-time webcam face filter application with hand gesture controls.

Integrates all modules:
  - Camera capture (dual resolution)
  - Face detection (MediaPipe Face Mesh, up to 2 faces)
  - Hand gesture recognition (MediaPipe Hands, 5 gestures)
  - Filter engine (RGBA overlay with rotation & alpha blending)
  - HUD overlay (FPS, filter, gesture, recording indicator)
  - Keyboard controls (quit, screenshot, recording, next filter)
"""

import sys
import time
import logging
from collections import deque

import cv2

from config import PROCESSING_SCALE, FILTER_ORDER
from modules.camera import CameraManager
from modules.face_detector import FaceDetector
from modules.hand_gesture import GestureDetector
from modules.filter_engine import FilterEngine
from modules.hud import HUD
from modules.controls import ControlHandler, Action

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("FilterFlick")


def main():
    """Main application loop — full integration of all modules."""

    # ── Initialize modules ──────────────────────
    logger.info("Initializing FilterFlick...")

    try:
        camera = CameraManager()
    except RuntimeError as e:
        logger.error(f"Camera initialization failed: {e}")
        sys.exit(1)

    face_detector = FaceDetector()
    gesture_detector = GestureDetector()
    filter_engine = FilterEngine()
    hud = HUD()
    controls = ControlHandler()

    # ── State ───────────────────────────────────
    filter_index = 0
    active_filter = FILTER_ORDER[filter_index]
    last_gesture = "None"

    # FPS tracking (rolling average over 30 frames)
    frame_times = deque(maxlen=30)
    fps = 0.0

    logger.info("FilterFlick ready!")
    logger.info("Controls: Q=Quit | S=Screenshot | R=Record | N=Next Filter")
    logger.info("Gestures: Peace=Next | Thumbs Up=Sunglasses | Fist=Remove | Open Palm=Dog | One Finger=Crown")

    # ── Main loop ───────────────────────────────
    try:
        while camera.is_opened():
            t_start = time.perf_counter()

            # ─── 1. Capture frame ──────────────────
            full_frame, proc_frame = camera.read_frame()
            if full_frame is None:
                logger.warning("Frame capture failed — retrying...")
                continue

            # ─── 2. Face detection (half-scale) ────
            faces = face_detector.detect(proc_frame)

            # ─── 3. Hand gesture detection (half-scale) ──
            gesture = gesture_detector.detect(proc_frame)

            if gesture and gesture_detector.check_cooldown():
                last_gesture = gesture.name

                # Apply gesture action
                if gesture.action == "next":
                    filter_index = (filter_index + 1) % len(FILTER_ORDER)
                    active_filter = FILTER_ORDER[filter_index]
                    logger.info(f"[Gesture: {gesture.name}] → Next filter: {active_filter}")
                elif gesture.action == "none":
                    active_filter = "none"
                    filter_index = 0
                    logger.info(f"[Gesture: {gesture.name}] → Filter removed")
                elif gesture.action in FILTER_ORDER:
                    active_filter = gesture.action
                    filter_index = FILTER_ORDER.index(active_filter)
                    logger.info(f"[Gesture: {gesture.name}] → Filter: {active_filter}")

                gesture_detector.reset_cooldown()

            # ─── 4. Apply filter to each face (full-scale) ──
            for face in faces:
                # Scale coordinates: processing → display resolution
                face.scale(1.0 / PROCESSING_SCALE)
                try:
                    full_frame = filter_engine.apply(full_frame, face, active_filter)
                except Exception as e:
                    # Edge case: if filter overlay fails for any reason, skip gracefully
                    logger.debug(f"Filter overlay error (skipping): {e}")

            # ─── 5. Draw HUD ──────────────────────
            hud.draw(
                full_frame,
                fps=fps,
                active_filter=active_filter,
                last_gesture=last_gesture,
                face_count=len(faces),
                is_recording=controls.is_recording,
            )

            # ─── 6. Write frame to recording if active ──
            if controls.is_recording:
                controls.write_frame(full_frame)

            # ─── 7. Display ───────────────────────
            cv2.imshow("FilterFlick", full_frame)

            # ─── 8. Handle keyboard controls ──────
            action = controls.check_key()
            if action == Action.QUIT:
                logger.info("Quit requested.")
                break
            elif action == Action.SCREENSHOT:
                path = controls.save_screenshot(full_frame)
                logger.info(f"Screenshot saved: {path}")
            elif action == Action.TOGGLE_RECORDING:
                h, w = full_frame.shape[:2]
                controls.toggle_recording((h, w))
            elif action == Action.NEXT_FILTER:
                filter_index = (filter_index + 1) % len(FILTER_ORDER)
                active_filter = FILTER_ORDER[filter_index]
                logger.info(f"[Key: N] → Filter: {active_filter}")

            # ─── 9. FPS calculation ───────────────
            t_end = time.perf_counter()
            frame_times.append(t_end - t_start)
            if len(frame_times) > 1:
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0.0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    # ── Cleanup ─────────────────────────────────
    logger.info("Shutting down...")
    controls.release()
    camera.release()
    face_detector.release()
    gesture_detector.release()
    cv2.destroyAllWindows()
    logger.info("FilterFlick shut down cleanly.")


if __name__ == "__main__":
    main()
