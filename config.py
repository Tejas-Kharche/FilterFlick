"""
FilterFlick — Central Configuration
All constants and tunable parameters live here.
"""

import os

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
RESOLUTION = (640, 480)            # Display resolution (width, height)
PROCESSING_SCALE = 0.5             # Internal processing at half resolution

# ──────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────
MAX_FACES = 2
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
GESTURE_CONFIDENCE_THRESHOLD = 0.6

# ──────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────
GESTURE_COOLDOWN = 1.2             # Seconds between gesture triggers

# ──────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────
FILTER_ORDER = ["none", "sunglasses", "dog", "crown", "mask"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILTER_ASSETS_DIR = os.path.join(BASE_DIR, "assets", "filters")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Scale multipliers relative to eye_distance (or bbox_width for mask)
FILTER_SCALES = {
    "sunglasses": 2.5,
    "dog_ears":   3.0,
    "dog_nose":   1.2,
    "dog_tongue": 1.0,
    "crown":      2.8,
    "mask":       0.9,    # relative to bbox width, not eye_distance
}

# Vertical offset multipliers (relative to eye_distance)
# Positive = move downward, Negative = move upward
FILTER_OFFSETS_Y = {
    "sunglasses": 0.0,
    "dog_ears":  -1.0,
    "dog_nose":   0.0,
    "dog_tongue": 0.6,
    "crown":     -1.4,
    "mask":       0.0,
}

# ──────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────
HUD_FONT_SCALE = 0.7
HUD_THICKNESS = 2
HUD_COLOR_FPS = (0, 255, 0)       # Green
HUD_COLOR_FILTER = (255, 255, 255) # White
HUD_COLOR_GESTURE = (255, 255, 0)  # Cyan (BGR)

# ──────────────────────────────────────────────
# Recording
# ──────────────────────────────────────────────
RECORDING_FPS = 20.0
RECORDING_CODEC = "XVID"

# ──────────────────────────────────────────────
# Face Mesh Landmark Indices
# ──────────────────────────────────────────────
LM_RIGHT_EYE = 33
LM_LEFT_EYE = 263
LM_NOSE_TIP = 1
LM_FOREHEAD = 10
LM_CHIN = 152
