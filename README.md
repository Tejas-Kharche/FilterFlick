# FilterFlick 📸

FilterFlick is a real-time, lightweight webcam face filter application powered by computer vision. It allows users to control and switch between dynamic facial filters using **hand gestures**, supporting up to two concurrent faces. 

Built using Python, OpenCV, and MediaPipe, the application runs a dual-resolution pipeline to guarantee 20–30 FPS on standard CPUs while delivering high-quality visual overlays.

---

## ✨ Features

- **Real-Time Face Tracking:** Maps 468 facial landmarks per face using MediaPipe Face Mesh.
- **Hand Gesture Controls:** Recognize 5 different hand gestures via MediaPipe Hands to cycle or trigger specific filters.
- **Alpha-Blended Overlays:** Vectorized NumPy blending ensures smooth, transparent PNG rendering without pixel loops.
- **Dynamic Anchoring & Scaling:** Filters scale automatically based on distance between the user's eyes and tilt based on head rotation.
- **Dual-Face Support:** Applies independent filters and rotations to up to two people in the frame simultaneously.
- **HUD & Media Controls:** Real-time Heads-Up Display showing FPS and states. Supports screenshotting and AVI video recording.

---

## 📂 Folder Structure

```text
FilterFlick/
├── main.py                 # Application loop orchestrator
├── config.py               # Centralized constants, scales, and offsets
├── requirements.txt        # Pinned Python dependencies
├── crop_assets.py          # Utility script to strip transparent padding
├── process_assets.py       # Utility script to chroma-key green backgrounds
│
├── modules/                # Core application logic
│   ├── __init__.py         
│   ├── camera.py           # Webcam capture & resolution scaling
│   ├── face_detector.py    # MediaPipe Face Mesh extraction logic
│   ├── hand_gesture.py     # MediaPipe Hands classification logic
│   ├── filter_engine.py    # PNG loading, rotation, scaling, & overlay pipeline
│   ├── hud.py              # On-screen HUD drawing (FPS, filters, gestures)
│   └── controls.py         # Keyboard event handling & VideoWriter
│
├── assets/                 
│   └── filters/            # Cropped, minimalist RGBA PNG filter graphics
│       ├── crown.png
│       ├── dog_ears.png
│       ├── dog_nose.png
│       ├── dog_tongue.png
│       ├── mask.png
│       └── sunglasses.png
│
└── output/                 # Destination for screenshots and recordings
```

---

## 🧩 Architecture & Approach

The project adheres to a highly modular architecture where each component is isolated with a single responsibility. 

### Performance Approach: Dual-Resolution Processing
Machine Learning inference (MediaPipe) is computationally expensive. To maintain 20-30 FPS, the `camera.py` module produces two frames per capture:
1. **Processing Frame (320x240):** Passed to MediaPipe for face and hand landmark extraction.
2. **Display Frame (640x480):** Used by `filter_engine.py` to alpha-blend the high-resolution PNG filters. The extracted landmarks are mathematically scaled back up to match this display frame.

### 1. `modules/camera.py`
Initializes `cv2.VideoCapture` and reads frames synchronously. Flips the frame horizontally to provide a natural "selfie" view. 

### 2. `modules/face_detector.py`
Wraps `mediapipe.solutions.face_mesh`. Extracts 468 landmarks and distils them down into a lightweight `FaceData` dataclass containing just the points we need: `left_eye_center`, `right_eye_center`, `nose_tip`, `forehead`, and `chin`. It computes head tilt angle and Euclidean eye distance for scaling.

### 3. `modules/hand_gesture.py`
Wraps `mediapipe.solutions.hands`. It applies custom rule-based heuristics to determine which fingers are extended (checking if Finger Tip Y < Finger MCP Y). Crucially, the thumb uses an X-axis check that is *handedness-aware*, meaning it correctly identifies a thumbs-up whether you use your left or right hand. Includes a 1.2s cooldown timer to prevent rapid, flickering filter switches.

### 4. `modules/filter_engine.py`
The most complex module. Uses vectorized matrix multiplication in NumPy to perform RGBA alpha blending directly onto the camera frame, bypassing slow standard `cv2.addWeighted` setups or pixel loops. It dynamically calculates an affine rotation matrix to match head tilt and uses bounding-box intersection calculations to ensure filters that clip off the edge of the screen don't crash the application.

### 5. `modules/hud.py` & `modules/controls.py`
Manages the user interface. `hud.py` draws shadow-dropped text overlaid on the camera feed. `controls.py` listens to `cv2.waitKey` and marshals requests to write `cv2.VideoWriter` frames or save `.png` snapshots.

---

## ✋ Supported Gestures

Simply hold your hand up to the camera to trigger these actions:

| Gesture | Action Triggered |
|---------|-----------------|
| ✌️ **Peace Sign** | Cycle to the **next filter** |
| 👍 **Thumbs Up** | Equip the **Sunglasses** |
| 🖐️ **Open Palm** | Equip the **Dog** filter (Ears, Nose, and Tongue) |
| ☝️ **One Finger** | Equip the **Crown** |
| ✊ **Closed Fist** | **Clear/Remove** active filter |

---

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Tejas-Kharche/FilterFlick.git
cd FilterFlick
```

**2. Create and activate a Virtual Environment**
> MediaPipe has known compilation issues with Python 3.12+. Use **Python 3.10 or 3.11**.
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Application**
```bash
python main.py
```

### Keyboard Fallbacks
If you don't want to use hand gestures, you can use your keyboard:
- `N`: Cycle to Next Filter
- `S`: Save a Screenshot
- `R`: Start / Stop Video Recording
- `Q`: Quit Application
