📸 Functional Requirements
--------------------------

### 1\. Webcam Input

*   Capture real-time video using OpenCV (cv2.VideoCapture)
    
*   Target resolution: **640×480**
    
*   Maintain smooth performance of **20–30 FPS**
    
*   Process frames at 50% scale internally; display at full resolution
    

### 2\. Face Detection & Landmark Tracking

*   Use **MediaPipe Face Mesh** to detect up to **2 faces** simultaneously
    
*   Extract all **468 facial landmarks** per face
    
*   Compute and expose the following per detected face:
    
    *   left\_eye\_center, right\_eye\_center → pixel coordinates
        
    *   nose\_tip → pixel coordinates
        
    *   forehead\_center → pixel coordinates
        
    *   eye\_distance → pixel distance between eye centers (used for scaling)
        
    *   head\_tilt\_angle → angle in degrees between eye centers (used for rotation)
        
    *   bbox → bounding box (x, y, w, h) of the face
        

### 3\. Hand Gesture Recognition

*   Use **MediaPipe Hands** to detect **1 hand** at a time
    
*   Classify the following gestures using rule-based finger extension logic:
    

GestureFingers ExtendedAction Triggered✌️ PeaceIndex + Middle onlyNext filter👍 Thumbs UpThumb onlySunglasses filter✊ Closed FistNoneRemove filter🖐️ Open PalmAll 5Dog filter☝️ One FingerIndex onlyCrown filter

*   A finger is **extended** if its tip landmark Y-value is above its knuckle (MCP) Y-value
    
*   For the thumb: extended if tip X-value is beyond the IP joint (mirrored for left hand)
    
*   Check handedness label to correctly mirror thumb logic for left vs right hand
    
*   Return gesture name and a **confidence score** (0.0–1.0) — only act on gestures above a minimum threshold
    

### 4\. Gesture Cooldown

*   After any gesture triggers a filter change, enforce a **1.2-second cooldown**
    
*   During cooldown, ignore all incoming gestures
    
*   Prevents rapid flickering when holding a gesture pose
    

### 5\. Filter System

*   Support the following **4 filters** plus a **"none" state**:
    
    *   sunglasses — overlaid across both eyes
        
    *   dog — ears on forehead + nose overlay + tongue below chin
        
    *   crown — placed above forehead
        
    *   mask — covers lower half of face
        
*   Each filter must:
    
    *   Load from a **RGBA PNG** file (transparent background)
        
    *   **Resize dynamically** based on eye\_distance and a per-filter scale multiplier
        
    *   **Rotate** to match head\_tilt\_angle using affine transformation
        
    *   **Overlay** onto the frame using per-pixel alpha blending (vectorized NumPy — no pixel loops)
        

### 6\. Filter Placement Logic

Per filter, compute placement using face landmarks:

FilterAnchor PointScale MultiplierSunglassesMidpoint of both eye centerseye\_distance × 2.5Dog EarsForehead center (offset upward)eye\_distance × 3.0Dog NoseNose tipeye\_distance × 1.2CrownForehead center (offset further up)eye\_distance × 2.8MaskMidpoint of nose and chinbbox width × 0.9

*   Clamp all placement coordinates to **frame boundaries** before blending to prevent crashes on edge cases
    

### 7\. Multi-Face Support

*   Detect and track **up to 2 faces** simultaneously
    
*   Apply the **same active filter independently** to each detected face
    
*   Each face uses its own eye\_distance, head\_tilt\_angle, and anchor points — filters scale and rotate per-face
    

### 8\. On-Screen HUD

*   Display in real time on the video frame:
    
    *   **FPS counter** — top-left corner
        
    *   **Active filter name** — bottom-left corner
        
    *   **Last detected gesture** — bottom-right corner
        
*   HUD must not obstruct the face region
    

### 9\. Keyboard Controls

KeyActionQQuit the applicationSSave a screenshot to /output/ with a timestamp filenameRToggle video recording on/off (saves to /output/recorded.avi)NManually cycle to the next filter (fallback if gesture fails)

### 10\. Error & Edge Case Handling

*   **No face detected** → display webcam feed normally, no crash
    
*   **No hand detected** → active filter remains unchanged
    
*   **Partial face at frame edge** → clip overlay to frame bounds, render what's visible
    
*   **Filter PNG missing** → log a warning, skip that filter silently, do not crash