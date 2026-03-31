import cv2
import numpy as np
import os

assets = {
    "sunglasses.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\sunglasses_v2_1774960719532.png",
    "dog_ears.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\dog_ears_v2_1774960736415.png",
    "dog_nose.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\dog_nose_v2_1774960759195.png",
    "dog_tongue.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\dog_tongue_v2_1774960779210.png",
    "crown.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\crown_v2_1774960796487.png",
    "mask.png": r"C:\Users\Tejas Yogesh Kharche\.gemini\antigravity\brain\d2876086-62d6-47ee-a417-6748ea2d56e1\mask_v2_1774960817256.png",
}

output_dir = r"c:\Users\Tejas Yogesh Kharche\FilterFlick\assets\filters"
os.makedirs(output_dir, exist_ok=True)

# #00FF00 is green = (0, 255, 0)
# In BGR it is (0, 255, 0)
# Let's use HSV for better tolerance assuming it might have anti-aliasing artifacts
# We can just define a strict green range or exact match as it's AI generated

for out_name, in_path in assets.items():
    print(f"Processing {in_path} -> {out_name}")
    img = cv2.imread(in_path)
    if img is None:
        print(f"Failed to load {in_path}")
        continue
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define green range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for green background
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert mask to get foreground
    fg_mask = cv2.bitwise_not(mask)
    
    # Smooth edges using morphological operations or GaussianBlur on mask
    fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
    
    # Convert original to BGRA
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel to our smoothed mask
    bgra[:, :, 3] = fg_mask
    
    # Let's completely clear the color where mask is 0 (optional, but cleaner)
    bgra[fg_mask < 10] = [0, 0, 0, 0]
    
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, bgra)
    print(f"Saved {out_path}")

print("Done processing assets.")
