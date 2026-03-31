import cv2
import glob

print('Cropping transparent padding from filter assets...')
for path in glob.glob('assets/filters/*.png'):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        continue
    
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
        cv2.imwrite(path, cropped)
        print(f'{path}: cropped from {img.shape[1]}x{img.shape[0]} to {w}x{h}')
        
print('Done.')
