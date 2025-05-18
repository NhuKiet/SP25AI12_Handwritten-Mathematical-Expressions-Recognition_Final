import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# Load image
input_path = 'testdoan.jpeg'
img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deskew
coords = np.column_stack(np.where(gray < 255))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = gray.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Adaptive threshold
th = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35, 2
)

# Morphological cleaning
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

# Skeletonize
skeleton = skeletonize(clean.astype(bool))
skeleton = (skeleton * 255).astype(np.uint8)

# Dilation for uniform stroke
stroke_width = 2
sw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width, stroke_width))
printed = cv2.dilate(skeleton, sw_kernel, iterations=1)

# Smooth edges
blur = cv2.GaussianBlur(printed, (3, 3), 0)
_, final = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# Display original and processed
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Grayscale')
plt.axis('off')
plt.imshow(gray, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Processed (Printed-like)')
plt.axis('off')
plt.imshow(final, cmap='gray')

plt.show()
