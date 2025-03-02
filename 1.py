#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "old_coins.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Median Blur to reduce noise while keeping edges sharp
blurred = cv2.medianBlur(gray, 5)

# Adaptive Thresholding (Fine-tuned)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 15, 5)

# Apply Canny Edge Detection
edges = cv2.Canny(thresh, 100, 200)

# Morphological operations to close small holes
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Apply Distance Transform
dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(morph, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to filter contours based on area and shape
def filter_contours(contours, min_area=100, max_area=20000, circularity_threshold=0.5):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > circularity_threshold:
                filtered_contours.append(contour)
    return filtered_contours

# Apply contour filtering
filtered_contours = filter_contours(contours)

# Count total coins
total_coins = len(filtered_contours)

# Draw contours on the original image
output = image.copy()
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)

# Display the total coin count on the image
cv2.putText(output, f"Total Coins: {total_coins}", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Extract and store segmented coins
segmented_coins = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    coin = image[y:y+h, x:x+w]
    segmented_coins.append(coin)

# Display results
plt.figure(figsize=(15, 5))

# Adaptive Thresholding Output
plt.subplot(1, 4, 1)
plt.title("Adaptive Thresholding")
plt.imshow(thresh, cmap="gray")

# Canny Edge Detection Output
plt.subplot(1, 4, 2)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap="gray")

# Contour Detection Output
plt.subplot(1, 4, 3)
plt.title(f"Detected Coins: {total_coins}")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

# Segmented Coins
fig, axes = plt.subplots(1, len(segmented_coins), figsize=(15, 5))
for ax, coin in zip(axes, segmented_coins):
    ax.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    ax.axis("off")

plt.show()

# Print total count
print(f"Total number of coins detected: {total_coins}")

# %%
