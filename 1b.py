#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "old_coins.png"  # Update this path if needed
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Create a mask for segmentation
mask = np.zeros_like(gray)

# Draw contours to segment coins
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Extract and store segmented coins
segmented_coins = []
for i, contour in enumerate(contours):
    # Get bounding box around the coin
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the coin region
    coin = image[y:y+h, x:x+w]
    
    # Store the segmented coin
    segmented_coins.append(coin)

# Display the results
plt.figure(figsize=(15, 5))

# Show Edge Detection and Contours
plt.subplot(1, 3, 1)
plt.title("Edge Detection")
plt.imshow(edges, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Detected Coins")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

# Show segmented coins
plt.subplot(1, 3, 3)
plt.title("Segmented Coins")
plt.axis("off")

# Display each segmented coin separately
fig, axes = plt.subplots(1, len(segmented_coins), figsize=(15, 5))
for ax, coin in zip(axes, segmented_coins):
    ax.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    ax.axis("off")

plt.show()

# %%
