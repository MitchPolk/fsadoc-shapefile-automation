import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image and convert to HSV (already done in your step 1)
img = cv2.imread("FSA_Plot_CROPPED.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the yellow color range in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the HSV image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Step 1: Dilate the yellow areas to thicken the lines
dilation_kernel = np.ones((25, 25), np.uint8)  # Use a 5x5 kernel, adjust based on the image
mask_dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

# Step 2: Apply morphological closing to reconnect the boundaries
close_kernel = np.ones((100, 100), np.uint8)
mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, close_kernel)

# Step 3: Erode the image to thin the lines back to their original size
erode_kernel = np.ones((25, 25), np.uint8)
mask_thinned = cv2.erode(mask_closed, erode_kernel, iterations=1)

# Optional: Clean small noise after these operations
clean_kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_thinned, cv2.MORPH_OPEN, clean_kernel)

# Convert images to RGB for matplotlib compatibility (except for the mask images)
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask_dilated_rgb = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2RGB)
mask_closed_rgb = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2RGB)
mask_thinned_rgb = cv2.cvtColor(mask_thinned, cv2.COLOR_GRAY2RGB)
mask_cleaned_rgb = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2RGB)

# Create a 2-row by 3-column subplot
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# List of images and their titles
images = [hsv_rgb, mask_rgb, mask_dilated_rgb, mask_closed_rgb, mask_thinned_rgb, mask_cleaned_rgb]
titles = ['HSV Image', 'Mask (Yellow)', 'Dilated Mask', 'Closed Mask', 'Thinned Mask', 'Cleaned Mask']

# Plot each image in the subplot grid
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(titles[i])
    ax.axis('off')  # Hide axis

# Display the plot
plt.tight_layout()
plt.show()