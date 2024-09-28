import cv2
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon

# Load the original image and convert to HSV (already done in your step 1)
img = cv2.imread("FSA_Plot_CROPPED.jpg")
height, width = img.shape[:2]  # Get image dimensions
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
# hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
# mask_dilated_rgb = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2RGB)
# mask_closed_rgb = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2RGB)
# mask_thinned_rgb = cv2.cvtColor(mask_thinned, cv2.COLOR_GRAY2RGB)
# mask_cleaned_rgb = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2RGB)

# Plot Image Processing Steps
# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
# images = [hsv_rgb, mask_rgb, mask_dilated_rgb, mask_closed_rgb, mask_thinned_rgb, mask_cleaned_rgb]
# titles = ['HSV Image', 'Mask (Yellow)', 'Dilated Mask', 'Closed Mask', 'Thinned Mask', 'Cleaned Mask']

# for i, ax in enumerate(axs.flat):
#     ax.imshow(images[i])
#     ax.set_title(titles[i])
#     ax.axis('off')  # Hide axis

# plt.tight_layout()
# plt.show()

# Find cleaned mask contours
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# GPS coordinates for known points (NW, SW, SE)
nw_gps = (-86.85730942572702, 40.54745748561711)
sw_gps = (-86.85731927962509, 40.54513029767603)
se_gps = (-86.8550136956354, 40.54515420546383)

# Define pixel coordinates of the corresponding corners in the image
all_points = np.concatenate(contours, axis=0)

# Find the extreme points (nw, sw, se)
nw_pixel = tuple(all_points[all_points[:, :, 0].argmin()][0])  # Furthest left (min x)
sw_pixel = (all_points[all_points[:, :, 0].argmin()][0][0], all_points[all_points[:, :, 1].argmax()][0][1])  # Furthest left and down (min x, max y)
se_pixel = tuple(all_points[all_points[:, :, 0].argmax()][0])  # Furthest right and down (max x, max y)

# Compute the affine transformation matrix using the corner pixels and GPS coordinates
src_pts = np.array([nw_pixel, sw_pixel, se_pixel], dtype="float32")
dst_pts = np.array([nw_gps, sw_gps, se_gps], dtype="float32")

# Use OpenCV’s getAffineTransform to get the transformation matrix
affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)

# Function to transform image points to GPS
def transform_point(point, affine_matrix):
    pt = np.array([point[0], point[1], 1.0])
    transformed_pt = np.dot(affine_matrix, pt)
    return transformed_pt[0], transformed_pt[1]

# Step 1: Loop through contours and transform each point from image to GPS coordinates
polygons = []
for contour in contours:
    # Transform each point in the contour
    gps_points = [transform_point(pt[0], affine_matrix) for pt in contour]
    # Create a polygon from the transformed points
    polygon = Polygon(gps_points)
    polygons.append(polygon)

# Step 2: Create a GeoDataFrame with the polygons
gdf = gpd.GeoDataFrame(geometry=polygons)

# Set the CRS (Coordinate Reference System) to WGS84 (EPSG:4326 for latitude/longitude)
gdf.set_crs("EPSG:4326", inplace=True)

# Step 3: Save the polygons as a shapefile
gdf.to_file("output_shapefile.shp")
