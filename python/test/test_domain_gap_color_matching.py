import cv2
import numpy as np
from skimage import exposure

# Load the images
image_path = '/home/jjiao/Pictures/test_img.png'
image = cv2.imread(image_path)

# Split the image into left and right parts
height, width, _ = image.shape
left_image = image[:, :width // 2]
right_image = image[:, width // 2:]

# Convert the images to the LAB color space
left_lab = cv2.cvtColor(left_image, cv2.COLOR_BGR2LAB)
right_lab = cv2.cvtColor(right_image, cv2.COLOR_BGR2LAB)

# Perform histogram matching on the LAB color channels
left_matched = np.empty_like(left_lab)
for i in range(3):
    left_matched[:, :, i] = exposure.match_histograms(left_lab[:, :, i], right_lab[:, :, i])

# Convert the matched image back to BGR color space
left_matched_bgr = cv2.cvtColor(left_matched, cv2.COLOR_LAB2BGR)

# Combine the left and right images back together
combined_image = np.hstack((left_matched_bgr, right_image))

# Save the resulting image
output_path = '/home/jjiao/Pictures/left_image_matched.png'
cv2.imwrite(output_path, combined_image)

# Display the original and the color matched image
cv2.imshow("Original Image", image)
cv2.imshow("Color Matched Image", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
