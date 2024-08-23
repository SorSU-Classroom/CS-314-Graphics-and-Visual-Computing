import cv2
import numpy as np

# Load the image
image = cv2.imread("../../../assets/images/fruits.jpg")

# Create a window to display the image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow("Image", image)

# Select the region of interest
# The region of interest is defined by the top-left corner and the bottom-right corner
# The top-left corner is (100, 100) and the bottom-right corner is (500, 500)
roi = image[100:500, 100:500]

# Display the region of interest
cv2.imshow("ROI", roi)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()