'''
Downsample and Concatenate Operators

Downsampling
- is the process of reducing the resolution of an image. This is done by
removing pixels from the image. The downsample operator is used to reduce
the resolution of an image by a factor of 2. The concatenate operator is
used to concatenate two images along the horizontal or vertical axis.
'''

import cv2
import numpy as np

def display_image(image, title="Image", x=512, y=512):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, x, y)
    cv2.imshow(title, image)

# Load the image
image = cv2.imread("../../../assets/images/fruits.jpg")

# Downsample the image by a factor of resize_factor
resize_factor = 16
downsampled_image = image[::resize_factor, ::resize_factor]

# Resize the downsampled image to the original image size but no blur
resized_image = cv2.resize(downsampled_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Print the dimensions of the original, downsampled and concatenated images
print("Original Image Shape:", image.shape)
print("Downsampled Image Shape:", downsampled_image.shape)
print("Resized Image Shape:", resized_image.shape)

# Divide the Resized Image into four sub-images of same shape
sub_image1 = resized_image[0:resized_image.shape[0]//2, 0:resized_image.shape[1]//2]
sub_image2 = resized_image[0:resized_image.shape[0]//2, resized_image.shape[1]//2:]
sub_image3 = resized_image[resized_image.shape[0]//2:, 0:resized_image.shape[1]//2]
sub_image4 = resized_image[resized_image.shape[0]//2:, resized_image.shape[1]//2:]

# Concatenate the sub-images along the horizontal axis
concatenated_image1 = np.concatenate((sub_image4, sub_image1), axis=1)
concatenated_image2 = np.concatenate((sub_image3, sub_image2), axis=1)
concatenated_image = np.concatenate((concatenated_image1, concatenated_image2), axis=0)

# Display the original and downsampled images
window_size = 512
display_image(image, "Original Image")
display_image(downsampled_image, "Downsampled Image")
display_image(concatenated_image, "Concatenated Image", window_size, window_size)

# Wait until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()